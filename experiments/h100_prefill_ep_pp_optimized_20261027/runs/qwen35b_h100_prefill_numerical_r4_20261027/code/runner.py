"""Guard-owned full-model EP2/PP2 measurement and separate profiling."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import select
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

import process_helpers as h
from export_ready import await_exports

# Spawned scheduler ranks re-import this entrypoint; install the same private
# adapter there as well, before any model construction (not only in HTTP parent).
if os.environ.get('EP2_PP2_ARM'):
    import runtime_adapter
    runtime_adapter.install()


def check(run, cfg, env, local=False):
    return h.authenticate(run, cfg, env, local=local)


def progress(run, cfg, stage, **extra):
    metrics = list(Path(run).glob('*/metrics.*.json'))
    h.save(Path(run) / 'task/progress.json', dict(experiment_id=cfg['id'], task_id='comparison',
        stage=stage, timestamp=time.time(), controller_pid=os.getpid(),
        valid_points=len(metrics), expected=2 * len(cfg['layouts']), **extra))


def engine_args(cfg, arm):
    row = next(r for r in cfg['layouts'] if r['id'] == arm)
    args = ['--model-path', cfg['model_path'], '--host', '127.0.0.1', '--port', str(cfg['port']),
        '--dist-init-addr', '127.0.0.1:' + str(cfg['dist_port']), '--tp-size', str(row['tp']),
        '--ep-size', str(row['ep']), '--dp-size', str(row['dp']), '--pp-size', str(row['pp']),
        '--dtype', 'bfloat16', '--kv-cache-dtype', 'bfloat16',
        '--context-length', '8192', '--mem-fraction-static', '.90', '--page-size', '64',
        '--max-total-tokens', '49152', '--max-running-requests', '16',
        '--max-mamba-cache-size', '32', '--mamba-ssm-dtype', 'float32',
        # DP-attention resolves this global chunk budget by dividing DP size.
        # Freeze the same actual 4096-token budget per Attention instance.
        '--chunked-prefill-size', '8192' if arm == 'ep2' else '4096',
        '--max-prefill-tokens', '8192',
        '--attention-backend', 'triton', '--linear-attn-backend', 'triton',
        '--linear-attn-prefill-backend', 'triton', '--linear-attn-decode-backend', 'triton',
        '--moe-runner-backend', 'triton', '--moe-a2a-backend', 'deepep' if arm == 'ep2' else 'none',
        '--disable-cuda-graph', '--cuda-graph-backend-prefill', 'disabled',
        '--disable-overlap-schedule', '--disable-custom-all-reduce', '--skip-server-warmup',
        '--json-model-override-args', '{"language_model_only":true}']
    if arm == 'ep2':
        args += ['--enable-dp-attention', '--deepep-mode', 'normal',
                 '--deepep-dispatcher-output-dtype', 'bf16']
    else:
        args += ['--pp-max-micro-batch-size', '4', '--pp-async-batch-depth',
                 str(cfg.get('pp_async_batch_depth', 0))]
    return args


def engine_entry(run, cfg, arm):
    check(run, cfg, os.environ, local=True)
    import runtime_adapter
    runtime_adapter.install()
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.launch_server import run_server
    from sglang.srt.utils import kill_process_tree
    load_plugins()
    args = prepare_server_args(engine_args(cfg, arm))
    args.enable_multimodal = False
    args.api_key = args.admin_api_key = h.read_key(cfg)
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def environment(cfg, arm, inherited):
    prefixes = ('SGLANG_', 'VLLM_', 'DEEPEP_', 'NVSHMEM_', 'NCCL_', 'GLOO_', 'HUMMING_', 'QWEN_A100_')
    env = {k:v for k,v in inherited.items() if not k.startswith(prefixes)
           and k not in ('PYTHONPATH', 'LD_PRELOAD', 'CUDA_VISIBLE_DEVICES')}
    env.update({k:str(v) for k,v in cfg['launch_env'].items()})
    env.update(CUDA_VISIBLE_DEVICES='0,1', PYTHONDONTWRITEBYTECODE='1',
        CUDA_MODULE_LOADING='EAGER', OMP_NUM_THREADS='4', MKL_NUM_THREADS='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', SGLANG_PROFILE_V2='0',
        EP2_PP2_RUN=str(Path(cfg['code_root']).parent), EP2_PP2_ARM=arm,
        PYTHONPATH=os.pathsep.join([cfg['code_root'], str(Path(cfg['engine_root']) / 'python'),
            str(Path(cfg['deepep_native']).parent), cfg['deepep_wrapper']]))
    return env


def launch(run, cfg, arm, env, key):
    use = next(r for r in cfg['layouts'] if r['id'] == arm)['usage_logs'][0]
    command = [cfg['python'], str(Path(cfg['guard_scripts']) / 'model_cache_adapter.py'),
        'exec-local', '--bindings', env[h.GUARD_KEYS[3]], '--', cfg['python'],
        str(Path(__file__).resolve()), '--run', str(run), '--engine-entry', '--arm', arm]
    proc = subprocess.Popen(command, env=environment(cfg, arm, env), cwd=cfg['engine_root'],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    identity = h.process_record(proc.pid)
    if not identity or identity['pgid'] != proc.pid:
        proc.terminate()
        raise RuntimeError('Owned engine process group was not established')
    thread = threading.Thread(target=h.stream_log, args=(proc.stdout, use['path'], key), daemon=True)
    thread.start()
    return dict(process=proc, identity=identity, log_thread=thread)


def packet(cfg, arm, tokens, output, index):
    payload = dict(input_ids=tokens, sampling_params=dict(temperature=0,
        max_new_tokens=output, ignore_eos=True), stream=True)
    if arm == 'ep2':
        payload['data_parallel_rank'] = index % 2
    return payload


def stream_one(cfg, key, payload, gate):
    encoded = json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/generate", data=encoded,
        headers={'Authorization':'Bearer ' + key, 'Content-Type':'application/json'})
    gate.wait()
    start = time.perf_counter()
    first = last = None
    events = []
    with urllib.request.urlopen(req, timeout=300) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            value = json.loads(data)
            if 'error' in value:
                raise RuntimeError(str(value['error']))
            meta = value.get('meta_info', {})
            finish = meta.get('finish_reason')
            if isinstance(finish,dict) and finish.get('type') in ('abort','error'):
                raise RuntimeError('Native generation aborted: '+str(finish))
            count = meta.get('completion_tokens', 0)
            now = time.perf_counter()
            if count:
                first = now if first is None else first
                last = now
                events.append(dict(time_s=now-start, completion_tokens=count))
            final = value
    if first is None or last is None:
        raise RuntimeError('No generated token reached the stream')
    count = final['meta_info']['completion_tokens']
    if count != payload['sampling_params']['max_new_tokens']:
        raise RuntimeError('Output length differs from the fixed workload')
    return dict(ttft_s=first-start, e2e_s=last-start,
        tpot_ms=(last-first)*1000/(count-1) if count > 1 else None,
        output_tokens=count, started=start, finished=last,
        cached_tokens=final['meta_info'].get('cached_tokens'),
        prompt_tokens=final['meta_info'].get('prompt_tokens'), events=events,
        validation_logits={k:final.get('meta_info', {}).get(k) for k in ('output_token_logprobs','output_top_logprobs')}, output_ids=final.get('output_ids'), text=final.get('text', ''))


def cohort(cfg, key, payloads):
    gate = threading.Barrier(len(payloads))
    with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
        futures = [pool.submit(stream_one, cfg, key, value, gate) for value in payloads]
        rows = [f.result() for f in futures]
    duration = max(r['finished'] for r in rows) - min(r['started'] for r in rows)
    return dict(requests=rows, makespan_s=duration,
        output_tokens_per_s=sum(r['output_tokens'] for r in rows)/duration,
        input_tokens_per_s=sum(len(p['input_ids']) for p in payloads)/duration)


def flush_cache(cfg, key):
    # Native /flush_cache deliberately returns PlainTextResponse, not JSON.
    req=urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/flush_cache",
        headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(req,timeout=60) as response:
        body=response.read().decode()
        if response.status!=200 or not body.startswith('Cache flushed.'):
            raise RuntimeError('Native cache flush was not acknowledged: '+body[:200])
    return body


def profile_action(cfg, key, path, payload=None):
    """Native admin actions acknowledge success with text, never JSON."""
    expected = {'/start_profile': 'Start profiling.', '/stop_profile': 'Stop profiling.'}[path]
    req = urllib.request.Request(f"http://127.0.0.1:{cfg['port']}{path}",
        data=json.dumps(payload or {}).encode(), method='POST',
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        body = response.read().decode()
        if response.status != 200 or not body.startswith(expected):
            raise RuntimeError('Native profiler action was not acknowledged: ' + body[:200])
    return body


def summarize(rounds, workload):
    flat = [r for group in rounds for r in group['requests']]
    result = dict(workload=workload, rounds=rounds, valid=True, profiler_enabled=False)
    for metric in ('ttft_s','tpot_ms','e2e_s'):
        values = sorted(r[metric] for r in flat if r[metric] is not None)
        result[metric] = (dict(mean=statistics.mean(values), median=statistics.median(values),
                              p99=values[min(len(values)-1, int(.99*len(values)))]) if values else None)
    result['output_tokens_per_s'] = statistics.mean(r['output_tokens_per_s'] for r in rounds)
    result['input_tokens_per_s'] = statistics.mean(r['input_tokens_per_s'] for r in rounds)
    return result


def make_tokens(cfg):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
    result = []
    for i in range(cfg['requests']):
        start = tok.encode(f'Example {i}: Explain this technical document carefully.\n', add_special_tokens=False)
        body = tok.encode((f' Chapter {i}. A distributed system exchanges messages and executes work. '
                          'Independent requests have their own context and output.\n') * 1000,
                          add_special_tokens=False)
        result.append((start + body)[:cfg['prefix_tokens']])
    if any(len(r) != cfg['prefix_tokens'] for r in result):
        raise RuntimeError('Token construction did not cover the frozen length')
    return result


def run_workloads(run, cfg, arm, key, health):
    tokens = make_tokens(cfg)
    # Prefill uses a flushed cache. Tokenization, payload construction and warmup
    # are outside the client timing barrier, identical for the two layouts.
    pref = [packet(cfg, arm, t, 1, i) for i,t in enumerate(tokens)]
    decode = [packet(cfg, arm, t + [198], cfg['output_tokens'], i) for i,t in enumerate(tokens)]
    def flush():
        health()
        flush_cache(cfg, key)
    flush()
    cohort(cfg, key, pref)
    pre_rounds = []
    for repeat in range(cfg['repeats']):
        flush()
        measured = cohort(cfg, key, pref)
        if any((r['cached_tokens'] or 0) > 64 for r in measured['requests']):
            raise RuntimeError('Cold Prefill unexpectedly reused a prefix')
        pre_rounds.append(measured)
        h.save(run / arm / 'prefill.partial.json', dict(rounds=pre_rounds))
    h.save(run / arm / 'metrics.prefill.json', summarize(pre_rounds, '8 requests x 4096 fresh tokens -> 1 output'))
    # All prefixes are model-produced on the GPU; retain their native GDN state.
    # Each Decode round starts with the same context, not an ever-growing chat.
    flush()
    cohort(cfg, key, pref)
    warm = cohort(cfg, key, decode)
    dec_rounds = []
    for repeat in range(cfg['repeats']):
        health()
        measured = cohort(cfg, key, decode)
        if any((r['cached_tokens'] or 0) < cfg['prefix_tokens'] - 64 for r in measured['requests']):
            raise RuntimeError('Decode prefix did not restore from native GPU cache')
        dec_rounds.append(measured)
        h.save(run / arm / 'decode.partial.json', dict(rounds=dec_rounds))
    h.save(run / arm / 'metrics.decode.json', summarize(dec_rounds, '8 requests x 4096 cached tokens + 1 fresh token -> 128 outputs'))
    # Separate trace acquisition; these requests NEVER contribute to performance.
    for phase, payloads in [('prefill', pref), ('decode', decode)]:
        if phase == 'prefill':
            flush()
        out = run / arm / 'traces' / phase
        out.mkdir(parents=True, exist_ok=True)
        profile_action(cfg, key, '/start_profile', dict(output_dir=str(out),
            num_steps=5 if phase == 'decode' else None,
            activities=['CPU','GPU'], record_shapes=True,
            with_stack=False, profile_id=arm + '-' + phase, merge_profiles=False,
            detailed_annotations=True))
        profiled = cohort(cfg, key, payloads)
        h.save(out / 'cohort.json', dict(performance_sample=False, result=profiled))
        # Decode auto-stops after five forwards. Calling /stop_profile again
        # would correctly return HTTP 500 (no active profile). Prefill captures
        # the whole bounded eight-request cohort and is stopped explicitly.
        if phase == 'prefill':
            profile_action(cfg, key, '/stop_profile')
        exports = await_exports(out, health, timeout=180)
        h.save(out / 'manifest.json', dict(performance_sample=False, files=exports))



def worker(run, cfg):
    packet_in = json.loads(sys.stdin.readline())
    env = dict(os.environ, **packet_in['environment'])
    check(run, cfg, env, local=True)
    def interrupted(signum, frame):
        raise RuntimeError('Guard/controller requested owned worker cleanup')
    signal.signal(signal.SIGTERM, interrupted)
    def watch_parent():
        sys.stdin.read()
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=watch_parent, daemon=True).start()
    key = h.read_key(cfg)
    # One topology observation under the acquired lease; no extra preflight.
    topo = subprocess.run(['nvidia-smi','topo','-m'], text=True, capture_output=True, timeout=20)
    (run / 'topology.txt').write_text(topo.stdout + topo.stderr)
    def parent_health():
        check(run, cfg, env, local=True)
    for arm in cfg.get('arm_order', ['ep2', 'pp2']):
        child = None
        try:
            progress(run, cfg, 'loading_' + arm)
            child = launch(run, cfg, arm, env, key)
            deadline = time.monotonic() + cfg['startup_timeout_s']
            while True:
                parent_health()
                if child['process'].poll() is not None:
                    raise RuntimeError(f'{arm} exited during model startup')
                try:
                    info = h.http_json(cfg['port'], '/get_server_info', key, timeout=3)
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Full model startup exceeded the bounded deadline')
                    time.sleep(2)
            quals = list((run / arm / 'qualification').glob('*.json'))
            if len(quals) != 2 or not all(json.loads(p.read_text()).get('passed') for p in quals):
                raise RuntimeError('Both actual ranks must pass before measurement')
            h.save(run / arm / 'server_info.json', h.safe_server_info(info))
            req = urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/freeze_gc",
                headers={'Authorization': 'Bearer ' + key})
            with urllib.request.urlopen(req, timeout=60) as response:
                if not response.read().decode().startswith('Garbage collection frozen.'):
                    raise RuntimeError('Native GC freeze was not acknowledged')
            progress(run, cfg, 'measuring_' + arm)
            run_workloads(run, cfg, arm, key, parent_health)
        finally:
            if child:
                evidence = h.cleanup([child])
                h.save(run / arm / 'cleanup.json', evidence)
                if not evidence['cleanup_verified']:
                    raise RuntimeError('Owned engine cleanup failed verification')
    return 0


def controller(run, cfg):
    check(run, cfg, os.environ)
    packet_out = dict(environment={k:os.environ[k] for k in h.GUARD_KEYS})
    command = [cfg['python'], str(Path(__file__).resolve()), '--worker', '--run', str(run)]
    failure = None
    with (run / 'controller.log').open('ab', buffering=0) as log:
        proc = subprocess.Popen(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15',
            cfg['node'], shlex.join(command)], stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
        proc.stdin.write((json.dumps(packet_out) + '\n').encode()); proc.stdin.flush()
        try:
            while proc.poll() is None:
                check(run, cfg, os.environ)
                time.sleep(2)
            if proc.returncode:
                raise RuntimeError('Worker failed; inspect controller and exact rank loader logs')
        except BaseException as error:
            failure = type(error).__name__
            raise
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=85)
            except subprocess.TimeoutExpired:
                proc.terminate(); proc.wait(timeout=5)
            evidence = [json.loads(p.read_text()) for p in run.glob('*/cleanup.json')]
            passed = bool(evidence) and all(x.get('cleanup_verified') is True for x in evidence)
            progress(run, cfg, 'failed' if failure else 'complete', error=failure)
            h.save(run / 'task/cleanup.json', dict(experiment_id=cfg['id'], task_id='comparison',
                timestamp=time.time(), passed=passed, gpu_idle=passed, model_processes_exited=passed,
                source='actual owned-group worker cleanup', worker_evidence_count=len(evidence)))
    return 0


import sampling
sampling.install(sys.modules[__name__])
import benchmark
benchmark.install(sys.modules[__name__])


import workload_benchmark
workload_benchmark.install(sys.modules[__name__])

import lookahead_workloads
lookahead_workloads.install(sys.modules[__name__])

import prefill_benchmark
prefill_benchmark.install(sys.modules[__name__])

import numerical_replay
numerical_replay.install(sys.modules[__name__])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--engine-entry', action='store_true')
    ap.add_argument('--arm', choices=['ep2','pp2'])
    args = ap.parse_args()
    cfg = json.loads((args.run / 'config.json').read_text())
    if args.engine_entry:
        engine_entry(args.run, cfg, args.arm)
    elif args.worker:
        try:
            exit_code = worker(args.run, cfg)
        except BaseException:
            traceback.print_exc()
            exit_code = 1
        sys.exit(exit_code)
    else:
        sys.exit(controller(args.run, cfg))
