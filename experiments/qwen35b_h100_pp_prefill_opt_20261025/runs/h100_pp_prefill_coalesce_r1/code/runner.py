"""One supervised node, two disjoint parallel serving pairs; real full model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

import base_runner as b
import process_helpers as h
from protocol import cases, logical_capacity, microbatch_size, own_capacity_case


def progress(run, cfg, stage, **extra):
    h.save(run / 'task/progress.json', dict(experiment_id=cfg['id'], task_id='comparison',
        timestamp=time.time(), stage=stage, valid_points=len(list(run.glob('*/metrics.*.json'))),
        expected=12, **extra))


def arm_cfg(cfg, arm):
    result = dict(cfg)
    # DP controller also consumes adjacent rendezvous ports. Two servers
    # require disjoint PORT RANGES, not merely different primary TCP ports.
    result['port'] = cfg['port'] + dict(ep2=0, pp2=100, single=200)[arm]
    return result


def engine_args(cfg, arm):
    row = next(r for r in cfg['layouts'] if r['id'] == arm)
    args = ['--model-path', cfg['model_path'], '--host', '127.0.0.1', '--port', str(cfg['port']),
        '--dist-init-addr', f"127.0.0.1:{cfg['dist_port'] + dict(ep2=0, pp2=100, single=200)[arm]}",
        '--tp-size', str(row['tp']), '--ep-size', str(row['ep']), '--dp-size', str(row['dp']),
        '--pp-size', str(row['pp']), '--dtype', 'bfloat16', '--kv-cache-dtype', 'bfloat16',
        '--context-length', '262144', '--mem-fraction-static', '.94',
        '--page-size', '64', '--max-running-requests', '32',
        # Native DP divides both options once more: global 88 -> 44 states
        # per Attention instance; global graph16 -> local graph8.
        '--max-mamba-cache-size', '168' if arm == 'ep2' else '164',
        # Retain the final prefix state, not every obsolete intermediate
        # chunk checkpoint along its path. Native cache pruning preserves
        # tails, forks and locked states and does not change model math.
        '--mamba-max-states-per-path', '1',
        '--mamba-ssm-dtype', 'float32', '--cuda-graph-max-bs-decode', '32' if arm == 'ep2' else '16',
        '--cuda-graph-backend-prefill', 'disabled',
        '--chunked-prefill-size', '8192' if arm == 'ep2' else '4096',
        '--max-prefill-tokens', '8192' if arm == 'ep2' else '4096',
        '--attention-backend', cfg['attention_backend'], '--linear-attn-backend', 'triton',
        '--moe-runner-backend', 'deep_gemm', '--moe-a2a-backend', 'deepep' if arm == 'ep2' else 'none',
        '--skip-server-warmup', '--json-model-override-args', '{"language_model_only":true}']
    if arm == 'ep2':
        args += ['--enable-dp-attention', '--deepep-mode', 'auto', '--deepep-dispatcher-output-dtype', 'bf16']
    elif arm == 'pp2':
        args += ['--pp-max-micro-batch-size', '4', '--pp-async-batch-depth', '0']
    else:
        for flag, value in [('--max-running-requests','2'), ('--max-mamba-cache-size','12'),
                            ('--cuda-graph-max-bs-decode','2')]:
            args[args.index(flag)+1] = value
    return args


def environment(cfg, arm, inherited):
    env = b.environment(cfg, arm, inherited)
    row = next(r for r in cfg['layouts'] if r['id'] == arm)
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, row['gpu_groups'][0]))
    env['PP_READY_POLICY_PATH'] = cfg['policy_path']
    return env


def engine_entry(run, cfg, arm):
    h.authenticate(run, cfg, os.environ, local=True)
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


def launch(run, cfg, arm, env, key):
    use = next(r for r in cfg['layouts'] if r['id'] == arm)['usage_logs'][0]
    command = [cfg['python'], str(Path(cfg['guard_scripts']) / 'model_cache_adapter.py'),
        'exec-local', '--bindings', env[h.GUARD_KEYS[3]], '--', cfg['python'],
        str(Path(__file__).resolve()), '--run', str(run), '--engine-entry', '--arm', arm]
    proc = subprocess.Popen(command, env=environment(cfg, arm, env), cwd=cfg['engine_root'],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True)
    identity = h.process_record(proc.pid)
    if not identity or identity['pgid'] != proc.pid:
        proc.terminate()
        raise RuntimeError('Owned process group was not established')
    thread = threading.Thread(target=h.stream_log, args=(proc.stdout, use['path'], key), daemon=True)
    thread.start()
    return dict(process=proc, identity=identity, log_thread=thread, arm=arm)


def freeze_gc(cfg, key):
    request = urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/freeze_gc",
        data=b'{}', method='POST', headers={'Authorization':'Bearer '+key,
                                          'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read().decode()
        if response.status != 200 or body.strip() != 'Garbage collection frozen.':
            raise RuntimeError('Native GC initialization was not acknowledged')
    return body.strip()


def wait_ready(run, cfg, arm, child, health):
    deadline = time.monotonic() + cfg['startup_timeout_s']
    while True:
        health()
        if child['process'].poll() is not None:
            raise RuntimeError(arm + ' exited during native Hopper startup; inspect its cache log')
        try:
            info = h.http_json(cfg['port'], '/get_server_info', h.read_key(cfg), timeout=3)
            break
        except Exception:
            if time.monotonic() > deadline:
                raise TimeoutError(arm + ' startup exceeded its bounded deadline')
            time.sleep(2)
    # skip-server-warmup can fire the engine's GC callback before the HTTP
    # socket exists. Complete the identical initialization explicitly AFTER
    # readiness for every arm, outside all performance and profile windows.
    gc_result = freeze_gc(cfg, h.read_key(cfg))
    h.save(run / arm / 'gc_freeze.json', dict(passed=True, result=gc_result,
        initialization_only=True, timestamp=time.time()))
    quals = list((run / arm / 'qualification').glob('*.json'))
    if len(quals) != (1 if arm == 'single' else 2) or not all(json.loads(p.read_text()).get('passed') for p in quals):
        raise RuntimeError('Both native full-model ranks must qualify')
    h.save(run / arm / 'server_info.json', h.safe_server_info(info))
    return (h.allocator_capacity(info) if arm == 'single'
            else logical_capacity(arm, h.allocator_capacity(info)))


def make_tokens(cfg, requests, length):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
    result = []
    for index in range(requests):
        # The initial unique token prevents between-request shared prefixes.
        prefix = [5000 + index] + tok.encode(f'Document {index}. ', add_special_tokens=False)
        body = tok.encode('A distributed service keeps a record of independent requests. '
                          'Explain the computation and communication clearly.\n', add_special_tokens=False)
        result.append((prefix + body * (length // len(body) + 2))[:length])
    return result


def set_microbatch(cfg, arm, key, requests):
    if arm != 'pp2':
        return
    response = h.http_json(cfg['port'], '/set_internal_state', key,
        dict(server_args=dict(pp_max_micro_batch_size=microbatch_size(requests))), timeout=30)
    if isinstance(response, dict) and response.get('success') is False:
        raise RuntimeError('Native PP microbatch update was rejected')


def check_cached(round_, context):
    counts = [r.get('cached_tokens') for r in round_['requests']]
    if any(type(n) is not int or n < context - 64 for n in counts):
        raise RuntimeError(f'Expected already cached context {context}; observed {counts}')


def prime(cfg, arm, key, tokens, health):
    health()
    b.flush_cache(cfg, key)
    begin = time.monotonic()
    # Bound simultaneous working/ping-pong GDN slots while retaining already
    # prepared prefixes. This does NOT throttle formal request arrivals.
    for first in range(0, len(tokens), 4):
        health()
        b.cohort(cfg, key, [b.packet(cfg, arm, t, 1, first + i)
            for i, t in enumerate(tokens[first:first + 4])])
    return dict(preparation_s=time.monotonic() - begin, prefix_requests=len(tokens),
                tokens_per_request=len(tokens[0]), preparation_cohort_limit=4,
                measured_performance_sample=False)


def payloads(cfg, arm, tokens, fresh, outputs, branch):
    # Every timed Prefill branch starts with a different new token. Repeating
    # the exact prompt would turn later Prefill rounds into unintended hits.
    return [b.packet(cfg, arm, t + [7000 + branch] + [198] * (fresh - 1), outputs, i)
            for i, t in enumerate(tokens)]


def tuning(run, cfg, arm, key, health):
    import pp_ready_send as ready
    tokens = make_tokens(cfg, 8, 4096)
    set_microbatch(cfg, arm, key, 8)
    priming = dict(preparation_s=0, cold_prefill=True)
    # Compile/warm the exact cold 4096-token shape before either A or B.
    b.flush_cache(cfg, key)
    b.cohort(cfg, key, [b.packet(cfg, arm, t, 1, i) for i, t in enumerate(tokens)])
    rows = []
    order = [False, True, True, False] if arm == 'pp2' else [False, False]
    for index, policy in enumerate(order):
        if arm == 'pp2':
            ready.set_policy(cfg['policy_path'], policy)
        b.flush_cache(cfg, key)
        req = [b.packet(cfg, arm, t, 1, i) for i, t in enumerate(tokens)]
        health()
        result = b.cohort(cfg, key, req)
        rows.append(dict(ready_send=policy, makespan_s=result['makespan_s'],
                         output_ids=[r['output_ids'] for r in result['requests']]))
    # Same first unseen token across the two policies in a separate flushed
    # context proves the scheduler-only path did not alter these test outputs.
    agreement = True
    if arm == 'pp2':
        outputs = []
        for policy in (False, True):
            ready.set_policy(cfg['policy_path'], policy)
            b.flush_cache(cfg, key)
            result = b.cohort(cfg, key, [b.packet(cfg, arm, t, 1, i) for i, t in enumerate(tokens)])
            outputs.append([r['output_ids'] for r in result['requests']])
        agreement = outputs[0] == outputs[1] and all(x is not None for x in outputs[0])
    baseline = statistics.median(r['makespan_s'] for r in rows if not r['ready_send'])
    candidate = statistics.median(r['makespan_s'] for r in rows if r['ready_send']) if arm == 'pp2' else baseline
    selected = bool(arm == 'pp2' and agreement and candidate < .99 * baseline)
    if arm == 'pp2':
        ready.set_policy(cfg['policy_path'], False)
    evidence = dict(prefill_ready_send_selected=selected, decode_ready_send_selected=False,
        output_agreement=agreement, baseline_s=baseline, candidate_s=candidate,
        trials=rows, preparation=priming, cuda_graph_enabled=True,
        comparison_scope='short local ABBA; not a global optimum claim')
    h.save(run / arm / 'tuning.json', evidence)
    return evidence


def measure_case(run, cfg, arm, key, row, tuned, health):
    import pp_ready_send as ready
    set_microbatch(cfg, arm, key, row['requests'])
    tokens = make_tokens(cfg, row['requests'], row['cached_context_tokens'])
    cold_prefill = row['prefill_cached_context_tokens'] == 0
    prefill_preparations = []
    if arm == 'pp2':
        ready.set_policy(cfg['policy_path'], tuned['prefill_ready_send_selected'])
    # Shape-specific warmup is excluded from performance.
    def prefill_payloads(branch):
        if cold_prefill:
            b.flush_cache(cfg, key)
            return [b.packet(cfg, arm, t, 1, i) for i, t in enumerate(tokens)]
        # Hybrid radix caches retain a GDN state for every fresh branch.
        # Keeping all previous timed branches can evict an untouched base
        # prefix even while the FullKV pool has room. Reset and reproduce
        # the SAME base before every trial, outside its timed cohort.
        prefill_preparations.append(prime(cfg, arm, key, tokens, health))
        return payloads(cfg, arm, tokens, row['new_prefill_tokens'], 1, branch)
    b.cohort(cfg, key, prefill_payloads(300))
    rounds = []
    for repeat in range(cfg['repeats']):
        health()
        result = b.cohort(cfg, key, prefill_payloads(301 + repeat))
        if not cold_prefill:
            check_cached(result, row['cached_context_tokens'])
        elif any((r.get('cached_tokens') or 0) != 0 for r in result['requests']):
            raise RuntimeError('Cold Prefill unexpectedly hit a cached prefix')
        rounds.append(result)
    pre = b.summarize(rounds, 'cached-context Prefill')
    pre.update(case=row, fresh_input_tokens_per_s=statistics.mean(
        row['requests'] * row['new_prefill_tokens'] / r['makespan_s'] for r in rounds),
        timing_includes_context_preparation=False,
        preparation=dict(cold_prefill=cold_prefill, trials=prefill_preparations,
                         reset_before_each_trial=True),
        selected_ready_send=tuned['prefill_ready_send_selected'], pipeline_slots=2 if arm == 'pp2' else None)
    h.save(run / arm / ('metrics.' + row['id'] + '.prefill.json'), pre)
    if arm == 'pp2':
        ready.set_policy(cfg['policy_path'], False)
    # New preparation isolates Decode from Prefill branches and GDN states.
    prime_record = prime(cfg, arm, key, tokens, health)
    req = payloads(cfg, arm, tokens, 1, row['output_tokens'], 600)
    warm = b.cohort(cfg, key, req)
    check_cached(warm, row['cached_context_tokens'])
    rounds = []
    for repeat in range(cfg['repeats']):
        health()
        result = b.cohort(cfg, key, req)
        check_cached(result, row['cached_context_tokens'])
        rounds.append(result)
    dec = b.summarize(rounds, 'cached-context Decode')
    dec.update(case=row, timing_includes_context_preparation=False, preparation=prime_record,
               selected_ready_send=False, pipeline_slots=2 if arm == 'pp2' else None)
    h.save(run / arm / ('metrics.' + row['id'] + '.decode.json'), dec)
    return req


def profile_case(run, cfg, arm, key, row, req, health):
    # A short bounded trace is diagnostic only, after formal performance.
    trace_dir = run / arm / 'traces' / row['id']
    trace_dir.mkdir(parents=True, exist_ok=True)
    b.profile_action(cfg, key, '/start_profile', dict(output_dir=str(trace_dir), num_steps=4,
        activities=['CPU', 'GPU'], record_shapes=True, with_stack=False,
        profile_id=arm + '-' + row['id'], merge_profiles=False, detailed_annotations=True))
    b.cohort(cfg, key, req)
    deadline = time.monotonic() + 90
    while len(list(trace_dir.glob('*.trace.json.gz'))) < 2:
        health()
        if time.monotonic() > deadline:
            raise TimeoutError('Both actual ranks must finish trace export')
        time.sleep(1)
    # File existence alone is not complete export. Require gzip CRC and JSON.
    import gzip
    import zlib
    while True:
        try:
            for path in trace_dir.glob('*.trace.json.gz'):
                json.loads(gzip.decompress(path.read_bytes()))
            break
        except (OSError, EOFError, zlib.error, json.JSONDecodeError):
            health()
            if time.monotonic() > deadline:
                raise TimeoutError('Decode trace export remained incomplete')
            time.sleep(1)
    h.save(trace_dir / 'manifest.json', dict(performance_sample=False,
        graph_enabled=True, files=[str(p) for p in trace_dir.glob('*.trace.json.gz')]))


def worker(run, cfg):
    incoming = json.loads(sys.stdin.readline())
    env = dict(os.environ, **incoming['environment'])
    h.authenticate(run, cfg, env, local=True)
    def interrupted(signum, frame):
        raise RuntimeError('Guard or controller requested owned cleanup')
    signal.signal(signal.SIGTERM, interrupted)
    def watch_parent():
        sys.stdin.read()
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=watch_parent, daemon=True).start()
    key = h.read_key(cfg)
    import pp_ready_send as ready
    ready.create_policy(cfg['policy_path'])
    children = []
    failure = None
    try:
        topology = subprocess.run(['nvidia-smi', 'topo', '-m'], text=True,
            capture_output=True, timeout=20, check=True)
        (run / 'topology.txt').write_text(topology.stdout)
        devices = subprocess.run(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,driver_version',
            '--format=csv'], text=True, capture_output=True, timeout=20, check=True)
        (run / 'devices.csv').write_text(devices.stdout)
        def health():
            h.authenticate(run, cfg, env, local=True)
            for child in children:
                if child['process'].poll() is not None:
                    raise RuntimeError(child['arm'] + ' exited while serving')
        # Resolve Transformers' lazy AutoTokenizer import once, before any
        # worker thread and before allocating GPU models. Its first concurrent
        # import can observe a partially initialized lazy module.
        make_tokens(cfg, 1, 16)
        progress(run, cfg, 'single_gpu_fit_qualification')
        single = launch(run, arm_cfg(cfg, 'single'), 'single', env, key)
        children.append(single)
        single_capacity = wait_ready(run, arm_cfg(cfg, 'single'), 'single', single, health)
        proof = b.cohort(arm_cfg(cfg, 'single'), key,
            [b.packet(cfg, 'single', make_tokens(cfg, 1, 4096)[0], 4, 0)])
        h.save(run / 'single_gpu_fit.json', dict(passed=True,
            allocator_tokens=single_capacity, proof=proof,
            scope='Complete BF16 model, nonzero KV pool and real 4096-input/4-output request on GPU0'))
        evidence = h.cleanup([single])
        h.save(run / 'single_gpu_cleanup.json', evidence)
        if not evidence['cleanup_verified']:
            raise RuntimeError('Single-GPU qualification cleanup failed')
        children.clear()
        progress(run, cfg, 'parallel_model_loading')
        for arm in ('ep2', 'pp2'):
            children.append(launch(run, arm_cfg(cfg, arm), arm, env, key))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {arm: pool.submit(wait_ready, run, arm_cfg(cfg, arm), arm, child, health)
                       for arm, child in zip(('ep2', 'pp2'), children)}
            capacities = {arm: future.result() for arm, future in futures.items()}
        matrix = cases(capacities)
        h.save(run / 'capacity_matrix.json', dict(logical_capacities=capacities, cases=matrix))
        # Qualify the largest REQUEST count before running a full sweep.
        # A capacity-only check is insufficient for hybrid FullKV/GDN caches.
        row = max(matrix, key=lambda r:r['requests'])
        def qualify_cached_requests(arm):
            local = arm_cfg(cfg, arm)
            qualified_row = (own_capacity_case(capacities[arm]) if arm == 'pp2'
                             and capacities[arm] > 1.15 * min(capacities.values()) else row)
            set_microbatch(local, arm, key, qualified_row['requests'])
            tokens = make_tokens(local, qualified_row['requests'], qualified_row['cached_context_tokens'])
            prepared = prime(local, arm, key, tokens, health)
            result = b.cohort(local, key, payloads(local, arm, tokens, 512, 1, 270))
            check_cached(result, qualified_row['cached_context_tokens'])
            h.save(run / arm / 'cache_residency_qualification.json',
                   dict(passed=True, case=qualified_row, preparation=prepared,
                        cached_tokens=[r['cached_tokens']
                        for r in result['requests']], performance_sample=False))
        progress(run, cfg, 'cache_residency_qualification')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(qualify_cached_requests, arm) for arm in ('ep2', 'pp2')]
            for future in futures:
                future.result()
        progress(run, cfg, 'parallel_local_tuning', capacities=capacities, matrix=matrix)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {arm: pool.submit(tuning, run, arm_cfg(cfg, arm), arm, key, health)
                       for arm in ('ep2', 'pp2')}
            tuned = {arm: f.result() for arm, f in futures.items()}
        # Case barriers keep workload order aligned without timing the barrier.
        for row in matrix:
            progress(run, cfg, 'parallel_' + row['id'], workload=row)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {arm: pool.submit(measure_case, run, arm_cfg(cfg, arm), arm, key, row,
                                            tuned[arm], health) for arm in ('ep2', 'pp2')}
                reqs = {arm: future.result() for arm, future in futures.items()}
            # Profiling cannot overlap the OTHER configuration's formal samples.
            progress(run, cfg, 'profiling_' + row['id'])
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(profile_case, run, arm_cfg(cfg, arm), arm, key,
                                       row, reqs[arm], health) for arm in ('ep2', 'pp2')]
                for future in futures:
                    future.result()
        extra_cases = []
        # Common workloads answer the apples-to-apples performance question.
        # An extra PP-only capacity point shows its own boundary if capacity
        # differs substantially. Different context sizes are never ranked as
        # a matched-latency comparison.
        if capacities['pp2'] > 1.15 * min(capacities.values()):
            row = own_capacity_case(capacities['pp2'])
            progress(run, cfg, 'pp_own_capacity', workload=row)
            req = measure_case(run, arm_cfg(cfg, 'pp2'), 'pp2', key, row, tuned['pp2'], health)
            profile_case(run, arm_cfg(cfg, 'pp2'), 'pp2', key, row, req, health)
            extra_cases.append(dict(arm='pp2', case=row))
        import prefill_followup
        progress(run, cfg, 'prefill_component_profiles')
        for row in matrix:
            diagnostic = dict(row, id=row['id'] + '_prefill')
            def sample_prefill(arm):
                local = dict(arm_cfg(cfg, arm), prefill_ready_send=tuned[arm]['prefill_ready_send_selected'])
                prefill_followup.profile_cell(sys.modules[__name__], run, local, arm, key, diagnostic, health)
            with ThreadPoolExecutor(max_workers=2) as pool:
                for f in [pool.submit(sample_prefill, arm) for arm in ('ep2', 'pp2')]:
                    f.result()
        h.save(run / 'completed.json', dict(passed=True, logical_capacities=capacities,
            cases=matrix, extra_cases=extra_cases))
    except BaseException:
        failure = traceback.format_exc()
        (run / 'worker_error.txt').write_text(failure)
        raise
    finally:
        evidence = h.cleanup(children) if children else dict(cleanup_verified=False, reason='no owned model group')
        h.save(run / 'worker_cleanup.json', evidence)
        Path(cfg['policy_path']).unlink(missing_ok=True)
        if not evidence['cleanup_verified'] and failure is None:
            raise RuntimeError('Owned node cleanup failed verification')
    return 0


def controller(run, cfg):
    h.authenticate(run, cfg, os.environ)
    command = [cfg['python'], str(Path(__file__).resolve()), '--run', str(run), '--worker']
    failed = None
    with (run / 'controller.log').open('ab', buffering=0) as log:
        proc = subprocess.Popen(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            cfg['node'], shlex.join(command)], stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
        proc.stdin.write((json.dumps(dict(environment={k: os.environ[k] for k in h.GUARD_KEYS})) + '\n').encode())
        proc.stdin.flush()
        try:
            while proc.poll() is None:
                h.authenticate(run, cfg, os.environ)
                time.sleep(2)
            if proc.returncode:
                raise RuntimeError('H100 worker failed; inspect worker_error.txt and the two loader logs')
        except BaseException as error:
            failed = str(error)
            raise
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=85)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=5)
            path = run / 'worker_cleanup.json'
            evidence = json.loads(path.read_text()) if path.exists() else {}
            passed = evidence.get('cleanup_verified') is True
            progress(run, cfg, 'failed' if failed else 'complete', error=failed)
            h.save(run / 'task/cleanup.json', dict(experiment_id=cfg['id'], task_id='comparison',
                timestamp=time.time(), passed=passed, gpu_idle=passed, model_processes_exited=passed,
                source='actual owned groups on leased node', evidence=str(path)))
    return 0


from optimization import worker

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--engine-entry', action='store_true')
    ap.add_argument('--arm', choices=['ep2', 'pp2', 'single'])
    args = ap.parse_args()
    cfg = json.loads((args.run / 'config.json').read_text())
    try:
        result = (engine_entry(args.run, arm_cfg(cfg, args.arm), args.arm) if args.engine_entry
                  else worker(args.run, cfg) if args.worker else controller(args.run, cfg))
    except Exception:
        traceback.print_exc()
        result = 1
    sys.exit(result or 0)
