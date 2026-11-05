"""Native full-model four-GPU comparison; no component latency reconstruction."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import urllib.request


def instance_config(cfg, arm, instance):
    result = deepcopy(cfg)
    row = next(x for x in cfg['layouts'] if x['id'] == arm)
    assert 0 <= instance < row['replicas']
    offset = instance * 10 if arm == 'replicas' else 20
    result.update(port=cfg['port'] + offset, dist_port=cfg['dist_port'] + offset,
                  ep=row['ep'], dp=row['dp'], instance=instance, arm=arm,
                  physical_gpus=row['gpu_groups'][instance],
                  rank_offset=instance * 2 if arm == 'replicas' else 0,
                  usage_log=row['usage_logs'][instance])
    return result


def byte(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_NOFOLLOW)
    try:
        os.pwrite(fd, bytes([value]), 0)
    finally:
        os.close(fd)


def install(base):
    original_environment = base.environment
    epoch = [0]

    def args(cfg, arm):
        width = cfg['ep']
        return [
            '--model-path', cfg['model_path'], '--host', '127.0.0.1', '--port', str(cfg['port']),
            '--dist-init-addr', f"127.0.0.1:{cfg['dist_port']}",
            '--tp-size', str(width), '--ep-size', str(width), '--dp-size', str(width), '--pp-size', '1',
            '--enable-dp-attention', '--dtype', 'bfloat16', '--kv-cache-dtype', 'bfloat16',
            '--context-length', '8192', '--mem-fraction-static', '0.90', '--page-size', '64',
            '--max-total-tokens', '196608', '--max-running-requests', str(16 * width),
            '--max-mamba-cache-size', str(84 * width), '--mamba-max-states-per-path', '1',
            '--mamba-ssm-dtype', 'float32', '--chunked-prefill-size', str(4096 * width),
            '--max-prefill-tokens', str(4096 * width), '--attention-backend', 'triton',
            '--linear-attn-backend', 'triton', '--moe-runner-backend', 'triton',
            '--moe-a2a-backend', 'none', '--cuda-graph-bs-decode', '2', '8', '16',
            '--cuda-graph-backend-prefill', 'disabled',
            '--disable-custom-all-reduce', '--skip-server-warmup', '--enable-cache-report',
            '--json-model-override-args', '{"language_model_only":true}',
            '--enable-hierarchical-cache', '--hicache-size', '64',
            '--hicache-host-memory-mode', 'cache', '--hicache-write-policy', 'write_through',
            '--hicache-io-backend', 'kernel', '--hicache-mem-layout', 'page_first',
        ] + ([] if cfg['overlap'] else ['--disable-overlap-schedule'])

    def environment(cfg, arm, inherited):
        env = original_environment(cfg, arm, inherited)
        env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, cfg['physical_gpus'])),
            EP2_PP2_ARM='ep2', EP2_PP2_RUN=str(Path(cfg['code_root']).parent / arm / f"i{cfg['instance']}"),
            PP_FIXED_COHORT_CONTROL=cfg['fixed_cohort_control'], PP_FIXED_COHORT_SIZE=cfg['fixed_cohort_size'],
            MATCHED_RANK_OFFSET=str(cfg['rank_offset']), DEEPEP_LL_NVLINK_PAIR_ONLY='0', EP_STEADY_HARDWARE='A100')
        return env

    def launch(run, cfg, arm, inherited, key):
        command = [cfg['python'], str(Path(cfg['guard_scripts']) / 'model_cache_adapter.py'),
            'exec-local', '--bindings', inherited[base.h.GUARD_KEYS[3]], '--', cfg['python'],
            str(Path(base.__file__).resolve()), '--run', str(run), '--engine-entry', '--arm', arm,
            '--instance', str(cfg['instance'])]
        proc = subprocess.Popen(command, env=environment(cfg, arm, inherited), cwd=cfg['engine_root'],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        identity = base.h.process_record(proc.pid)
        if not identity or identity['pgid'] != proc.pid:
            proc.terminate()
            raise RuntimeError('Owned engine process group was not established')
        thread = threading.Thread(target=base.h.stream_log, args=(proc.stdout, cfg['usage_log']['path'], key), daemon=True)
        thread.start()
        return dict(process=proc, identity=identity, log_thread=thread)

    def progress(run, cfg, stage, **extra):
        base.h.save(run / 'task/progress.json', dict(experiment_id=cfg['id'], task_id='comparison',
            stage=stage, timestamp=time.time(), controller_pid=os.getpid(),
            valid_points=len(list(run.glob('*/workloads/*/completed.json'))), expected=cfg['expected_points'], **extra))

    def cohort(cfg, instances, key, tokens, output):
        epoch[0] = epoch[0] % 254 + 1
        byte(cfg['fixed_cohort_size'], len(tokens))
        byte(cfg['fixed_cohort_control'], epoch[0])
        gate = threading.Barrier(len(tokens))
        packets = []
        for i, token in enumerate(tokens):
            instance = (i % 4) // 2 if len(instances) == 2 else 0
            local = i % 2 if len(instances) == 2 else i % 4
            payload = dict(input_ids=token, sampling_params=dict(temperature=0, max_new_tokens=output, ignore_eos=True),
                           stream=True, data_parallel_rank=local)
            packets.append((instances[instance], payload))
        with ThreadPoolExecutor(max_workers=len(tokens)) as pool:
            futures = [pool.submit(base.stream_one, c, key, p, gate) for c, p in packets]
            rows = [f.result() for f in futures]
        receipts = []
        for inst in instances:
            info = base.h.http_json(inst['port'], '/get_server_info', key, timeout=30)
            local = [s['experiment_prefill_ep_receipt'] for s in info['internal_states']
                     if 'experiment_prefill_ep_receipt' in s]
            assert len(local) == inst['ep'] and {s['dp_rank'] for s in local} == set(range(inst['ep']))
            for receipt in local:
                assert receipt['epoch'] == epoch[0]
                global_rank = inst['rank_offset'] + receipt['dp_rank']
                assert receipt['ordered_markers'] == list(range(5000 + global_rank, 5000 + len(rows), 4))
                receipts.append(dict(**receipt, physical_rank=global_rank, instance=inst['instance']))
        elapsed = max(r['finished'] for r in rows) - min(r['started'] for r in rows)
        return dict(requests=rows, makespan_s=elapsed, output_tokens_per_s=len(rows) * output / elapsed,
                    input_tokens_per_s=sum(map(len, tokens)) / elapsed, admission_receipts=receipts)

    def workloads(run, cfg, arm, instances, key, health):
        from previous_workloads import prompts

        def flush():
            health()
            for inst in instances:
                base.flush_cache(inst, key)

        for count in cfg['batches']:
            current = dict(cfg, requests=count)
            tokens = prompts(current, dict(family='diverse', requests=count))
            prime_tokens = tokens
            for phase in ('prefill_cold', 'prefill_cached', 'decode'):
                case = f'{phase}_b{count}'
                directory = run / arm / 'workloads' / case
                directory.mkdir(parents=True, exist_ok=True)
                cached = 0 if phase == 'prefill_cold' else cfg['prefix_tokens']
                extension = [198] * (cfg['new_prefill_tokens'] if phase == 'prefill_cached' else 1)
                actual = tokens if phase == 'prefill_cold' else [t + extension for t in tokens]
                output = cfg['output_tokens'] if phase == 'decode' else 1
                base.h.save(directory / 'protocol.json', dict(phase=phase, global_requests=count,
                    cached_prefix_per_request=cached, logical_input_tokens_per_request=len(actual[0]),
                    logical_new_tokens_per_request=len(actual[0]) - cached, output_tokens_per_request=output,
                    native_gpu_prime=True, cpu_checkpoint_replay=False, per_gpu_requests=count // 4,
                    profiler_off=True, total_gpus=4, ep_width=instances[0]['ep'], replicas=len(instances),
                    prompt_sha256=[hashlib.sha256(str(t).encode()).hexdigest() for t in actual]))

                def check(group, expected_cache=cached, expected_output=output, prompt_len=len(actual[0])):
                    assert len(group['requests']) == count
                    for row in group['requests']:
                        assert row['output_tokens'] == expected_output
                        assert row['prompt_tokens'] == prompt_len
                        assert row['cached_tokens'] == expected_cache, row['cached_tokens']
                        assert len(row.get('output_ids') or []) == expected_output
                        details = row.get('cached_tokens_details')
                        if isinstance(details, dict):
                            assert not details.get('host', 0) and not details.get('storage', 0), details

                def prepare():
                    flush()
                    if cached:
                        prime = cohort(current, instances, key, prime_tokens, 1)
                        check(prime, expected_cache=0, expected_output=1, prompt_len=cfg['prefix_tokens'])
                        return prime

                prepared = prepare()
                if prepared:
                    base.h.save(directory / 'prefix_preparation.json', dict(performance_sample=False, result=prepared))
                for warm in range(cfg['warmups']):
                    progress(run, cfg, f'warmup_{arm}_{case}', repeat=warm + 1)
                    check(cohort(current, instances, key, actual, output))
                formal = []
                for repeat in range(cfg['repeats']):
                    if phase != 'decode':
                        prepared = prepare()
                        if prepared:
                            base.h.save(directory / f'prime.{repeat}.json', dict(performance_sample=False, result=prepared))
                    health()
                    progress(run, cfg, f'timing_{arm}_{case}', repeat=repeat + 1)
                    measured = cohort(current, instances, key, actual, output)
                    check(measured)
                    formal.append(measured)
                    base.h.save(directory / f'cohort.{repeat}.json', dict(performance_sample=True, result=measured))
                metrics = base.summarize(formal, cfg['protocol'])
                metrics.update(phase=phase, requests=count, replicas=len(instances), ep=instances[0]['ep'])
                base.h.save(directory / 'metrics.json', metrics)
                base.h.save(directory / 'completed.json', dict(passed=True, formal_cohorts=len(formal)))
                byte(cfg['fixed_cohort_control'], 0)
            for inst in instances:
                info = base.h.http_json(inst['port'], '/get_server_info', key, timeout=30)
                base.h.save(run / arm / f'i{inst["instance"]}' / f'server_info.b{count}.json', base.h.safe_server_info(info))
        base.h.save(run / arm / 'completed.json', dict(passed=True, workloads=9))

    def worker(run, cfg):
        incoming = json.loads(base.sys.stdin.readline())
        inherited = dict(os.environ, **incoming['environment'])
        base.check(run, cfg, inherited, local=True)

        def interrupted(signum, frame):
            raise RuntimeError('Guard/controller requested owned worker cleanup')

        signal.signal(signal.SIGTERM, interrupted)

        def watch_parent():
            base.sys.stdin.read()
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=watch_parent, daemon=True).start()
        key = base.h.read_key(cfg)
        topo = subprocess.run(['nvidia-smi', 'topo', '-m'], text=True, capture_output=True, timeout=20, check=True)
        (run / 'topology.txt').write_text(topo.stdout + topo.stderr)
        clean = re.sub(chr(27) + r'\[[0-9;]*[A-Za-z]', '', topo.stdout)
        rows = [s.split() for s in clean.splitlines() if s.strip()]
        header = next(r for r in rows if r[0] == 'GPU0' and 'GPU1' in r and 'X' not in r)
        links = {}
        for a, b in [(0, 1), (2, 3), (0, 2)]:
            row = next(r for r in rows if r[0] == f'GPU{a}' and len(r) > 1 and r[a + 1] == 'X')
            links[f'{a}-{b}'] = row[header.index(f'GPU{b}') + 1]
        assert links['0-1'].startswith('NV') and links['2-3'].startswith('NV')
        base.h.save(run / 'topology_qualification.json', dict(passed=True, links=links, observed_under_guard=True))

        def health():
            base.check(run, cfg, inherited, local=True)

        for arm in cfg['arm_order']:
            instances = [instance_config(cfg, arm, i) for i in range(next(x for x in cfg['layouts'] if x['id'] == arm)['replicas'])]
            children = []
            try:
                progress(run, cfg, 'loading_' + arm)
                for inst in instances:
                    children.append(launch(run, inst, arm, inherited, key))
                deadline = time.monotonic() + cfg['startup_timeout_s']
                for inst, child in zip(instances, children):
                    while True:
                        health()
                        if child['process'].poll() is not None:
                            raise RuntimeError(f'{arm}/i{inst["instance"]} exited during startup')
                        try:
                            info = base.h.http_json(inst['port'], '/get_server_info', key, timeout=3)
                            break
                        except Exception:
                            if time.monotonic() >= deadline:
                                raise TimeoutError('Native full-model startup exceeded its deadline')
                            time.sleep(2)
                    quals = list((run / arm / f'i{inst["instance"]}' / 'ep2/qualification').glob('*.json'))
                    assert len(quals) == inst['ep'] and all(json.loads(p.read_text()).get('passed') for p in quals)
                    base.h.save(run / arm / f'i{inst["instance"]}' / 'server_info.start.json', base.h.safe_server_info(info))
                    req = urllib.request.Request(f'http://127.0.0.1:{inst["port"]}/freeze_gc',
                                                  headers={'Authorization': 'Bearer ' + key})
                    with urllib.request.urlopen(req, timeout=60) as response:
                        assert response.read().decode().startswith('Garbage collection frozen.')
                workloads(run, cfg, arm, instances, key, health)
            finally:
                if children:
                    evidence = base.h.cleanup(children)
                    base.h.save(run / arm / 'cleanup.json', evidence)
                    if not evidence['cleanup_verified']:
                        raise RuntimeError('Owned engine cleanup failed verification')
        return 0

    base.engine_args, base.environment, base.worker, base.progress = args, environment, worker, progress
