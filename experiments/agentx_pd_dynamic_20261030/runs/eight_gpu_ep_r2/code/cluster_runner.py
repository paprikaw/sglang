"""One exact two-node Guard lease, owned services, and compute-node replay client."""
import argparse
import concurrent.futures as cf
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

import agentx_replay as replay


def load_base(run, root, node):
    sys.path.insert(0, str(run/'code'))
    spec = importlib.util.spec_from_file_location('cluster_base', run/'code/runner.py')
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    cfg = root['node_configs'][node]
    original_args = base.engine_args

    def args(c, arm, replica):
        result = original_args(c, arm, replica)
        inst = base.layout(c, arm)['instances'][replica]
        result[result.index('--host')+1] = '0.0.0.0'
        if inst['role'] != 'colocated':
            result += ['--disaggregation-mode', inst['role'],
                       '--disaggregation-transfer-backend', 'mooncake',
                       '--disaggregation-bootstrap-port', str(inst['bootstrap_port']),
                       '--disaggregation-ib-device', 'mlx5_bond_0']
            if inst['role'] == 'decode':
                result += ['--disaggregation-decode-enable-radix-cache']
        return result

    base.engine_args = args
    # The inherited launch helper invokes this file, preserving Guard auth and
    # the exact owned process-group cleanup used by prior qualified experiments.
    base.__file__ = str(Path(__file__).resolve())
    return base, cfg


def worker(run, root, arm):
    node = socket.gethostname().split('.')[0]
    base, cfg = load_base(run, root, node)
    packet = json.loads(sys.stdin.readline())
    env = dict(os.environ, **packet['environment'])
    base.check(run, cfg, env, True)
    key = base.h.read_key(cfg)
    children = []

    def stop(signum, frame):
        raise InterruptedError('Owned service worker requested to clean up')

    signal.signal(signal.SIGTERM, stop)

    def parent_closed():
        sys.stdin.read()
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=parent_closed, daemon=True).start()
    try:
        Path(cfg['gap_policy_path']).write_bytes(b'0')
        Path(cfg['fixed_cohort_control']).write_bytes(bytes([0]))
        layout = base.layout(cfg, arm)
        for i in range(layout['replicas']):
            children.append(base.launch(run, cfg, arm, i, env, key))
        infos = []
        deadline = time.monotonic()+cfg['startup_timeout_s']
        for i, child in enumerate(children):
            while True:
                base.check(run, cfg, env, True)
                if child['process'].poll() is not None:
                    raise RuntimeError(f'{node}/{arm}/{i} exited during startup')
                try:
                    info = replay.http(node, base.port(cfg, arm, i), '/get_server_info', key, timeout=3)
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Model startup timed out')
                    time.sleep(2)
            assert all(s.get('enable_hierarchical_cache') for s in info['internal_states'])
            infos.append(base.h.safe_server_info(info))
        base.h.save(run/'nodes'/node/arm/'ready.json', {'passed': True, 'server_info': infos})
        while True:
            base.check(run, cfg, env, True)
            if any(c['process'].poll() is not None for c in children):
                raise RuntimeError('Owned model server exited')
            time.sleep(1)
    except InterruptedError:
        return 0
    finally:
        evidence = base.h.cleanup(children) if children else {'cleanup_verified': True}
        base.h.save(run/'nodes'/node/arm/'cleanup.json', evidence)
        if not evidence['cleanup_verified']:
            raise RuntimeError('Service cleanup unverified')


def coordinator(run, root, arm):
    node = socket.gethostname().split('.')[0]
    base, cfg = load_base(run, root, node)
    packet = json.loads(sys.stdin.readline())
    env = dict(os.environ, **packet['environment'])
    base.check(run, cfg, env, True)
    def interrupted(signum, frame):
        raise InterruptedError('Coordinator parent closed')
    signal.signal(signal.SIGTERM, interrupted)
    def parent_closed():
        sys.stdin.read()
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=parent_closed, daemon=True).start()
    key = base.h.read_key(cfg)
    def health():
        base.check(run, cfg, env, True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
    body = tokenizer.encode('def merge_records(items):\n    return sorted(items)\n', add_special_tokens=False)
    endpoints = root['endpoints'][arm]
    def flush():
        with cf.ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            replies = list(pool.map(lambda e: replay.http(e['host'], e['port'], '/flush_cache?timeout=60', key, timeout=70), endpoints))
        assert all(x.startswith('Cache flushed.') for x in replies)
    flush()
    p_eps = [e for e in endpoints if e['role'] in ('colocated', 'prefill')]
    d_eps = [e for e in endpoints if e['role'] == 'decode']
    responses = []
    for i, length in enumerate((512, 513, 2048, 2049, 4096, 4097)):
        tokens = ([5012] + body*512)[:length]
        result = replay.request_one(p_eps[i % len(p_eps)], d_eps[i % len(d_eps)] if d_eps else None,
                                    key, tokens, 32, i)
        responses.append(result)
    reference_file = run/'evidence/reference.json'
    exact = None
    if reference_file.exists():
        ref = json.loads(reference_file.read_text())
        exact = [r['output_ids'] for r in responses] == ref['output_ids']
    elif d_eps:
        raise RuntimeError('Cross-node PD requires an existing serial colocated reference')
    else:
        base.h.save(reference_file, {'source_arm': arm, 'output_ids': [r['output_ids'] for r in responses]})
    base.h.save(run/'evidence'/arm/'serial_qualification.json',
                {'responses': responses, 'exact_match_reference': exact, 'passed': exact is not False})
    if exact is False:
        raise RuntimeError('Cross-layout serial output qualification failed; performance withheld')
    graph_data = json.loads(Path(root['graphs_file']).read_text())
    for case in root['cases']:
        graphs = [g for g in graph_data['graphs'] if g['id'] in case['graph_ids']]
        assert len(graphs) == len(case['graph_ids'])
        policies = root['policies'].get(arm, ['sticky_hash'])
        for policy in policies:
            for repeat in range(root['repeats']):
                health()
                flush()
                out = run/'results'/arm/case['id']/policy
                base.h.save(run/'task/progress.json', dict(experiment_id=root['id'], task_id='comparison',
                    stage=f'{arm}:{case["id"]}:{policy}:r{repeat+1}', timestamp=time.time(),
                    valid_points=len(list(run.glob('results/*/*/*/formal.r*.json'))), expected=root['expected_points']))
                result = replay.replay(graphs, endpoints, policy, case['concurrency'], key, body, health,
                    namespace=f'agentx-{case["id"]}-repeat{repeat+1}', idle_gap_cap_s=root['idle_gap_cap_s'], copies=case.get('copies', 1))
                base.h.save(out/f'formal.r{repeat+1}.json', result)
                base.h.save(out/f'completed.r{repeat+1}.json', {'passed': True, 'requests': result['completed_requests']})
    return 0


def controller(run, root):
    base, first = load_base(run, root, root['nodes'][0])
    base.check(run, first, os.environ)
    packet = {'environment': {k: os.environ[k] for k in base.h.GUARD_KEYS}}
    base.h.save(run/'task/controller_identity.json', {'pid': os.getpid(), 'started': time.time()})
    failed = None
    try:
        for arm in root['arms']:
            processes = []
            coordinator_process = None
            try:
                for node in root['nodes']:
                    cfg = root['node_configs'][node]
                    logpath = run/'nodes'/node/arm/'worker.log'
                    logpath.parent.mkdir(parents=True, exist_ok=True)
                    log = logpath.open('ab', buffering=0)
                    command = [cfg['python'], str(Path(__file__).resolve()), '--run', str(run), '--worker', '--arm', arm]
                    process = subprocess.Popen(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', node, shlex.join(command)],
                        stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
                    process.stdin.write((json.dumps(packet)+'\n').encode()); process.stdin.flush()
                    processes.append((node, process, log))
                deadline = time.monotonic()+first['startup_timeout_s']+60
                while True:
                    for node, process, _ in processes:
                        if process.poll() is not None:
                            raise RuntimeError(f'Service worker failed on {node}')
                    ready = [run/'nodes'/n/arm/'ready.json' for n in root['nodes']]
                    if all(p.exists() for p in ready):
                        from artifact_visibility import wait_for_stable_json
                        for path in ready:
                            value, _ = wait_for_stable_json(path, timeout_s=30)
                            assert value['passed']
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError('Both nodes were not ready')
                    base.check(run, first, os.environ)
                    base.h.save(run/'task/progress.json', dict(experiment_id=root['id'], task_id='comparison',
                        stage='loading_'+arm, timestamp=time.time(), valid_points=len(list(run.glob('results/*/*/*/formal.r*.json'))), expected=root['expected_points']))
                    time.sleep(2)
                with (run/f'coordinator.{arm}.log').open('ab', buffering=0) as log:
                    command = [first['python'], str(Path(__file__).resolve()), '--run', str(run), '--coordinator', '--arm', arm]
                    coordinator_process = subprocess.Popen(['ssh', '-T', '-o', 'BatchMode=yes', root['nodes'][0], shlex.join(command)], stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
                    coordinator_process.stdin.write((json.dumps(packet)+'\n').encode()); coordinator_process.stdin.flush()
                    while coordinator_process.poll() is None:
                        base.check(run, first, os.environ)
                        progress_path = run/'task/progress.json'
                        value = json.loads(progress_path.read_text())
                        if time.time()-value.get('timestamp', 0) > 5:
                            value.update(timestamp=time.time(), heartbeat_source='Guard-authenticated two-node coordinator alive')
                            base.h.save(progress_path, value)
                        if any(p.poll() is not None for _, p, _ in processes):
                            raise RuntimeError('Service worker exited during replay')
                        time.sleep(2)
                    if coordinator_process.returncode:
                        raise RuntimeError(f'Coordinator failed for {arm}')
            finally:
                if coordinator_process is not None:
                    coordinator_process.stdin.close()
                    if coordinator_process.poll() is None:
                        coordinator_process.terminate()
                        coordinator_process.wait(timeout=15)
                for _, process, _ in processes:
                    process.stdin.close()
                for node, process, log in processes:
                    try:
                        process.wait(timeout=85)
                    except subprocess.TimeoutExpired:
                        process.terminate(); process.wait(timeout=5)
                    log.close()
                    evidence = json.loads((run/'nodes'/node/arm/'cleanup.json').read_text())
                    if not evidence['cleanup_verified']:
                        raise RuntimeError('Unverified remote cleanup')
    except BaseException as error:
        failed = type(error).__name__
        raise
    finally:
        evidence = [json.loads(p.read_text()) for p in run.glob('nodes/*/*/cleanup.json')]
        passed = bool(evidence) and all(e.get('cleanup_verified') for e in evidence)
        base.h.save(run/'task/cleanup.json', dict(experiment_id=root['id'], task_id='comparison',
            timestamp=time.time(), passed=passed, gpu_idle=passed, model_processes_exited=passed,
            source='Owned two-node worker process groups', worker_evidence_count=len(evidence)))
        base.h.save(run/'task/progress.json', dict(experiment_id=root['id'], task_id='comparison',
            stage='failed' if failed else 'complete', error=failed, timestamp=time.time(),
            valid_points=len(list(run.glob('results/*/*/*/formal.r*.json'))), expected=root['expected_points']))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--coordinator', action='store_true')
    parser.add_argument('--engine-entry', action='store_true')
    parser.add_argument('--arm')
    parser.add_argument('--replica', type=int, default=0)
    args = parser.parse_args()
    root = json.loads((args.run/'config.json').read_text())
    try:
        if args.engine_entry:
            base, cfg = load_base(args.run, root, socket.gethostname().split('.')[0])
            status = base.engine_entry(args.run, cfg, args.arm, args.replica)
        elif args.worker:
            status = worker(args.run, root, args.arm)
        elif args.coordinator:
            status = coordinator(args.run, root, args.arm)
        else:
            status = controller(args.run, root)
    except BaseException:
        traceback.print_exc()
        status = 1
    sys.exit(status or 0)
