"""One exact two-node Guard lease, owned services, and compute-node replay client."""
import argparse
import concurrent.futures as cf
import importlib.util
import json
import hashlib
import math
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
from episode_capture import replay_with_failure_capture

# Spawned scheduler children re-import this module instead of load_base().
# Install the same rank-local runtime before any scheduler constructs its model.
if os.environ.get('EP2_PP2_RUN') and os.environ.get('EP2_PP2_ARM'):
    import runtime_adapter
    runtime_adapter.install()


_FORWARD_EPOCH = 0


def warmup_pairs(endpoints):
    """Cover every Attention-DP rank, including unequal P:D role pools."""
    p = [dict(e, rank=rank) for e in endpoints if e['role'] in ('colocated', 'prefill') for rank in range(e['ep'])]
    d = [dict(e, rank=rank) for e in endpoints if e['role'] == 'decode' for rank in range(e['ep'])]
    if not d:
        return [(e, None) for e in p]
    return [(p[i % len(p)], d[i % len(d)]) for i in range(math.lcm(len(p), len(d)))]


def forward_snapshot(run, root, arm):
    """Read already-enabled CPU counters at an idle episode boundary."""
    from artifact_visibility import wait_for_stable_json
    control=Path(root['node_configs'][root['nodes'][0]]['forward_counter_control'])
    global _FORWARD_EPOCH
    _FORWARD_EPOCH = _FORWARD_EPOCH % 254 + 1
    epoch = _FORWARD_EPOCH
    from snapshot_transport import write_control
    def notify_node(node):
        if node == socket.gethostname().split('.')[0]:
            return write_control(str(control), epoch)
        command=[root['node_configs'][node]['python'],str(run/'code/snapshot_transport.py'),
                 '--source',str(control),'--epoch',str(epoch)]
        p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',node,
                          shlex.join(command)],capture_output=True,text=True,timeout=30,check=True)
        return json.loads(p.stdout)
    with cf.ThreadPoolExecutor(max_workers=len(root['nodes'])) as pool:
        receipts=list(pool.map(notify_node,root['nodes']))
    assert all(x['epoch']==epoch for x in receipts)
    # Shared byte is audit bookkeeping only; rank publishers read local tmpfs.
    fd=os.open(control,os.O_WRONLY|os.O_NOFOLLOW)
    try:os.pwrite(fd,bytes([epoch]),0)
    finally:os.close(fd)
    snapshots={}
    for node in root['nodes']:
        layout=next(x for x in root['node_configs'][node]['layouts'] if x['id']==arm)
        for index,inst in enumerate(layout['instances']):
            for pp in range(inst['pp']):
                for tp in range(inst['tp']):
                    path=run/'evidence'/node/f'{arm}.r{index}'/'forward_counters'/f'pp{pp}.tp{tp}.json'
                    deadline=time.monotonic()+30
                    while True:
                        value,_=wait_for_stable_json(path,timeout_s=30,poll_s=.03,min_stable_s=.03)
                        if value['epoch']==epoch:
                            break
                        if time.monotonic()>deadline:
                            raise TimeoutError('Forward snapshot epoch absent: '+str(path))
                        time.sleep(.03)
                    snapshots[f'{node}.{arm}.r{index}.pp{pp}.tp{tp}']=value
    assert len(snapshots)==8, 'Every GPU rank must publish an idle-boundary snapshot'
    return snapshots


def forward_delta(before, after):
    assert before.keys()==after.keys()
    delta={}
    for key,a in before.items():
        b=after[key]
        assert a['shape']==b['shape']
        counts={name:b['counts'].get(name,0)-a['counts'].get(name,0)
                for name in a['counts'].keys() | b['counts'].keys()}
        assert all(v>=0 for v in counts.values()), 'Counter reset during formal episode'
        delta[key]=dict(shape=a['shape'],counts=counts)
    return delta


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
        result[result.index('--hicache-size')+1] = str(inst['hicache_gb_per_rank'])
        result[result.index('--host')+1] = '0.0.0.0'
        if inst['role'] != 'colocated':
            result += ['--disaggregation-mode', inst['role'],
                       '--disaggregation-transfer-backend', 'mooncake',
                       '--disaggregation-bootstrap-port', str(inst['bootstrap_port']),
                       '--disaggregation-ib-device', 'mlx5_bond_0']
            if inst['role'] == 'decode':
                result += ['--disaggregation-decode-enable-radix-cache']
                if 'decode_extra_slots' in c:
                    result += ['--disaggregation-decode-extra-slots', str(c['decode_extra_slots'])]
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
    children = []
    local_root=Path(cfg['matrix_local_root'])
    local_root.mkdir(parents=True,exist_ok=True,mode=0o700)
    control=local_root/'control'
    fd=os.open(control,os.O_CREAT|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    try:os.pwrite(fd,b'\0',0)
    finally:os.close(fd)
    import matrix_hardware
    monitor_stop,monitor_thread=matrix_hardware.start(local_root)
    import matrix_memory
    memory_stop,memory_thread=matrix_memory.start(local_root,arm)

    def stop(signum, frame):
        raise InterruptedError('Owned service worker requested to clean up')

    signal.signal(signal.SIGTERM, stop)

    def parent_closed():
        sys.stdin.read()
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=parent_closed, daemon=True).start()
    try:
        base.check(run, cfg, env, True)
        key = base.h.read_key(cfg)
        from snapshot_transport import write_control
        write_control(cfg['forward_counter_control'], 0)
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
            inst = layout['instances'][i]
            alias = 'ep2' if inst['ep'] > 1 else 'pp2'
            qualification = run/'evidence'/node/f'{arm}.r{i}'/alias/'qualification'
            proofs = [json.loads(p.read_text()) for p in qualification.glob('*.json')]
            assert len(proofs) == inst['ep']*inst['pp'], 'Missing actual scheduler rank runtime qualification'
            assert all(p['passed'] for p in proofs)
            if inst['ep'] > 1:
                log_text = Path(layout['usage_logs'][i]['path']).read_text()
                expected_backend = 'none' if inst['ep'] == 4 else 'deepep'
                assert info['moe_a2a_backend'] == expected_backend
                if inst['ep'] == 4:
                    assert log_text.count('A100_NATIVE_ALLGATHER_QUALIFIED') == inst['ep']
                    assert all(p.get('transport') == 'native_nccl_allgather' for p in proofs)
                    assert log_text.count('QWEN_PRE_KV_DEEPEP_READY') == 0
                else:
                    assert log_text.count('QWEN_PRE_KV_DEEPEP_READY') == inst['ep'], 'Native DeepEP memory was not charged before KV allocation'
            if inst['role'] == 'decode' and 'decode_extra_slots' in cfg:
                assert all(s['disaggregation_decode_extra_slots'] == cfg['decode_extra_slots']
                           for s in info['internal_states'])
                launch_args = base.engine_args(cfg, arm, i)
                expected_slots = int(launch_args[launch_args.index('--max-mamba-cache-size')+1]) // inst['dp']
                log_text = Path(layout['usage_logs'][i]['path']).read_text()
                actual_slots = [int(x) for x in base.re.findall(r'Mamba Cache is allocated\. max_mamba_cache_size: (\d+)', log_text)]
                assert len(actual_slots) == inst['ep'] * inst['pp'], actual_slots
                assert set(actual_slots) == {expected_slots}, (actual_slots, expected_slots)
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
        # EOF and supervisor SIGTERM may arrive while cleanup already runs.
        # Finish the same bounded, identity-checked cleanup exactly once.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        evidence = base.h.cleanup(children) if children else {'cleanup_verified': True}
        base.h.save(run/'nodes'/node/arm/'cleanup.json', evidence)
        if not evidence['cleanup_verified']:
            raise RuntimeError('Service cleanup unverified')
        monitor_stop.set();monitor_thread.join(timeout=8)
        memory_stop.set();memory_thread.join(timeout=8)
        export=run/'telemetry'/node
        import matrix_workload
        command=[cfg['python'],str(run/'code/matrix_node.py'),'--root',cfg['matrix_local_root'],'--level','0','--export',str(export)]
        subprocess.run(command,check=True,timeout=60,capture_output=True)
        from snapshot_transport import remove_control
        remove_control(cfg['forward_counter_control'])


def finish_episode(base, run, root, arm, out, repeat, before, result):
    # Preserve completed client measurements before any optional diagnostics.
    base.h.save(out/f'replay_completed.r{repeat+1}.json', result)
    after = forward_snapshot(run, root, arm)
    base.h.save(out/f'forwards.r{repeat+1}.json', dict(
        before=before, after=after, delta=forward_delta(before, after),
        protocol='Existing per-rank counters; node-local notification and idle snapshots outside formal timing.'))
    (out/f'replay_completed.r{repeat+1}.json').replace(out/f'formal.r{repeat+1}.json')
    base.h.save(out/f'completed.r{repeat+1}.json', {'passed': True, 'requests': result['completed_requests']})


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
    # An explicit, untimed cold-prefix and prefix-reuse pass on every DP rank.
    # The six serial correctness probes alone only touch a subset of ranks.
    pairs = warmup_pairs(endpoints)
    warmup = []
    for warm_case in root['warmup_cases']:
        length=warm_case['input_tokens']
        width=warm_case['requests_per_attention_rank']
        run_pairs=pairs*width
        flush()
        prompts = [([8000+i] + body*(length//len(body)+2))[:length] for i in range(len(run_pairs))]
        for phase in ('cold', 'prefix_reuse'):
            health()
            with cf.ThreadPoolExecutor(max_workers=len(run_pairs)) as pool:
                futures = [pool.submit(replay.request_one, p, d, key,
                    tokens if phase == 'cold' else tokens+(body*64)[:64], 32, i)
                    for i, ((p, d), tokens) in enumerate(zip(run_pairs, prompts))]
                responses = [future.result() for future in futures]
            warmup.append(dict(input_length=length, requests_per_attention_rank=width, phase=phase, responses=responses))
    base.h.save(run/'evidence'/arm/'untimed_warmup.json', dict(
        passed=True, pairs=pairs, cases=warmup,
        protocol='All instance/Attention-DP ranks; cold and prefix-reuse requests at declared lengths and offered widths; actual batch sizes remain scheduler-dependent. Excluded from performance; native cache flushed before formal episodes.'))
    graph_data = json.loads(Path(root['graphs_file']).read_text())
    for case in root['cases']:
        graphs = [g for g in graph_data['graphs'] if g['id'] in case['graph_ids']]
        assert len(graphs) == len(case['graph_ids'])
        policies = root['policies'].get(arm, ['sticky_hash'])
        for policy in policies:
            for repeat in range(root['repeats']):
                out = run/'results'/arm/case['id']/policy
                relative = str((out/f'formal.r{repeat+1}.json').relative_to(run))
                imported = root.get('imported_episodes', {}).get(relative)
                if imported:
                    for artifact in imported['artifacts']:
                        p = run/artifact['relative_path']
                        assert hashlib.sha256(p.read_bytes()).hexdigest() == artifact['sha256']
                    continue
                health()
                flush()
                out = run/'results'/arm/case['id']/policy
                base.h.save(run/'task/progress.json', dict(experiment_id=root['id'], task_id='comparison',
                    stage=f'{arm}:{case["id"]}:{policy}:r{repeat+1}', timestamp=time.time(),
                    valid_points=len(list(run.glob('results/*/*/completed.json'))), expected=root['expected_points']))
                before = forward_snapshot(run, root, arm)
                result = replay_with_failure_capture(replay.replay, base.h.save, out/f'partial.r{repeat+1}.json',
                    graphs, endpoints, policy, case['concurrency'], key, body, health,
                    namespace=f'agentx-{case["id"]}-repeat{repeat+1}', idle_gap_cap_s=root['idle_gap_cap_s'], copies=case.get('copies', 1), episode_timeout_s=root['episode_timeout_s'])
                finish_episode(base, run, root, arm, out, repeat, before, result)
    return 0


def controller(run, root):
    base, first = load_base(run, root, root['nodes'][0])
    base.check(run, first, os.environ)
    packet = {'environment': {k: os.environ[k] for k in base.h.GUARD_KEYS}}
    base.h.save(run/'task/controller_identity.json', {'pid': os.getpid(), 'started': time.time()})
    failed = None
    try:
        total_started = time.monotonic()
        for arm in root['arms']:
            if arm in root.get('imported_arms', {}):
                assert json.loads((run/'results'/arm/'metrics.json').read_text())['qualified']
                assert json.loads((run/'results'/arm/'completed.json').read_text())['passed']
                continue
            if time.monotonic()-total_started > root['total_timeout_seconds']:
                raise TimeoutError('Total experiment budget exhausted before next arm')
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
                        stage='loading_'+arm, timestamp=time.time(), valid_points=len(list(run.glob('results/*/*/completed.json'))), expected=root['expected_points']))
                    time.sleep(2)
                with (run/f'coordinator.{arm}.log').open('ab', buffering=0) as log:
                    command = [first['python'], str(Path(__file__).resolve()), '--run', str(run), '--coordinator', '--arm', arm]
                    coordinator_process = subprocess.Popen(['ssh', '-T', '-o', 'BatchMode=yes', root['nodes'][0], shlex.join(command)], stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
                    coordinator_process.stdin.write((json.dumps(packet)+'\n').encode()); coordinator_process.stdin.flush()
                    while coordinator_process.poll() is None:
                        if time.monotonic()-total_started > root['total_timeout_seconds']:
                            raise TimeoutError('Total experiment wall time exhausted')
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
            except Exception as error:
                base.h.save(run/'failures'/f'{arm}.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
                if arm.startswith('Q_') or arm == 'M_PP2':
                    raise
            finally:
                if coordinator_process is not None:
                    coordinator_process.stdin.close()
                    if coordinator_process.poll() is None:
                        coordinator_process.terminate()
                        coordinator_process.wait(timeout=45)
                for _, process, _ in processes:
                    process.stdin.close()
                for node, process, log in processes:
                    try:
                        process.wait(timeout=85)
                    except subprocess.TimeoutExpired:
                        process.terminate(); process.wait(timeout=5)
                    log.close()
                    from artifact_visibility import wait_for_stable_json
                    evidence, _ = wait_for_stable_json(run/'nodes'/node/arm/'cleanup.json', timeout_s=30)
                    if not evidence['cleanup_verified']:
                        raise RuntimeError('Unverified remote cleanup')
    except BaseException as error:
        failed = type(error).__name__
        raise
    finally:
        evidence = [json.loads(p.read_text()) for p in run.glob('nodes/*/*/cleanup.json')]
        attempted=list(run.glob('nodes/*/*/worker.log'))
        passed = bool(evidence) and len(evidence)==len(attempted) and all(e.get('cleanup_verified') for e in evidence)
        base.h.save(run/'task/progress.json', dict(experiment_id=root['id'], task_id='comparison',
            stage='failed' if failed else 'complete', error=failed, timestamp=time.time(),
            valid_points=len(list(run.glob('results/*/*/completed.json'))), expected=root['expected_points']))
        base.h.save(run/'task/cleanup.json', dict(experiment_id=root['id'], task_id='comparison',
            timestamp=time.time(), passed=passed, gpu_idle=passed, model_processes_exited=passed,
            source='Owned two-node worker process groups', worker_evidence_count=len(evidence)))
    return 0


import official_workload
official_workload.install(globals())

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
