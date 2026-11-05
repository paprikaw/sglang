"""Guard-owned full-model A100 regression, not a performance benchmark.

Prepare an external runtime directory, then run this controller through the
canonical dashboard_guard. No allocation, cache builder or foreign PID control.
Runtime profiles are explicit: native BF16 adapter, optional PP2 ready/gap hooks.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
from functools import wraps
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT.parents[1]
PROFILE = ROOT / 'runtime_profiles/agentx_a100'
NATIVE = ROOT / 'runtime_profiles/a100_native_ep'
sys.path.insert(0, str(PROFILE))
import process_helpers as h


def desc(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def git(*args):
    return subprocess.check_output(['git', '-C', str(ENGINE), *args], text=True).strip()


def prepare(args):
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    if git('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('Commit tracked changes before freezing GPU qualification')
    commit = git('rev-parse', 'HEAD')
    if git('rev-parse', args.tag + '^{commit}') != commit or git('cat-file', '-t', args.tag) != 'tag':
        raise RuntimeError('An annotated tag at current HEAD is required')
    guard = Path('/home/bxb1/experiment-guard/skills/llm-experiment-guard/scripts')
    state = json.loads(Path(args.backfill_state).read_text())
    old = Path(state['nodes'][args.node]['run']) / 'cache_node_spec.json'
    node_spec = json.loads(old.read_text())
    if node_spec['node'] != args.node or str(node_spec['expected_job_id']) != args.job:
        raise RuntimeError('Existing CPU bank differs from requested booked node/job')
    if '35B' not in node_spec['model_path']:
        raise RuntimeError('This regression requires the existing native BF16 35B bank')
    (run / 'cache').mkdir()
    h.save(run / 'cache/node.json', node_spec)
    layouts = []
    definitions = [('ep2', 2, 1, [0, 1], None), ('ep4', 4, 1, [0, 1, 2, 3], None),
                   ('pp2', 1, 2, [0, 1], None), ('pd_p', 2, 1, [0, 1], 'prefill'),
                   ('pd_d', 1, 2, [2, 3], 'decode')]
    for i, (name, ep, pp, gpus, role) in enumerate(definitions):
        path = run / name
        path.mkdir()
        (path / 'pp2').mkdir()
        for file, value in [('ready.bin', b'1'), ('gap.bin', b'0'), ('counters.bin', b'\0')]:
            (path / file).write_bytes(value)
            (path / file).chmod(0o600)
        ranks = [f'PP{p} TP{t}' if pp > 1 else f'TP{t}'
                 for p in range(pp) for t in range(ep)]
        use = dict(id=run.name + '.' + name, path=str(run / 'cache' / (name + '.log')),
                   expected_ranks=ranks)
        layouts.append(dict(id=name, ep=ep, tp=ep, dp=ep, pp=pp, gpus=gpus,
                            role=role, port=24761 + i, dist_port=24861 + i,
                            usage_logs=[use]))
    sys.path.insert(0, str(guard))
    import cpu_memory_budget
    frozen = cpu_memory_budget.dependency_identity()['files']
    cache = dict(policy='required', provider='sglang_canonical_full_weight_v1',
        support_profile='qwen3_5moe-canonical-v1', on_miss='fail',
        node_specs={args.node: desc(run / 'cache/node.json')},
        usage_logs={args.node: [x['usage_logs'][0] for x in layouts]},
        hicache_reserve_bytes_per_node={args.node: 20 * 1024**3},
        cpu_memory_budget=dict(schema_version=1, kv_banks_per_node={args.node: []}))
    cfg = dict(id=run.name, node=args.node, job_id=args.job, guard_scripts=str(guard),
        engine_root=str(ENGINE), engine_commit=commit, engine_tag=args.tag,
        python=args.python, model_path=node_spec['model_path'],
        layouts=layouts, admin_key_file=str(run / 'admin.key'), model_cache=cache)
    h.save(run / 'config.json', cfg)
    (run / 'admin.key').write_text(secrets.token_urlsafe(32))
    (run / 'admin.key').chmod(0o600)
    files = [Path(__file__), run / 'config.json', run / 'cache/node.json',
             ROOT / 'dependencies.lock.json', ROOT / 'runtime_sources.json']
    files += sorted(PROFILE.glob('*.py')) + sorted(NATIVE.glob('*.py'))
    frozen += [desc(p) for p in files]
    (run / 'task').mkdir()
    spec = dict(schema_version=1, id=run.name, title='Integrated SGLang: A100 EP / PP / P-D regression',
        engine='sglang', protocol='task-events-v2', template='parallel-tasks-v5', port=24760,
        runtime_dir=str(run / 'dashboard_guard'), resource_priority='normal',
        features=['parallel-tasks', 'task-progress', 'cpu-model-cache', 'gpu-telemetry'],
        max_active_nodes=1, gpu_lease_ttl_s=30,
        gpu_registry_dir='/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/gpu_registry',
        model_cache=cache, frozen_inputs=frozen,
        sources=[dict(role='inference_engine', worktree=str(ENGINE), commit=commit, tag=args.tag)],
        tasks=[dict(id='regression', label='EP2, EP4, PP2 and EP2-P to PP2-D',
            nodes=[args.node], root=str(run / 'task'), controller_script=str(Path(__file__)), expected=4)])
    h.save(run / 'dashboard_spec.json', spec)
    h.save(run / 'task/progress.json', dict(experiment_id=run.name, task_id='regression',
        stage='queued', valid_points=0, expected=4, timestamp=time.time()))
    print(json.dumps(dict(run=str(run), guard=str(guard / 'dashboard_guard.py'))))


def environment(cfg, row, run):
    env = {k: v for k, v in os.environ.items() if not k.startswith((
        'SGLANG_', 'VLLM_', 'DEEPEP_', 'NVSHMEM_', 'NCCL_', 'GLOO_', 'HUMMING_'))
        and k not in ('PYTHONPATH', 'LD_PRELOAD', 'CUDA_VISIBLE_DEVICES')}
    deps = json.loads((ROOT / 'dependencies.lock.json').read_text())
    cuda = '/apps/easybuild-2022/easybuild/software/Core/CUDA/12.8.0'
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, row['gpus'])),
        PYTHONPATH=os.pathsep.join([str(ENGINE / 'python'), str(PROFILE),
            str(Path(deps['deepep']['native_path']).parent), deps['deepep']['wrapper_path'],
            deps['mooncake']['python_path']]),
        PATH=f"{Path(cfg['python']).parent}:{cuda}/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        LD_LIBRARY_PATH=f'/home/bxb1/conda/lib:{cuda}/lib64:/apps/easybuild-2022/easybuild/software/Core/OpenSSL/1.1/lib',
        CUDA_HOME=cuda, CUDA_MODULE_LOADING='EAGER', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='1',
        NCCL_CUMEM_HOST_ENABLE='0', NCCL_IB_HCA='mlx5_bond_0', NCCL_SOCKET_IFNAME='bond0.3027',
        GLOO_SOCKET_IFNAME='bond0.3027', SGLANG_NUMA_BIND_V2='0',
        MC_GID_INDEX='5', MC_TCP_ENABLE_CONNECTION_POOL='true',
        PP_READY_POLICY_PATH=str(run / row['id'] / 'ready.bin'),
        PP_GAP_POLICY_PATH=str(run / row['id'] / 'gap.bin'),
        PP_GAP_SEPARATE_CHANNELS='1' if row['id'] == 'pp2' else '0',
        BENCH_FORWARD_COUNTER_CONTROL=str(run / row['id'] / 'counters.bin'),
        EP2_PP2_RUN=str(run / row['id']), EP2_PP2_ARM='pp2' if row['pp'] > 1 else 'ep2',
        INTEGRATION_SERVER='1', INTEGRATION_RUN=str(run), INTEGRATION_LAYOUT=row['id'])
    return env


def server_hooks():
    # Spawn imports this entry point too: install before constructing workers.
    import torch
    spec = importlib.util.spec_from_file_location('integration_native_adapter', NATIVE / 'a100_runtime_adapter.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    sys.path.insert(0, str(NATIVE))
    adapter.install()
    import runtime_setup
    runtime_setup.install()
    sys.path.remove(str(NATIVE))
    import cache_probe, forward_observer
    cache_probe.install()
    forward_observer.install()
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    original = DecodeCudaGraphRunner.execute
    @wraps(original)
    def execute(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        forward_observer._COUNTS['actual_decode_graph_execute'] += 1
        return result
    DecodeCudaGraphRunner.execute = execute
    if os.environ['EP2_PP2_ARM'] == 'pp2':
        import pp_ready_send
        pp_ready_send.install(torch)
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
        original_send = SchedulerPPMixin._pp_send_dict_to_next_stage
        @wraps(original_send)
        def send(self, tensors, *args, **kwargs):
            result = original_send(self, tensors, *args, **kwargs)
            if kwargs.get('msg_type', 'default') in ('proxy', 'output') and any(
                    getattr(t, 'is_cuda', False) for t in tensors.values()):
                forward_observer._COUNTS['pp_producer_ready_send'] += 1
            return result
        SchedulerPPMixin._pp_send_dict_to_next_stage = send
        if os.environ['INTEGRATION_LAYOUT'] == 'pp2':
            import gap_schedule
            gap_schedule.install()


if os.environ.get('INTEGRATION_SERVER') == '1':
    server_hooks()


def engine_entry(cfg, row):
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.entrypoints.http_server import launch_server
    ep, pp = row['ep'], row['pp']
    args = ['--model-path', cfg['model_path'], '--host', '127.0.0.1', '--port', str(row['port']),
        '--dist-init-addr', f"127.0.0.1:{row['dist_port']}", '--tp-size', str(ep),
        '--ep-size', str(ep), '--dp-size', str(ep), '--pp-size', str(pp),
        '--dtype', 'bfloat16', '--kv-cache-dtype', 'bfloat16', '--context-length', '8192',
        '--mem-fraction-static', '.90', '--page-size', '64', '--max-running-requests', str(16 * ep),
        '--max-mamba-cache-size', str(64 * ep * pp), '--mamba-ssm-dtype', 'float32',
        '--mamba-max-states-per-path', '1', '--max-total-tokens', '32768',
        '--chunked-prefill-size', str(2048 * ep), '--max-prefill-tokens', str(4096 * ep),
        '--attention-backend', 'triton', '--linear-attn-backend', 'triton',
        '--moe-runner-backend', 'triton', '--moe-a2a-backend', 'none',
        '--cuda-graph-bs-decode', '1', '2', '4', '8', '16',
        '--cuda-graph-backend-prefill', 'disabled', '--disable-custom-all-reduce',
        '--skip-server-warmup', '--json-model-override-args', '{"language_model_only":true}',
        '--enable-cache-report', '--enable-metrics', '--enable-metrics-for-all-schedulers',
        '--enable-hierarchical-cache', '--hicache-size', '4', '--hicache-host-memory-mode', 'cache',
        '--hicache-write-policy', 'write_through', '--hicache-io-backend', 'kernel',
        '--hicache-mem-layout', 'page_first']
    if ep > 1:
        args += ['--enable-dp-attention']
    if pp > 1:
        args += ['--pp-max-micro-batch-size', '4', '--pp-async-batch-depth', '0']
    if row['role']:
        args += ['--disaggregation-mode', row['role'], '--disaggregation-transfer-backend', 'mooncake',
            '--disaggregation-bootstrap-port', '24980', '--disaggregation-ib-device', 'mlx5_bond_0']
        if row['role'] == 'decode':
            args += ['--disaggregation-decode-enable-radix-cache', '--disaggregation-decode-extra-slots', '8']
    server_args = ServerArgs.from_cli_args(args)
    key = h.read_key(cfg)
    server_args.api_key = key
    server_args.admin_api_key = key
    launch_server(server_args)


def launch(run, cfg, row, key):
    command = [cfg['python'], str(Path(cfg['guard_scripts']) / 'model_cache_adapter.py'),
        'exec-local', '--bindings', os.environ[h.GUARD_KEYS[3]], '--', cfg['python'],
        str(Path(__file__)), '--engine-entry', '--run', str(run), '--layout', row['id']]
    proc = subprocess.Popen(command, env=environment(cfg, row, run), cwd=ENGINE,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    identity = h.process_record(proc.pid)
    if not identity or identity['pgid'] != proc.pid or identity['session'] != proc.pid:
        proc.terminate()
        raise RuntimeError('Could not establish owned engine process group')
    thread = threading.Thread(target=h.stream_log,
        args=(proc.stdout, row['usage_logs'][0]['path'], key), daemon=True)
    thread.start()
    return dict(process=proc, identity=identity, log_thread=thread)


def worker(run, cfg):
    h.authenticate(run, cfg, os.environ, local=True)
    key = h.read_key(cfg)
    import agentx_replay
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
    body = tokenizer.encode('Explain how a pipeline preserves request order and reuses cached state. ' * 45)
    prompts = [tokenizer.encode(f'Example {i}. ') + body[:384] for i in range(8)]
    results = {}
    def stopped(*_):
        raise KeyboardInterrupt('Guard stopped regression')
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    final_cleanup = None
    try:
        for name in ['ep2', 'ep4', 'pp2', 'pd']:
            rows = [x for x in cfg['layouts'] if x['id'] in (['pd_p', 'pd_d'] if name == 'pd' else [name])]
            children = []
            try:
                for row in rows:
                    children.append(launch(run, cfg, row, key))
                for row, child in zip(rows, children):
                    deadline = time.monotonic() + 600
                    while True:
                        h.authenticate(run, cfg, os.environ, local=True)
                        if child['process'].poll() is not None:
                            raise RuntimeError('Engine exited: ' + row['id'])
                        try:
                            h.http_json(row['port'], '/get_server_info', key, timeout=3)
                            break
                        except (OSError, ValueError):
                            if time.monotonic() > deadline:
                                raise TimeoutError('Engine startup: ' + row['id'])
                            time.sleep(1)
                def endpoint(row):
                    return dict(host='127.0.0.1', port=row['port'], ep=row['ep'], bootstrap_port=24980)
                def cohort():
                    def one(i):
                        return agentx_replay.request_one(endpoint(rows[0]),
                            endpoint(rows[1]) if name == 'pd' else None,
                            key, prompts[i], 24, i, return_logprob=True)
                    with cf.ThreadPoolExecutor(max_workers=8) as pool:
                        replies = list(pool.map(one, range(8)))
                    for reply in replies:
                        if len(reply.get('output_ids') or []) != 24:
                            raise AssertionError('Missing generated token IDs')
                    return replies
                cold = cohort()
                warm = cohort()
                evidence = dict(cold=cold, warm=warm)
                # Repeat after an explicit CPU restore, keeping normal HiCache on.
                for row in rows:
                    for _ in range(60):
                        answer = h.http_json(row['port'], '/set_internal_state', key,
                            {'server_args': {'agentx_cache_cmd': 2}}, timeout=20)
                        if answer and all(x is True for x in answer):
                            break
                        time.sleep(.2)
                    else:
                        raise RuntimeError('Could not demote idle GPU prefixes')
                restored = cohort()
                evidence['restored'] = restored
                if [x['output_ids'] for x in warm] != [x['output_ids'] for x in restored]:
                    raise AssertionError('Output changed after CPU cache restore')
                if name == 'pp2':
                    (run / 'pp2/gap.bin').write_bytes(b'9')
                    optimized = cohort()
                    evidence['gap9'] = optimized
                    if [x['output_ids'] for x in warm] != [x['output_ids'] for x in optimized]:
                        raise AssertionError('Output changed with PP lookahead')
                evidence['server_info'] = []
                evidence['counters'] = []
                for row in rows:
                    info = h.http_json(row['port'], '/get_server_info', key, timeout=30)
                    evidence['server_info'].append(info)
                    (run / row['id'] / 'counters.bin').write_bytes(b'\1')
                    expected = row['ep'] * row['pp']
                    deadline = time.monotonic() + 20
                    while True:
                        files = list((run / row['id'] / 'forward_counters').glob('*.json'))
                        if len(files) == expected and all(json.loads(p.read_text()).get('epoch') == 1 for p in files):
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError('Missing rank counters: ' + row['id'])
                        time.sleep(.1)
                    counters = [json.loads(p.read_text()) for p in files]
                    if row['role'] != 'prefill' and not all(x['counts'].get('actual_decode_graph_execute', 0) > 0 for x in counters):
                        raise AssertionError('Decode Graph never executed: ' + row['id'])
                    evidence['counters'].append(dict(layout=row['id'], ranks=counters))
                    if not any(e.get('loaded_tokens', 0) > 0 for s in info['internal_states'] for e in s.get('agentx_restores', [])):
                        raise AssertionError('CPU restore not exercised: ' + row['id'])
                evidence['passed'] = True
                results[name] = evidence
                h.save(run / 'results.json', results)
                h.save(run / 'worker_progress.json', dict(completed=len(results), current=name, timestamp=time.time()))
                print(json.dumps(dict(layout=name, passed=True)), flush=True)
            finally:
                final_cleanup = h.cleanup(children)
                h.save(run / (name + '.cleanup.json'), final_cleanup)
                if not final_cleanup['cleanup_verified']:
                    raise RuntimeError('Owned model cleanup could not be verified')
    finally:
        if final_cleanup:
            h.save(run / 'worker_cleanup.json', final_cleanup)


def controller(run, cfg):
    h.authenticate(run, cfg, os.environ)
    if git('rev-parse', 'HEAD') != cfg['engine_commit'] or git('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('Frozen source changed')
    env = ['env'] + [k + '=' + os.environ[k] for k in h.GUARD_KEYS]
    command = env + [cfg['python'], str(Path(__file__)), '--worker', '--run', str(run)]
    proc = subprocess.Popen(['ssh', '-oBatchMode=yes', cfg['node'], shlex.join(command)])
    identity = h.process_record(os.getpid())
    def stop(*_):
        proc.terminate()
        raise KeyboardInterrupt('Guard stopped controller')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    code = 1
    try:
        while proc.poll() is None:
            h.authenticate(run, cfg, os.environ)
            progress = json.loads((run / 'worker_progress.json').read_text()) if (run / 'worker_progress.json').exists() else {}
            h.save(run / 'task/progress.json', dict(experiment_id=cfg['id'], task_id='regression',
                stage='running', valid_points=progress.get('completed', 0), expected=4,
                timestamp=time.time(), controller_pid=os.getpid(), process_start_ticks=identity['start_ticks']))
            time.sleep(2)
        code = proc.returncode
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=65)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        completed = len(json.loads((run / 'results.json').read_text())) if (run / 'results.json').exists() else 0
        h.save(run / 'task/progress.json', dict(experiment_id=cfg['id'], task_id='regression',
            stage='complete' if code == 0 and completed == 4 else 'failed', valid_points=completed,
            expected=4, timestamp=time.time()))
        cleanup = json.loads((run / 'worker_cleanup.json').read_text()) if (run / 'worker_cleanup.json').exists() else {}
        h.save(run / 'task/cleanup.json', dict(experiment_id=cfg['id'], task_id='regression', timestamp=time.time(),
            passed=cleanup.get('cleanup_verified', False), gpu_idle=cleanup.get('node_gpu_idle_observed', False),
            model_processes_exited=cleanup.get('process_groups_exited', False), remote_evidence=cleanup))
    raise SystemExit(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--engine-entry', action='store_true')
    ap.add_argument('--layout')
    ap.add_argument('--node', default='spartan-gpgpu168')
    ap.add_argument('--job', default='33463293')
    ap.add_argument('--tag', default='experiment/sglang/pp-elastic-integration-20261105-r1')
    ap.add_argument('--python', default='/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python')
    ap.add_argument('--backfill-state', default='/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/cluster_backfill_20261020/all_nodes_state.json')
    args = ap.parse_args()
    if args.prepare:
        return prepare(args)
    cfg = json.loads((args.run / 'config.json').read_text())
    if args.engine_entry:
        h.authenticate(args.run, cfg, os.environ, local=True)
        return engine_entry(cfg, next(x for x in cfg['layouts'] if x['id'] == args.layout))
    if args.worker:
        return worker(args.run, cfg)
    return controller(args.run, cfg)


if __name__ == '__main__':
    main()
