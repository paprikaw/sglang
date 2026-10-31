"""Freeze equal-eight-H100 AgentX comparisons; never starts a model itself."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[1]
NODES = ['spartan-gpgpu011', 'spartan-gpgpu171']
SEEDS = ['011_ep2_host_r3', '171_pp2_fix_r3']


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def desc(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def replace(value, left, right):
    if isinstance(value, str):
        return value.replace(left, right)
    if isinstance(value, list):
        return [replace(v, left, right) for v in value]
    if isinstance(value, dict):
        return {k: replace(v, left, right) for k, v in value.items()}
    return value


def shapes(arm, node_index):
    ep4 = [(4, 1, [0, 1, 2, 3])]
    ep2 = [(2, 1, [0, 1]), (2, 1, [2, 3])]
    pp2 = [(1, 2, [0, 1]), (1, 2, [2, 3])]
    single = [(1, 1, [i]) for i in range(4)]
    if arm.startswith('col_'):
        return {'col_ep4': ep4, 'col_ep2': ep2, 'col_pp2': pp2, 'col_single': single}[arm], 'colocated'
    choices = {
        'pd_ep4': [ep4, ep4], 'pd_ep2': [ep2, ep2], 'pd_pp2': [pp2, pp2],
        'pd_pp2_ep4': [pp2, ep4], 'pd_ep4_single': [ep4, single],
        'pd_pp2_single': [pp2, single],
        'pd_mixed': [pp2, [(2, 1, [0, 1]), (1, 2, [2, 3])]],
    }
    return choices[arm][node_index], 'prefill' if node_index == 0 else 'decode'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--arms', default='col_ep4,pd_ep4')
    ap.add_argument('--port', type=int, default=30221)
    ap.add_argument('--dashboard-port', type=int, default=28721)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--concurrencies', default='2,8')
    ap.add_argument('--engine', type=Path)
    ap.add_argument('--tag')
    ap.add_argument('--reference', type=Path)
    args = ap.parse_args()
    run = args.run.resolve()
    assert not run.exists()
    assert args.repeats >= 3
    seed = ROOT/'runs'/SEEDS[0]
    cfg0 = read(seed/'config.json')
    engine = args.engine.resolve() if args.engine else Path(cfg0['engine_root'])
    commit = subprocess.check_output(['git', '-C', str(engine), 'rev-parse', 'HEAD'], text=True).strip()
    assert not subprocess.check_output(['git', '-C', str(engine), 'status', '--porcelain'], text=True).strip()
    tag = args.tag or cfg0['engine_tag']
    assert subprocess.check_output(['git', '-C', str(engine), 'rev-parse', tag+'^{commit}'], text=True).strip() == commit
    subprocess.run(['python', '/home/bxb1/.codex/skills/llm-experiment-guard/scripts/backfill_trigger.py'], check=True, stdout=subprocess.DEVNULL)
    run.mkdir(parents=True)
    (run/'cache').mkdir()
    shutil.copytree(seed/'code', run/'code')
    for name in ['cluster_runner.py', 'agentx_replay.py']:
        shutil.copyfile(ROOT/name, run/'code'/name)
    # Node-unique diagnostic paths; server IDs and usage logs are also unique.
    mixed = run/'code/mixed_workload.py'
    text = mixed.read_text()
    assert "evidence=run/'evidence'/" in text
    mixed.write_text(text.replace("evidence=run/'evidence'/", "evidence=run/'evidence'/cfg['node']/"))
    for path in seed.glob('*.bin'):
        shutil.copyfile(path, run/path.name)
    shutil.copyfile(seed/'active_workload.txt', run/'active_workload.txt')
    shutil.copyfile(ROOT/'agentx_graph_candidates.json', run/'agentx_graphs.json')
    if args.reference:
        shutil.copyfile(args.reference, run/'reference.input.json')
        reference = read(args.reference)
        save(run/'evidence/reference.json', reference)
    arms = args.arms.split(',')
    root = dict(id=run.name, nodes=NODES, arms=arms, repeats=args.repeats,
                graphs_file=str(run/'agentx_graphs.json'), idle_gap_cap_s=10,
                policies={'pd_mixed': ['sticky_hash', 'least_outstanding', 'cache_queue']})
    # Entire real sessions, including two captured subagent families. Selection
    # was frozen without looking at any eight-GPU benchmark outcomes.
    graph_ids = [
        '1493faffdc8942e99daa22277f44871a40b8',
        'bbdcb12440a7ab3496b9fac8b5f9824b1672',
        'd876a71a07454e7499469e864f89cf6a3345',
        '5c5e408b76e5e22747853915f67be3c491a4',
        'df2638f16306b1a10635bdb8e3acf7de0a61',
        '12cfecf98c158ab54fb65980602421f00d69',
        '0e4206535a79f980e2b1091181202ea987fe',
        'a3344328ef1674cda1018e0791fb85a839a8',
    ]
    root['cases'] = [dict(id='agentx_c'+c, concurrency=int(c), graph_ids=graph_ids,
                          copies=2 if int(c) > 8 else 1) for c in args.concurrencies.split(',')]
    root['expected_points'] = args.repeats*len(root['cases'])*sum(len(root['policies'].get(arm, ['sticky_hash'])) for arm in arms)
    root['node_configs'] = {}
    root['endpoints'] = {arm: [] for arm in arms}
    model_cache = read(seed/'config.json')['model_cache']
    for field in ('node_specs', 'usage_logs', 'hicache_reserve_bytes_per_node'):
        model_cache[field] = {}
    model_cache['cpu_memory_budget']['kv_banks_per_node'] = {node: [] for node in NODES}
    for n, node in enumerate(NODES):
        node_seed = ROOT/'runs'/SEEDS[n]
        cfg = replace(read(node_seed/'config.json'), str(node_seed), str(run))
        cfg = replace(cfg, cfg['engine_root'], str(engine))
        cfg.update(id=run.name, engine_commit=commit, engine_tag=tag, port=args.port,
                   dist_port=args.port+2000, startup_timeout_s=1200,
                   expected_points=root['expected_points'], request_timeout_s=600)
        cfg['profile'].update(max_context=262144, max_requests=32, mamba_equiv=40)
        cfg.pop('colocated_reference', None)
        cfg['layouts'] = []
        for a, arm in enumerate(arms):
            shape, role = shapes(arm, n)
            instances, usage = [], []
            for replica, (ep, pp, gpus) in enumerate(shape):
                bootstrap_port = args.port+5000+a*100+replica*10
                instance = dict(ep=ep, tp=ep, dp=ep, pp=pp, gpus=gpus, role=role, bootstrap_port=bootstrap_port)
                instances.append(instance)
                use = dict(id=f'{run.name}.{node}.{arm}.r{replica}',
                           path=str(run/'cache'/node/f'{arm}.r{replica}.log'),
                           expected_ranks=[f'PP{s} TP0' for s in range(pp)] if pp > 1 else [f'TP{s}' for s in range(ep)])
                if ep == pp == 1:
                    use['single_instance_tp1'] = True
                usage.append(use)
                root['endpoints'][arm].append(dict(id=f'{node}.{arm}.r{replica}', host=node,
                    port=args.port+a*100+replica*10, ep=ep, pp=pp, role=role, bootstrap_port=bootstrap_port))
            cfg['layouts'].append(dict(id=arm, replicas=len(instances), instances=instances, usage_logs=usage))
        node_spec = run/'cache'/f'{node}.json'
        shutil.copyfile(node_seed/'cache/node.json', node_spec)
        model_cache['node_specs'][node] = desc(node_spec)
        model_cache['usage_logs'][node] = [u for row in cfg['layouts'] for u in row['usage_logs']]
        model_cache['hicache_reserve_bytes_per_node'][node] = 4*32*1024**3
        root['node_configs'][node] = cfg
    for cfg in root['node_configs'].values():
        cfg['model_cache'] = model_cache
    root['model_cache'] = model_cache
    root['protocol'] = ('Two existing 4xH100 nodes, eight GPUs per arm; entire selected AgentX sessions with official DAG reconstruction; '
                        'native model, native P/D KV+GDN handoff, HiCache 32 decimal GB/rank and GPU cache auto-sized at .95; '
                        'no turn-boundary flush, no speculation, no token-length truncation; native PP policy0; '
                        'three independent repeats, arithmetic mean and all raw requests; only global-idle gaps compressed to 10s; '
                        'not official AgentX score, no in-place weight redistribution claim.')
    save(run/'config.json', root)
    ds = replace(read(seed/'dashboard_spec.json'), str(seed), str(run))
    ds.update(id=run.name, title='AgentX eight-H100 colocated versus real PD', max_active_nodes=2,
              port=args.dashboard_port, model_cache=model_cache)
    ds['tasks'] = [dict(id='comparison', label='Full AgentX DAG episodes on eight H100', nodes=NODES,
                        root=str(run/'task'), controller_script=str(run/'code/cluster_runner.py'), expected=root['expected_points'])]
    for source in ds['sources']:
        if source['role'] == 'inference_engine':
            source.update(worktree=str(engine), commit=commit, tag=tag)
    # Preserve the canonical Guard/cache native provenance, replacing seed-run
    # mutable artifacts by the complete new bundle and actual engine patch files.
    files = [Path(item['path']) for item in read(seed/'dashboard_spec.json')['frozen_inputs']
             if not item['path'].startswith(str(seed)) and not item['path'].startswith(str(ROOT))
             and not item['path'].startswith(str(Path(cfg0['engine_root'])))]
    files += list((run/'code').glob('*.py'))+[run/'config.json', run/'agentx_graphs.json', Path(__file__).resolve(),
                                            ROOT/'cluster_runner.py', ROOT/'agentx_replay.py', ROOT/'export_agentx_graphs.py']
    files += [Path(x['path']) for x in model_cache['node_specs'].values()]
    files += [engine/p for p in subprocess.check_output(['git','-C',str(engine),'diff','--name-only',
                 '0f2f1990994774472b1089d2b0682f0bf926a6b4','HEAD'],text=True).splitlines()]
    if args.reference:
        files += [run/'reference.input.json']
    ds['frozen_inputs'] = list({str(p): desc(p) for p in files}.values())
    save(run/'dashboard_spec.json', ds)
    save(run/'launch_manifest.json', dict(id=run.name, engine_root=str(engine), engine_commit=commit, engine_tag=tag,
        model_revision=cfg0['model_revision'], model_cache=model_cache, nodes=NODES, arms=arms, endpoints=root['endpoints'],
        protocol=root['protocol'], cases=root['cases'], policies=root['policies'], frozen_inputs=ds['frozen_inputs']))
    save(run/'task/progress.json', dict(experiment_id=run.name, task_id='comparison', timestamp=time.time(),
                                      stage='queued', valid_points=0, expected=root['expected_points']))
    for path in (run/'code').glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    print(json.dumps(dict(run=str(run), arms=arms, expected=root['expected_points'], frozen_files=len(files))))


if __name__ == '__main__':
    main()
