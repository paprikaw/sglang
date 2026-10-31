"""Freeze two eight-H100 2P6D pilots, preserving the qualified E7/V4 harness.

No GPU operation is performed here. The frozen Qwen Guard owns admission.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SEED = ROOT/'runs/eight_gpu_ep4_queue32_e7_v4_b7200_r1'
TRIGGER = Path('/home/bxb1/.codex/skills/llm-experiment-guard/scripts/backfill_trigger.py')
GUARD = Path(json.loads((SEED/'config.json').read_text())['node_configs']['spartan-gpgpu011']['guard_scripts'])
NODES = ['spartan-gpgpu011', 'spartan-gpgpu013']
JOBS = ['33758541', '33758542']
ARMS = ['pd_2p6d_pp2_ep4ep2', 'pd_2p6d_pp2_3ep2']
COMMIT = '797f6488cf5fea2f88e38ccaeb1703443deec537'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def descriptor(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--port', type=int, default=30531)
    ap.add_argument('--dashboard-port', type=int, default=29831)
    opts = ap.parse_args()
    run = opts.run.resolve()
    assert not run.exists()
    original = read(SEED/'config.json')
    old_spec = read(SEED/'dashboard_spec.json')
    engine = Path(original['node_configs'][original['nodes'][0]]['engine_root'])
    assert subprocess.check_output(['git','-C',str(engine),'rev-parse','HEAD'],text=True).strip() == COMMIT
    assert not subprocess.check_output(['git','-C',str(engine),'status','--porcelain'],text=True).strip()
    subprocess.run([sys.executable,str(TRIGGER)],check=True,stdout=subprocess.DEVNULL)
    run.mkdir(parents=True)
    shutil.copytree(SEED/'code', run/'code')
    for source in SEED.glob('*.bin'):
        assert source.stat().st_size == 1
        # Policy files use ASCII '0'; capture counters use binary zero.
        # Preserve the already-frozen initial representation of each control.
        (run/source.name).write_bytes(source.read_bytes())
        os.chmod(run/source.name, 0o600)
    shutil.copyfile(SEED/'active_workload.txt',run/'active_workload.txt')
    shutil.copyfile(SEED/'agentx_graphs.json',run/'agentx_graphs.json')
    shutil.copyfile(SEED/'reference.input.json',run/'reference.input.json')
    save(run/'evidence/reference.json',read(SEED/'evidence/reference.json'))
    key = run/'admin.key'
    fd = os.open(key, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(fd,'w') as out:
        out.write(secrets.token_hex(32)+'\n')

    def replace(value):
        if isinstance(value,str):
            return value.replace(str(SEED),str(run)).replace(SEED.name,run.name).replace('spartan-gpgpu171','spartan-gpgpu013')
        if isinstance(value,list):
            return [replace(v) for v in value]
        if isinstance(value,dict):
            return {replace(k):replace(v) for k,v in value.items()}
        return value

    cfg = replace(original)
    cfg.update(id=run.name,nodes=NODES,arms=ARMS,repeats=1,expected_points=4,
        graphs_file=str(run/'agentx_graphs.json'),imported_episodes={},
        decode_extra_slots=32,endpoint_capacity_policy='32 D active +32 D receive per DP; 64 P active per PP2 instance',
        role_gpu_counts={a:dict(colocated=0,prefill=2,decode=6) for a in ARMS},
        profile_override=dict(pd_split='2:6',max_requests_per_node=128,
            mamba_equiv_per_full_gpu='P128 / D256; actual native states256 per rank'),
        policies={a:['sticky_hash'] for a in ARMS})
    case = original['cases'][0]
    cfg['cases'] = [dict(case,id='agentx_c8',concurrency=8,copies=1,available_root_sessions=8),
                    dict(case,id='agentx_c128',concurrency=128,copies=16,available_root_sessions=128)]
    cfg['endpoints'] = {a:[] for a in ARMS}
    cache = cfg['model_cache']
    cache['on_miss'] = 'build'
    cache['usage_logs'] = {n:[] for n in NODES}
    cache['node_specs'] = {}
    cache['cpu_memory_budget']['kv_banks_per_node'] = {n:[] for n in NODES}
    cache['hicache_reserve_bytes_per_node'] = {n:4*32*1024**3 for n in NODES}
    seed_node_spec = read(Path(original['model_cache']['node_specs'][original['nodes'][0]]['path']))
    for ni,node in enumerate(NODES):
        local = cfg['node_configs'][node]
        local.update(id=run.name,node=node,job_id=JOBS[ni],port=opts.port,dist_port=opts.port+2000,
            guard_scripts=str(GUARD),decode_extra_slots=32,expected_points=4,arm_order=ARMS)
        local['profile'].update(max_requests=128,mamba_equiv=128)
        local['pp_stream_microbatch'] = 16
        local['layouts'] = []
        for ai,arm in enumerate(ARMS):
            if ni == 0:
                shapes = [(1,2,[0,1],'prefill'),(2,1,[2,3],'decode')]
            elif arm == ARMS[0]:
                shapes = [(4,1,[0,1,2,3],'decode')]
            else:
                shapes = [(2,1,[0,1],'decode'),(2,1,[2,3],'decode')]
            instances,logs = [],[]
            for i,(ep,pp,gpus,role) in enumerate(shapes):
                port = opts.port+ai*100+i*10
                bootstrap = opts.port+5000+ai*100+i*10
                instances.append(dict(ep=ep,tp=ep,dp=ep,pp=pp,gpus=gpus,role=role,
                    bootstrap_port=bootstrap,profile_override=dict(mamba_equiv=128 if role=='prefill' else 256)))
                cfg['endpoints'][arm].append(dict(id=f'{node}.{arm}.r{i}',host=node,port=port,
                    ep=ep,pp=pp,role=role,bootstrap_port=bootstrap))
                path = run/'cache'/node/f'{arm}.r{i}.log'
                path.parent.mkdir(parents=True,exist_ok=True)
                path.touch(mode=0o600,exist_ok=False)
                logs.append(dict(id=f'{run.name}.{node}.{arm}.r{i}',path=str(path),
                    expected_ranks=[f'PP{s} TP0' for s in range(pp)] if pp>1 else [f'TP{s}' for s in range(ep)]))
            local['layouts'].append(dict(id=arm,replicas=len(instances),instances=instances,usage_logs=logs))
            cache['usage_logs'][node] += logs
        spec = dict(seed_node_spec,node=node,expected_job_id=JOBS[ni])
        cache_path = run/'cache'/f'{node}.json'
        save(cache_path,spec)
        cache['node_specs'][node] = descriptor(cache_path)
        local['model_cache'] = cache

    # Only the declared per-instance native memory budget changes engine args.
    mixed = run/'code/mixed_workload.py'
    old = "inst=instance(cfg,arm,replica);ep,pp=inst['ep'],inst['pp'];p=cfg['profile']"
    new = "inst=instance(cfg,arm,replica);ep,pp=inst['ep'],inst['pp'];p=dict(cfg['profile'],**inst.get('profile_override',{}))"
    text = mixed.read_text()
    assert text.count(old)==1
    mixed.write_text(text.replace(old,new))
    cfg['warmup_cases'] = [dict(input_tokens=8192,requests_per_attention_rank=w) for w in (1,4,16,32)]
    cfg['warmup_cases'] += [dict(input_tokens=n,requests_per_attention_rank=1) for n in (65536,208320)]
    cfg['protocol'] = (
        'One exploratory episode each at C8 and C128 for PP2 P + EP2/EP4 D, then PP2 P +3 EP2 D. '
        'Two currently RUNNING 4xH100 nodes011/013; eight GPUs,2P6D in every arm. '
        'E7 engine797f6488 and frozen V4 corrected AgentX branch timing; no imported performance results. '
        '32 active D requests/DP worker +32 receive slots with native GDN256/rank; PP2 P64 active with GDN256/stage. '
        'Declared GDN memory precedes automatic GPU KV sizing at .95; PP microbatch16. '
        'HiCache ON every GPU,32 decimal GB/rank, native P/D KV+GDN handoff; no within-episode flush. '
        'Same eight source families, C8 one copy, C128 sixteen cache-isolated copies. '
        'Sticky conversation affinity and counterbalanced clone placement over six D owners. '
        'D EP2 on P node receives local handoff; D on second node receives cross-node handoff; paths reported separately. '
        'Serial six-sequence exact output gate and all-owner cold/reuse warmup are outside timing. '
        'Each episode starts with a native cache flush; no saved CPU-KV checkpoint injected. '
        'Only global idle above10s compressed; source-hash synthetic prompts retain source lengths. '
        'Finite custom replay, not official AgentX score, steady capacity or dynamic-policy evaluation. '
        'One pilot repeat has no uncertainty estimate; existing E5/V3 and E7/V3 results are historical context only.')
    save(run/'config.json',cfg)
    sys.path.insert(0,str(run/'code'))
    cluster = load('two_p_six_d_cluster',run/'code/cluster_runner.py')
    budgets=[]
    commands=[]
    for node in NODES:
        base,local=cluster.load_base(run,cfg,node)
        assert base.h.read_key(local)
        for arm in ARMS:
            for i,inst in enumerate(base.layout(local,arm)['instances']):
                args=base.engine_args(local,arm,i)
                value=lambda flag:int(args[args.index(flag)+1])
                assert '--enable-hierarchical-cache' in args and '--max-total-tokens' not in args
                active=value('--max-running-requests')//inst['dp']
                states=value('--max-mamba-cache-size')//inst['dp']
                receive=value('--disaggregation-decode-extra-slots') if inst['role']=='decode' else 0
                required=(active+receive)*3
                assert states==256 and states-required==64,(inst,active,states,required)
                budgets.append(dict(node=node,arm=arm,replica=i,role=inst['role'],gpus=inst['gpus'],
                    requested_active_per_dp=active,receive_slots=receive,states_per_rank=states,
                    request_state_upper_bound=required,residual_for_cached_checkpoints=states-required))
                commands.append(dict(node=node,arm=arm,replica=i,args=args))
    save(run/'preflight.json',dict(passed=True,kind='CPU-only frozen command checks, actual GPU allocation must qualify',
        decode_slot_budget=budgets,commands=commands,engine_commit=COMMIT,
        replay=dict(source=str(SEED/'code/agentx_replay.py'),**descriptor(run/'code/agentx_replay.py')),
        cpu_kv_checkpoint_policy='Do not restore any saved checkpoint: each independent AgentX episode begins cold; live within-episode HiCache retained'))
    ds=replace(old_spec)
    ds.update(id=run.name,title='AgentX 2P6D: PP2 P, EP4+EP2 or3 EP2 D; GDN capacity pilot',
        port=opts.dashboard_port,model_cache=cache,runtime_dir=str(run/'dashboard_guard'),max_active_nodes=2,
        tasks=[dict(id='comparison',label='2P6D two D layouts x C8/C128 one pilot each',nodes=NODES,
                    root=str(run/'task'),controller_script=str(run/'code/cluster_runner.py'),expected=4)])
    # Retain source provenance from the audited preparation; runtime/key/log files
    # from that preparation are never imported. Hash the complete new bundle.
    files=[]
    for entry in old_spec['frozen_inputs']:
        path=Path(entry['path'])
        if path.is_relative_to(SEED):
            continue
        if 'guard_bundle' in path.parts:
            continue
        if path.exists():
            files.append(path)
    files += list((run/'code').glob('*.py'))+list(GUARD.glob('*.py'))
    files += [Path(__file__).resolve(),run/'config.json',run/'preflight.json',run/'agentx_graphs.json',
              key,run/'reference.input.json',run/'evidence/reference.json']
    files += [Path(x['path']) for x in cache['node_specs'].values()]
    sys.path.insert(0,str(GUARD))
    import cpu_memory_budget
    files += [Path(x['path']) for x in cpu_memory_budget.dependency_identity()['files']]
    ds['frozen_inputs']=list({str(p):descriptor(p) for p in files}.values())
    save(run/'dashboard_spec.json',ds)
    save(run/'launch_manifest.json',dict(id=run.name,engine='sglang',engine_root=str(engine),
        engine_commit=COMMIT,engine_tag=cfg['node_configs'][NODES[0]]['engine_tag'],
        model_revision=cfg['node_configs'][NODES[0]]['model_revision'],
        model_cache=cache,nodes=NODES,jobs=dict(zip(NODES,JOBS)),arms=ARMS,cases=cfg['cases'],
        repeats=1,role_gpu_counts=cfg['role_gpu_counts'],endpoints=cfg['endpoints'],
        protocol=cfg['protocol'],state_budgets=budgets,branch_timing_policy=cfg['branch_timing_policy'],
        frozen_inputs=ds['frozen_inputs']))
    save(run/'task/progress.json',dict(experiment_id=run.name,task_id='comparison',timestamp=time.time(),
        stage='queued',valid_points=0,expected=4))
    for p in (run/'code').glob('*.py'):
        ast.parse(p.read_text(),filename=str(p))
    print(json.dumps(dict(run=str(run),arms=ARMS,expected=4,budgets=budgets,frozen=len(ds['frozen_inputs']))))


if __name__=='__main__':
    main()
