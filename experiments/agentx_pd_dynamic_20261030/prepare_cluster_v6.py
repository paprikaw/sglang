"""Extend frozen cluster protocol with P:D splits and explicit load capacity.

Each custom arm retains the known native family; instances are constant and
role counts are recorded. This prepares fixed baselines, not dynamic results.
"""
import argparse
import importlib.util
import math
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
extra = argparse.ArgumentParser(add_help=False)
extra.add_argument('--pd-split', choices=['4:4','2:6','6:2'], default='4:4')
extra.add_argument('--request-cap', type=int, default=32)
extra.add_argument('--mamba-equiv', type=int, default=40)
opts, rest = extra.parse_known_args(sys.argv[1:])
assert opts.request_cap > 0 and opts.request_cap % 4 == 0
assert opts.mamba_equiv > 0
sys.argv = [sys.argv[0]] + rest
run = Path(sys.argv[sys.argv.index('--run')+1]).resolve()
runpy.run_path(str(ROOT/'prepare_cluster_v5.py'), run_name='__main__')
spec = importlib.util.spec_from_file_location('prep_final', ROOT/'prepare_cluster.py')
prep = importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
cfg = prep.read(run/'config.json')
for case in cfg['cases']:
    case['copies'] = max(1, math.ceil(case['concurrency']/len(case['graph_ids'])))
    case['available_root_sessions'] = len(case['graph_ids'])*case['copies']
for node_cfg in cfg['node_configs'].values():
    node_cfg['profile'].update(max_requests=opts.request_cap,mamba_equiv=opts.mamba_equiv)
if opts.pd_split != '4:4':
    assert all(arm in ('col_ep2','col_pp2','pd_ep2','pd_pp2') for arm in cfg['arms']), 'Split baseline requires homogeneous two-GPU tiles'
    for arm in cfg['arms']:
        if not arm.startswith('pd_'):
            continue
        p_count = int(opts.pd_split.split(':')[0])//2
        instances = [(n,i,s) for n in cfg['nodes'] for i,s in enumerate(next(r for r in cfg['node_configs'][n]['layouts'] if r['id']==arm)['instances'])]
        assert len(instances)==4 and all(len(s['gpus'])==2 for _,_,s in instances)
        for index,(node,i,instance) in enumerate(instances):
            instance['role']='prefill' if index<p_count else 'decode'
            endpoint=next(e for e in cfg['endpoints'][arm] if e['id']==f'{node}.{arm}.r{i}')
            endpoint['role']=instance['role']
cfg['profile_override']=dict(pd_split=opts.pd_split,max_requests_per_node=opts.request_cap,mamba_equiv_per_full_gpu=opts.mamba_equiv)
cfg['role_gpu_counts']={}
for arm in cfg['arms']:
    counts={role:0 for role in ('colocated','prefill','decode')}
    for node_cfg in cfg['node_configs'].values():
        for s in next(r for r in node_cfg['layouts'] if r['id']==arm)['instances']:
            counts[s['role']]+=len(s['gpus'])
    assert sum(counts.values())==8
    cfg['role_gpu_counts'][arm]=counts
cfg['protocol'] += f' Fixed role split {opts.pd_split} where applicable; native cache auto-sizing with node request cap {opts.request_cap} and full-model-equivalent GDN slots {opts.mamba_equiv}; cloned trace families are cache-isolated and numerous enough for the named root concurrency.'
prep.save(run/'config.json',cfg)
sys.path.insert(0,str(run/'code'))
spec=importlib.util.spec_from_file_location('final_preflight',run/'code/cluster_runner.py')
cluster=importlib.util.module_from_spec(spec);spec.loader.exec_module(cluster)
preflight=prep.read(run/'preflight.json');preflight['instances']=[]
for node in cfg['nodes']:
    base,c=cluster.load_base(run,cfg,node)
    for arm in cfg['arms']:
        for i,s in enumerate(base.layout(c,arm)['instances']):
            args=base.engine_args(c,arm,i)
            assert '--enable-hierarchical-cache' in args and '--max-total-tokens' not in args
            if s['role'] != 'colocated':
                assert args[args.index('--disaggregation-mode')+1]==s['role']
            preflight['instances'].append(dict(node=node,arm=arm,replica=i,role=s['role'],ep=s['ep'],pp=s['pp']))
preflight['role_gpu_counts']=cfg['role_gpu_counts'];preflight['profile_override']=cfg['profile_override']
prep.save(run/'preflight.json',preflight)
ds=prep.read(run/'dashboard_spec.json')
paths=[Path(d['path']) for d in ds['frozen_inputs']]+[Path(__file__).resolve()]
ds['frozen_inputs']=list({str(p):prep.desc(p) for p in paths}.values())
prep.save(run/'dashboard_spec.json',ds)
manifest=prep.read(run/'launch_manifest.json')
manifest.update(protocol=cfg['protocol'],cases=cfg['cases'],endpoints=cfg['endpoints'],
                profile_override=cfg['profile_override'],role_gpu_counts=cfg['role_gpu_counts'],frozen_inputs=ds['frozen_inputs'])
prep.save(run/'launch_manifest.json',manifest)
print('Explicit capacity and P:D split preflight passed:',cfg['profile_override'],cfg['role_gpu_counts'])
