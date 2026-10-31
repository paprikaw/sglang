"""Freeze paired colocated/PD qualification using the existing admitted banks."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parent
WS=ROOT.parents[1]
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def desc(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--node',required=True);ap.add_argument('--job',required=True)
    ap.add_argument('--family',choices=['ep2','pp2'],required=True);ap.add_argument('--port',type=int,required=True)
    a=ap.parse_args();run=a.run.resolve()
    spec=importlib.util.spec_from_file_location('old_prepare',WS/'experiments/h100_pd_mix_20261030/prepare.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    sys.argv=[str(spec.origin),'--run',str(run),'--node',a.node,'--job',a.job,'--port',str(a.port),
              '--layouts','two_ep2' if a.family=='ep2' else 'two_pp2','--cases','cold_b32_q512']
    old.main();cfg=read(run/'config.json')
    col=cfg['layouts'][0];col['id']='colocated';col['mode']='colocated'
    for i,u in enumerate(col['usage_logs']):u.update(id=f'{run.name}.colocated.r{i}',path=str(run/'cache'/f'colocated.r{i}.log'))
    pd=json.loads(json.dumps(col));pd['id']='pd';pd['mode']='pd'
    for i,inst in enumerate(pd['instances']):inst['role']='prefill' if i==0 else 'decode'
    for i,u in enumerate(pd['usage_logs']):u.update(id=f'{run.name}.pd.r{i}',path=str(run/'cache'/f'pd.r{i}.log'))
    cfg.update(layouts=[col,pd],family=a.family,port=28321,dist_port=31321,
        expected_points=4,startup_timeout_s=1200,screening_windows=str(run/'screening_windows.json'),
        hicache_gb_per_rank=32,hicache_gib_per_rank=32,
        profile=dict(max_context=66048,max_requests=32,mamba_equiv=40,kv_token_cap_per_full_gpu=393216),
        pp_stream_microbatch=8,request_timeout_s=180,
        protocol='Four H100 per deployment; two colocated two-GPU replicas versus one two-GPU Prefill plus one two-GPU Decode. '
        'Native Mooncake KV and GDN transfer; HiCache enabled on every role with 32 decimal GB per GPU host pool, no speculation. Exact serial output qualification first. '
        'AgentX-derived eight contiguous five-turn windows; exact source lengths/hash-prefix sharing and inter-turn waits; '
        'C1 and C8 session lanes, three repetitions, warm first turn excluded from per-request timing summary but included in full episode wall time. '
        'GPU KV uses native automatic capacity at static memory fraction 0.95, no artificial max-total-tokens cap. '
        'Session affinity retains prefixes across all five turns; flush only at independent repeat boundaries, never between turns. '
        'Decode radix cache enabled for PD; include real GPU/host cache hit records. '
        'PP native scheduling policy0; no claim of best optimized PP, official AgentX score, cross-node transfer, or dynamic resource allocation.')
    cfg['launch_env'].update(PP_GAP_SEPARATE_CHANNELS='0',MC_GID_INDEX='5',
        MC_TCP_ENABLE_CONNECTION_POOL='true',SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT='120',
        SGLANG_DISAGGREGATION_WAITING_TIMEOUT='120')
    cfg['model_cache']['usage_logs']={a.node:[u for row in cfg['layouts'] for u in row['usage_logs']]}
    # Reserve a conservative 32 GiB per rank for the 32 decimal GB host pool,
    # including alignment and small auxiliary state; exactly four ranks/node.
    cfg['model_cache']['hicache_reserve_bytes_per_node']={a.node:4*32*1024**3}
    shutil.copyfile(ROOT/'screening_windows.json',run/'screening_windows.json')
    shutil.copyfile(ROOT/'pd_workload.py',run/'code/pd_workload.py')
    runner=run/'code/runner.py';s=runner.read_text()
    s=s.replace('mixed_workload.install(sys.modules[__name__])','mixed_workload.install(sys.modules[__name__])\nimport pd_workload\npd_workload.install(sys.modules[__name__])')
    runner.write_text(s)
    save(run/'config.json',cfg)
    spec=read(run/'dashboard_spec.json');spec.update(title='AgentX-derived Qwen35B real PD qualification '+a.family,model_cache=cfg['model_cache'])
    spec['tasks'][0].update(expected=cfg['expected_points'],label='Qualified colocated/PD episode cases')
    files=[Path(x['path']) for x in spec['frozen_inputs']]+[run/'code/pd_workload.py',run/'screening_windows.json',ROOT/'prepare.py',ROOT/'PROTOCOL.md']
    spec['frozen_inputs']=list({str(p):desc(p) for p in files}.values());save(run/'dashboard_spec.json',spec)
    manifest=read(run/'launch_manifest.json');manifest.update(protocol=cfg['protocol'],layouts=cfg['layouts'],
        frozen_inputs=spec['frozen_inputs'],dataset=read(WS/'datasets/agentx_20261030/cc-traces-weka-062126-256k/download_manifest.json'),
        claim_boundary='Feasibility and narrow static PD screen only. Dynamic EP/PP resource-ratio benefit remains unmeasured.')
    save(run/'launch_manifest.json',manifest)
    print(json.dumps(dict(run=str(run),node=a.node,family=a.family)))

if __name__=='__main__':main()
