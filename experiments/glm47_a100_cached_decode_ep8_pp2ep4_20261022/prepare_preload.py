"""Prepare required full canonical GLM bank contract, CPU only, no serving."""
import json,sys,socket
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime'))
from common import *
sys.path.insert(0,str(GUARD))
import model_cache_adapter,cpu_memory_budget

def descriptor(p):return dict(path=str(p),sha256=sha(p))
def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
def main():
    run=ROOT/'runs/preload_r2';run.mkdir(mode=0o700,parents=True,exist_ok=False)
    nodes=['spartan-gpgpu065','spartan-gpgpu167'];jobs=['33463294','32715983']
    specs={};usage={}
    for node,job in zip(nodes,jobs):
        p=run/(node+'.json')
        value=dict(node=node,expected_job_id=job,engine_root=str(ENGINE),python=PYTHON,source_commit=COMMIT,
            socket_path='/tmp/sglang-canonical-weights-15540/cache.sock',cache_scheme='canonical_full_weight',
            model_path=str(MODEL),max_bytes=192*1024**3,headroom_bytes=32*1024**3)
        save(p,value);specs[node]=descriptor(p)
        usage[node]=[dict(id='preload-'+node,path=str(run/(node+'.usage.log')),expected_ranks=['TP0'])]
    old=json.loads(Path('/home/bxb1/experiment-runtime/glm47_a100_hicache_cold_b8_20261015_r2/dashboard_spec.json').read_text())
    frozen={d['path']:d for d in old['frozen_inputs'] if Path(d['path']).is_relative_to(ENGINE)}
    for p in [*GUARD.glob('*.py'),ROOT/'prepare_preload.py',ROOT/'runtime/common.py',MODEL/'config.json',MODEL/'model.safetensors.index.json']:
        frozen[str(p)]=descriptor(p)
    for d in [*specs.values(),*cpu_memory_budget.dependency_identity()['files']]:frozen[d['path']]=d
    contract=dict(policy='required',provider='sglang_canonical_full_weight_v1',support_profile='glm4moe-canonical-v1',on_miss='build',
        node_specs=specs,usage_logs=usage,hicache_reserve_bytes_per_node={n:0 for n in nodes},
        cpu_memory_budget=dict(schema_version=1,kv_banks_per_node={n:[] for n in nodes}))
    spec=dict(schema_version=1,id='glm47-a100-full-bank-preload-20261022-r2',title='GLM complete CPU bank preload',engine='sglang',
        protocol='task-events-v2',template='parallel-tasks-v5',features=['parallel-tasks','task-progress','guard','cpu-model-cache'],
        port=port(),max_active_nodes=2,gpu_lease_ttl_s=90,cleanup_wait_s=90,runtime_dir=str(run/'dashboard_guard'),
        model_cache=contract,frozen_inputs=list(frozen.values()),sources=[dict(worktree=str(ENGINE),commit=COMMIT)],
        tasks=[dict(id='preload',label='Full GLM bank; no model server',nodes=nodes,root=str(run/'task'),expected=1)])
    save(run/'dashboard_spec.json',spec);model_cache_adapter.validate_contract(spec,launch=True)
    print(json.dumps(dict(spec=str(run/'dashboard_spec.json'),frozen_files=len(frozen),gpu_launched=False)))
if __name__=='__main__':main()
