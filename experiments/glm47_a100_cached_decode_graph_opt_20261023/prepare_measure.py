"""Freeze a full-model matched Decode run; no GPU/model launch here."""
import ast,argparse,json,socket,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'runtime'))
from common import *
sys.path.insert(0,str(GUARD));import cpu_memory_budget,model_cache_adapter

def descriptor(p):return dict(path=str(p),sha256=sha(p))
def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='graph_r1');a=p.parse_args()
    run=ROOT/'runs'/a.name;run.mkdir(mode=0o700,parents=True,exist_ok=False);(run/'task').mkdir(mode=0o700)
    if subprocess.check_output(['git','-C',str(ENGINE),'rev-parse','HEAD'],text=True).strip()!=COMMIT:raise RuntimeError('Engine HEAD changed')
    subprocess.run(['git','-C',str(ENGINE),'diff','--exit-code','HEAD','--'],check=True,capture_output=True)
    cfg=dict(schema='glm47-cached-decode-v1',id='glm47-a100-cached-decode-graph-20261023-'+a.name,run_dir=str(run),
        python=PYTHON,engine_commit=COMMIT,engine_root=str(ENGINE),guard_scripts=str(GUARD),
        nodes=['spartan-gpgpu065','spartan-gpgpu167'],jobs={'spartan-gpgpu065':'33463294','spartan-gpgpu167':'32715983'},
        layouts=LAYOUTS,prefix_lengths=[81920],api_port=port(),dist_port=port(),timeout_s=2400)
    save(run/'config.json',cfg)
    preload=ROOT.parent/'glm47_a100_cached_decode_ep8_pp2ep4_20261022/runs/preload_r2';specs={};usage={}
    for node in cfg['nodes']:
        specs[node]=descriptor(preload/(node+'.json'));usage[node]=[]
        for layout in LAYOUTS:
            (run/layout/node).mkdir(parents=True,mode=0o700)
            usage[node].append(dict(id=layout+'-'+node,path=str(run/layout/node/'server.log'),expected_ranks=expected_ranks(layout,cfg['nodes'].index(node))))
    for layout in LAYOUTS:
        for d in ['capacity','receipts']:(run/layout/d).mkdir(mode=0o700)
    paths=[*GUARD.glob('*.py'),*(ROOT/'runtime').glob('*.py'),ROOT/'prepare_measure.py',run/'config.json',MODEL/'config.json',MODEL/'model.safetensors.index.json',NATIVE,SHIM]
    # Freeze the Humming SM80 seed used by each isolated worker cache.
    paths.extend(f for f in SEED.rglob('*') if f.is_file() and f.name!='lock' and not f.name.endswith('.lock'))
    paths.extend(f for f in DEEP.rglob('*.py') if '__pycache__' not in f.parts)
    frozen={str(x):descriptor(x) for x in paths}
    for d in [*specs.values(),*cpu_memory_budget.dependency_identity()['files']]:frozen[d['path']]=d
    contract=dict(policy='required',provider='sglang_canonical_full_weight_v1',support_profile='glm4moe-canonical-v1',on_miss='fail',
        node_specs=specs,usage_logs=usage,hicache_reserve_bytes_per_node={n:144*1024**3 for n in cfg['nodes']},
        cpu_memory_budget=dict(schema_version=1,kv_banks_per_node={n:[] for n in cfg['nodes']}))
    spec=dict(schema_version=1,id=cfg['id'],title='GLM A100 matched Full Decode Graph: EP8 vs PP2xEP4',engine='sglang',
        protocol='task-events-v2',template='parallel-tasks-v5',features=['parallel-tasks','task-progress','guard','cpu-model-cache'],
        port=port(),max_active_nodes=2,gpu_lease_ttl_s=90,cleanup_wait_s=90,runtime_dir=str(run/'dashboard_guard'),
        model_cache=contract,frozen_inputs=list(frozen.values()),sources=[dict(worktree=str(ENGINE),commit=COMMIT)],
        workload=dict(kind='full_model_cached_decode',model_loaded=True,full_model=True,batch_size=8,prefix_tokens=81920,query_tokens=1,output_tokens=128),
        tasks=[dict(id='decode',label='Matched B8 P81920 Q1 ->128; CPU ready GPU empty',nodes=cfg['nodes'],root=str(run/'task'),expected=2,controller_script=str(ROOT/'runtime/controller.py'))])
    save(run/'dashboard_spec.json',spec);model_cache_adapter.validate_contract(spec,launch=True)
    for f in (ROOT/'runtime').glob('*.py'):ast.parse(f.read_text())
    print(json.dumps(dict(spec=str(run/'dashboard_spec.json'),command=[PYTHON,str(ROOT/'runtime/controller.py'),'--run',str(run)],frozen_files=len(frozen),gpu_launched=False)))
if __name__=='__main__':main()
