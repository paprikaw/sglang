"""Freeze a full-model matched Decode run; no GPU/model launch here."""
import ast,argparse,json,socket,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'runtime'))
from common import *
from mtp_profile import PROFILE, validate as validate_mtp
from workload import cached_prefix_length
sys.path.insert(0,str(GUARD));import cpu_memory_budget,model_cache_adapter

def descriptor(p):return dict(path=str(p),sha256=sha(p))
def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True)
    p.add_argument('--mode',choices=('baseline','candidate'),required=True)
    p.add_argument('--engine-root',type=Path,default=ENGINE)
    p.add_argument('--engine-commit',required=True)
    p.add_argument('--prefix-tokens',type=int,choices=(4096,16384),default=4096)
    p.add_argument('--production-output-tokens',type=int,default=64)
    p.add_argument('--profile-output-tokens',type=int,default=64)
    a=p.parse_args()
    engine=a.engine_root.resolve();commit=a.engine_commit
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('Exact lowercase source commit required')
    if a.production_output_tokens<32 or a.profile_output_tokens<64:raise ValueError('Enough real Decode steps required')
    inputs=ROOT/'semantic_inputs'/f'p{a.prefix_tokens}'
    manifest=json.loads((inputs/'manifest.json').read_text())
    if (manifest['raw_prefix_tokens']!=a.prefix_tokens or not manifest['kv_requires_actual_model_prefill'] or len(manifest.get('source_commit',''))!=40):raise ValueError('Semantic input provenance mismatch')
    run=ROOT/'runs'/a.name;run.mkdir(mode=0o700,parents=True,exist_ok=False);(run/'task').mkdir(mode=0o700)
    if subprocess.check_output(['git','-C',str(engine),'rev-parse','HEAD'],text=True).strip()!=commit:raise RuntimeError('Engine HEAD changed')
    subprocess.run(['git','-C',str(engine),'diff','--exit-code','HEAD','--'],check=True,capture_output=True)
    cfg=dict(schema='glm47-pp-microbatch-witness-v1',id='glm47-a100-pp-microbatch-witness-20261023-'+a.mode+'-'+a.name,mode=a.mode,run_dir=str(run),
        python=PYTHON,engine_commit=commit,engine_root=str(engine),semantic_source_commit=manifest['source_commit'],semantic_inputs=str(inputs),guard_scripts=str(GUARD),
        nodes=['spartan-gpgpu065','spartan-gpgpu167'],jobs={'spartan-gpgpu065':'33463294','spartan-gpgpu167':'32715983'},
        layouts=LAYOUTS,prefix_lengths=[a.prefix_tokens],api_port=port(),dist_port=port(),timeout_s=3600,
        context_kind='Real source-code/documentation; actual model Prefill KV',synthetic_kv=False,
        production_output_tokens=a.production_output_tokens,profile_output_tokens=a.profile_output_tokens,
        arm_order=['pp2ep4'],profile_after_production_only=False,profile_enabled=False,correctness_diagnostic=True,performance_comparison_claim_valid=False,atomic_batch_admission=True,production_repeats=2,untimed_witness_cohorts=2,deterministic_inference=True,prefill_truncation_alignment=1024)
    save(run/'config.json',cfg)
    preload=ROOT.parent/'glm47_a100_cached_decode_ep8_pp2ep4_20261022/runs/preload_r2';specs={};usage={}
    for node in cfg['nodes']:
        specs[node]=descriptor(preload/(node+'.json'));usage[node]=[]
        for layout in LAYOUTS:
            (run/layout/node).mkdir(parents=True,mode=0o700)
            usage[node].append(dict(id=layout+'-target-'+node,path=str(run/layout/node/'target_weight.log'),expected_ranks=expected_ranks(layout,cfg['nodes'].index(node))))
            if layout=='ep8' or cfg['nodes'].index(node)==1:
                usage[node].append(dict(id=layout+'-draft-'+node,path=str(run/layout/node/'draft_weight.log'),expected_ranks=expected_ranks(layout,cfg['nodes'].index(node))))
    for layout in LAYOUTS:
        for d in ['capacity','receipts']:(run/layout/d).mkdir(mode=0o700)
    paths=[*GUARD.glob('*.py'),*(ROOT/'runtime').glob('*.py'),ROOT/'prepare_measure.py',run/'config.json',MODEL/'config.json',MODEL/'model.safetensors.index.json',NATIVE,SHIM]
    paths.extend(inputs.iterdir())
    paths.extend((ROOT/'build_semantic_inputs.py',MODEL/'tokenizer_config.json',MODEL/'tokenizer.json',MODEL/'chat_template.jinja'))
    # Freeze the Humming SM80 seed used by each isolated worker cache.
    paths.extend(f for f in SEED.rglob('*') if f.is_file() and f.name!='lock' and not f.name.endswith('.lock'))
    paths.extend(f for f in DEEP.rglob('*.py') if '__pycache__' not in f.parts)
    frozen={str(x):descriptor(x) for x in paths}
    for d in [*specs.values(),*cpu_memory_budget.dependency_identity()['files']]:frozen[d['path']]=d
    contract=dict(policy='required',provider='sglang_canonical_full_weight_v1',support_profile='glm4moe-canonical-v1',on_miss='fail',
        node_specs=specs,usage_logs=usage,hicache_reserve_bytes_per_node={n:144*1024**3 for n in cfg['nodes']},
        cpu_memory_budget=dict(schema_version=1,kv_banks_per_node={n:[] for n in cfg['nodes']}))
    spec=dict(schema_version=1,id=cfg['id'],title='GLM A100 PP '+a.mode+': first-divergence witness correctness diagnostic',engine='sglang',
        protocol='task-events-v2',template='parallel-tasks-v5',features=['parallel-tasks','task-progress','guard','cpu-model-cache'],
        port=port(),max_active_nodes=2,gpu_lease_ttl_s=90,cleanup_wait_s=90,runtime_dir=str(run/'dashboard_guard'),
        model_cache=contract,frozen_inputs=list(frozen.values()),sources=[dict(worktree=str(engine),commit=commit),dict(worktree=str(WORKSPACE/'.engine-worktrees/glm47-ep16-canonical-profile-20261004'),commit='870269702285d94bdd926a50cc29ad1a20bcf60c')],
        workload=dict(kind='full_model_cached_decode',model_loaded=True,full_model=True,batch_size=8,prefix_tokens=a.prefix_tokens,query_tokens=1,output_tokens=a.production_output_tokens,separate_profile_output_tokens=0,profile_enabled=False,correctness_diagnostic=True,performance_comparison_claim_valid=False,witness_limit_per_cohort_slot=16,witness_gpu_limit_bytes_per_tail_rank=768*1024**2,microbatch_mode=a.mode,microbatch_toggle=1 if a.mode=='candidate' else 0,atomic_batch_admission=True,production_repeats=2,untimed_witness_cohorts=2,deterministic_inference=True,prefill_truncation_alignment=1024,scheduling_policy='fcfs',speculative_profile=PROFILE,semantic_kv=True,synthetic_kv=False,preparation='actual_model_prefill_untimed',cuda_profiler_off_during_production=True),
        tasks=[dict(id='decode',label=f'{a.mode}: semantic B8 P{a.prefix_tokens} Q1 ->{a.production_output_tokens}; correctness witness, no CUDA profile',nodes=cfg['nodes'],root=str(run/'task'),expected=1,controller_script=str(ROOT/'runtime/controller.py'))])
    save(run/'mtp_profile_qualification.json',validate_mtp(spec,cfg))
    save(run/'dashboard_spec.json',spec);model_cache_adapter.validate_contract(spec,launch=True)
    for f in (ROOT/'runtime').glob('*.py'):ast.parse(f.read_text())
    print(json.dumps(dict(spec=str(run/'dashboard_spec.json'),command=[PYTHON,str(ROOT/'runtime/controller.py'),'--run',str(run)],frozen_files=len(frozen),gpu_launched=False)))
if __name__=='__main__':main()
