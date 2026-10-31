"""Freeze two MTP layouts with the same qualified Verify fix; no GPU launch."""
import ast,argparse,json,socket,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'runtime'))
from common import *
from mtp_profile import PROFILE,validate as validate_mtp
sys.path.insert(0,str(GUARD));import cpu_memory_budget,model_cache_adapter

def descriptor(path):return dict(path=str(path),sha256=sha(path))
def port():
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));return sock.getsockname()[1]

def build_contract(cfg,preload):
    specs={};usage={}
    for index,node in enumerate(cfg['nodes']):
        specs[node]=descriptor(preload/(node+'.json'));usage[node]=[]
        for arm,mode in execution_arms(cfg).items():
            base=Path(cfg['run_dir'])/arm/node;base.mkdir(parents=True,mode=0o700)
            usage[node].append(dict(id=arm+'-target-'+node,path=str(base/'target_weight.log'),expected_ranks=expected_ranks(mode['layout'],index)))
            if mode['mtp'] and (mode['layout']=='ep8' or index==1):
                usage[node].append(dict(id=arm+'-draft-'+node,path=str(base/'draft_weight.log'),expected_ranks=expected_ranks(mode['layout'],index)))
    return dict(policy='required',provider='sglang_canonical_full_weight_v1',support_profile='glm4moe-canonical-v1',on_miss='fail',
        node_specs=specs,usage_logs=usage,hicache_reserve_bytes_per_node={node:160*1024**3 for node in cfg['nodes']},
        cpu_memory_budget=dict(schema_version=1,kv_banks_per_node={node:[] for node in cfg['nodes']}))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True)
    parser.add_argument('--engine-root',type=Path,default=ENGINE);parser.add_argument('--engine-commit',default=PINNED_COMMIT)
    parser.add_argument('--reuse-semantic-bank',type=Path,help='Existing absolute READY bank; skip producer and run exactly two consumers')
    args=parser.parse_args();engine=args.engine_root.resolve();commit=args.engine_commit
    tests=subprocess.run([PYTHON,'-m','pytest','-q','-p','no:cacheprovider',str(ROOT/'test_local_feedback.py'),str(ROOT/'test_balance.py')],
        capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    if tests.returncode:raise RuntimeError('CPU protocol regressions failed: '+tests.stdout+tests.stderr)
    if args.reuse_semantic_bank is not None and not args.reuse_semantic_bank.is_absolute():raise ValueError('--reuse-semantic-bank requires an absolute path')
    if commit!=PINNED_COMMIT:raise ValueError('Four-way source must be exact qualified9e48 engine')
    if subprocess.check_output(['git','-C',str(engine),'rev-parse','HEAD'],text=True).strip()!=commit:raise RuntimeError('Engine HEAD changed')
    subprocess.run(['git','-C',str(engine),'diff','--exit-code','HEAD','--'],check=True,capture_output=True)
    inputs=ROOT.parent/'glm47_semantic_bank_fourway_20261023_v5/semantic_inputs/p4096';manifest=json.loads((inputs/'manifest.json').read_text())
    if manifest['raw_prefix_tokens']!=4096 or not manifest['kv_requires_actual_model_prefill']:raise ValueError('Semantic corpus provenance mismatch')
    run=ROOT/'runs'/args.name;run.mkdir(mode=0o700,parents=True,exist_ok=False);(run/'task').mkdir(mode=0o700)
    cfg=dict(schema=SCHEMA,id='glm47-pp-stage-balance-20261023-'+args.name,run_dir=str(run),python=PYTHON,
        engine_commit=commit,engine_root=str(engine),semantic_source_commit=manifest['source_commit'],semantic_inputs=str(inputs),guard_scripts=str(GUARD),
        nodes=['spartan-gpgpu065','spartan-gpgpu167'],jobs={'spartan-gpgpu065':'33463294','spartan-gpgpu167':'32715983'},
        layouts=LAYOUTS,reuse_semantic_bank=args.reuse_semantic_bank is not None,prefix_lengths=[4096],api_port=port(),dist_port=port(),timeout_s=3600,
        model_revision='safetensors-index-sha256:'+sha(MODEL/'model.safetensors.index.json'),
        semantic_bank_dir=str(args.reuse_semantic_bank.resolve() if args.reuse_semantic_bank else run/'semantic_bank'),bank_page_cache_max_bytes_per_node=12*1024**3,
        bank_temporary_copy_buffer_max_bytes_per_rank=16*1024**2,bank_disk_payload_max_bytes=12*1024**3,
        compiled_kernel_cache_dir='/tmp/15540-glm47-verify-attention-ab-20261023-original_vs_causal_decode_r1/triton',
        context_kind='Actual semantic model Prefill; immutable portable target+draft KV pages',synthetic_kv=False,
        production_output_tokens=64,profile_output_tokens=64,profile_after_production_only=True,profile_enabled=True,
        atomic_batch_admission=True,production_repeats=2,untimed_witness_cohorts=0,
        deterministic_inference=True,prefill_truncation_alignment=1024,
        optimization='Keep qualified local-tail feedback; move4 target layers to stage0 to account for tail MTP work',
        unfenced_flow_diagnostic=True,
        cold_cache_protocol='Reset and import immutable bank BEFORE each warm/timed/repeat/profile; native CPU4032/GPU0 at timer start',
        admission_control_protocol='EP8 CPU tokenizer perRID DP0:0..3/DP1:4..7 and local atomicB4 gate; PP stock atomicB8 split into two nativeB4 slots; no new world collective',
        communication_policy='Native DeepEP fixed pair NVLink; all outside-pair traffic RDMA; no PCIe CE path')
    cfg['arms']=execution_arms(cfg);cfg['arm_order']=list(cfg['arms'])
    cfg['bank_model_identity']=bank_identity(cfg)
    if cfg['reuse_semantic_bank']:cfg['source_semantic_bank_ready']=reuse_bank_descriptor(cfg['semantic_bank_dir'],cfg)
    reference=ROOT.parent/'glm47_pp_local_feedback_20261023/runs/paired_r1'
    source_config=json.loads((reference/'config.json').read_text())
    source_arm=reference/'pp2ep4_local_tail'
    owned=json.loads((source_arm/'arm_result.json').read_text())
    if not all(owned.get(k) for k in ('passed','performance_valid','profile_valid','cleanup_verified')):
        raise RuntimeError('Stage balance requires a completed qualified local-tail baseline')
    if (source_config['nodes']!=cfg['nodes'] or source_config['jobs']!=cfg['jobs'] or source_config['engine_commit']!=cfg['engine_commit']
        or source_config['source_semantic_bank_ready']!=cfg['source_semantic_bank_ready']):
        raise RuntimeError('Baseline node/job/engine/semantic bank differs')
    cfg['source_pp_reference']=dict(production=descriptor(source_arm/'p4096_decode_results.json'),
        repeat=descriptor(source_arm/'p4096_decode_repeat_results.json'),source_config=descriptor(reference/'config.json'),
        arm=descriptor(source_arm/'arm_result.json'),cleanup=descriptor(reference/'task/cleanup.json'))
    save(run/'config.json',cfg)
    save(run/'cpu_protocol_tests.json',dict(passed=True,gpu_execution=False,stdout=tests.stdout,stderr=tests.stderr))
    for arm in cfg['arm_order']:
        for directory in ('capacity','receipts'):(run/arm/directory).mkdir(parents=True,mode=0o700)
    preload=ROOT.parent/'glm47_a100_cached_decode_ep8_pp2ep4_20261022/runs/preload_r2'
    contract=build_contract(cfg,preload)
    paths=[*GUARD.glob('*.py'),*(ROOT/'runtime').glob('*.py'),ROOT/'analysis/stage_activity.py',ROOT/'analysis/event_spans.py',ROOT/'prepare_measure.py',ROOT/'test_local_feedback.py',ROOT/'test_balance.py',run/'cpu_protocol_tests.json',run/'config.json',
        MODEL/'config.json',MODEL/'model.safetensors.index.json',NATIVE,SHIM,*inputs.iterdir(),
        MODEL/'tokenizer_config.json',MODEL/'tokenizer.json',MODEL/'chat_template.jinja']
    paths.extend(path for path in SEED.rglob('*') if path.is_file() and path.name!='lock' and not path.name.endswith('.lock'))
    paths.extend(path for path in DEEP.rglob('*.py') if '__pycache__' not in path.parts)
    if cfg['reuse_semantic_bank']:
        bank=Path(cfg['semantic_bank_dir']);published=json.loads((bank/'READY.json').read_text())
        paths.extend([bank/'READY.json',*(bank/f'rank{rank}.json' for rank in range(8))])
        paths.extend(Path(row['path']) for row in published['publication_gate']['evidence'])
    frozen={str(path):descriptor(path) for path in paths}
    for row in cfg['source_pp_reference'].values():frozen[row['path']]=row
    for row in [*contract['node_specs'].values(),*cpu_memory_budget.dependency_identity()['files']]:frozen[row['path']]=row
    spec=dict(schema_version=1,id=cfg['id'],title='GLM PP2 MTP: target50/42 balances tail draft work',engine='sglang',
        protocol='task-events-v2',template='parallel-tasks-v5',features=['parallel-tasks','task-progress','guard','cpu-model-cache'],
        port=port(),max_active_nodes=2,gpu_lease_ttl_s=90,cleanup_wait_s=90,runtime_dir=str(run/'dashboard_guard'),
        model_cache=contract,frozen_inputs=list(frozen.values()),sources=[dict(worktree=str(engine),commit=commit),
            dict(worktree=str(WORKSPACE/'.engine-worktrees/glm47-ep16-canonical-profile-20261004'),commit='870269702285d94bdd926a50cc29ad1a20bcf60c')],
        workload=dict(kind='full_model_cached_decode',model_loaded=True,full_model=True,batch_size=8,prefix_tokens=4096,
            cached_tokens=4032,query_tokens=1,output_tokens=64,separate_profile_output_tokens=64,profile_enabled=True,
            profile_after_production_only=True,witness_enabled=False,atomic_batch_admission=True,production_repeats=2,
            deterministic_inference=True,prefill_truncation_alignment=1024,scheduling_policy='fcfs',speculative_profile=PROFILE,
            semantic_kv=True,synthetic_kv=False,preparation='reuse_existing_immutable_semantic_bank' if cfg['reuse_semantic_bank'] else 'one_actual_semantic_prefill_producer_then_immutable_bank_import',
            reuse_semantic_bank=cfg['reuse_semantic_bank'],source_semantic_bank_ready=cfg.get('source_semantic_bank_ready'),
            source_bank_import_timed=False,hicache_h2d_timed=True,cuda_profiler_off_during_production=True),
        tasks=[dict(id='decode',label='Fixed Verify EP8 vs two-slot PP2xEP4; semanticB8 P4096 Q1->64',
            nodes=cfg['nodes'],root=str(run/'task'),expected=1,controller_script=str(ROOT/'runtime/controller.py'))])
    save(run/'mtp_profile_qualification.json',validate_mtp(spec,cfg));save(run/'dashboard_spec.json',spec)
    model_cache_adapter.validate_contract(spec,launch=True)
    for path in (ROOT/'runtime').glob('*.py'):ast.parse(path.read_text())
    print(json.dumps(dict(spec=str(run/'dashboard_spec.json'),command=[PYTHON,str(ROOT/'runtime/controller.py'),'--run',str(run)],
        frozen_files=len(frozen),gpu_launched=False)))
if __name__=='__main__':main()
