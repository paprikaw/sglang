"""CPU-only freeze of two matched, compact A100 full-serving arms."""
import argparse
import ast
import json
import subprocess
import time
import shutil
from common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,type=Path)
    ap.add_argument('--include-rdma',action='store_true',help='Append same-native RDMA arms after the two CE arms')
    ap.add_argument('--api-port-base',type=int,default=32781)
    ap.add_argument('--dist-port-base',type=int,default=30781)
    ap.add_argument('--dashboard-port',type=int,default=18881)
    args=ap.parse_args();run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    assert not (run/'dashboard_spec.json').exists(),'Never rewrite a frozen run'
    arms=ARMS+('pp2ep4_rdma','ep8_rdma') if args.include_rdma else ARMS
    assert (ROOT/'ce_serving.py').is_file(), 'CE serving adapter must exist before freeze'
    nodes=['spartan-gpgpu168','spartan-gpgpu169'];jobs=['32183471','32446403']
    historical=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/runtime/glm47_h100_pp2ep4_boundary_v3_20261007_r1/config.json')
    old=json.loads(historical.read_text());workloads={}
    for key in ['prefill_c4']:
        entry=dict(old['workloads'][key]);dest=run/'inputs'/Path(entry['input_artifact']).name
        dest.parent.mkdir(exist_ok=True);shutil.copy2(entry['input_artifact'],dest)
        assert sha(dest)==entry['input_sha256'];entry['input_artifact']=str(dest);workloads[key]=entry
    assert sha(NATIVE)==NATIVE_SHA
    sources=[dict(worktree=str(tree),commit=commit,tag=tag) for tree,commit,tag in
        [(SG,SG_COMMIT,SG_TAG),(DAEMON_SG,DAEMON_COMMIT,DAEMON_TAG),(DEEP,DEEP_COMMIT,DEEP_TAG)]]
    specs={};usage={};cases={};paths=[]
    for idx,(node,job) in enumerate(zip(nodes,jobs)):
        previous=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/runtime/glm47_attention_tpdp_efficiency_20261006_r2/cache/specs')/(node+'.json')
        cache_node=json.loads(previous.read_text());cache_node['expected_job_id']=job
        assert cache_node['engine_root']==str(DAEMON_SG) and cache_node['source_commit']==DAEMON_COMMIT
        node_path=run/'model-cache/specs'/previous.name;save(node_path,cache_node)
        specs[node]=dict(path=str(node_path),sha256=sha(node_path));paths.append(node_path);usage[node]=[]
        for arm in arms:
            log=run/'model-cache/usage'/f'{node}.{arm}.server.log'
            log.parent.mkdir(parents=True,exist_ok=True,mode=0o700);log.parent.chmod(0o700)
            ranks=[f'PP{idx} TP{i}' for i in range(4)] if arm.startswith('pp2ep4') else [f'TP{i}' for i in range(idx*4,idx*4+4)]
            usage[node].append(dict(id=f'{run.name}-{node}-{arm}',path=str(log),expected_ranks=ranks))
    for idx,arm in enumerate(arms):
        pp=2 if arm.startswith('pp2ep4') else 1;ep=4 if pp==2 else 8;dp=1 if pp==2 else 2
        cases[arm]=dict(layout='pp2ep4' if pp==2 else 'ep8',transport='rdma' if arm.endswith('_rdma') else 'ce',nodes=nodes,jobs=jobs,world=8,tp=4 if pp==2 else 8,ep=ep,pp=pp,dp=dp,
            moe_layers_by_rank=[43]*4+[46]*4 if pp==2 else [89]*8,attn_tp=4,ep_capacity=256,
            local_prefill_chunk=1024,global_prefill_chunk=1024*dp,api_port=args.api_port_base+idx,dist_port=args.dist_port_base+idx,base_url=f'http://{nodes[0]}:{args.api_port_base+idx}',
            usage_log_ids={node:next(row['id'] for row in usage[node] if row['id'].endswith(arm)) for node in nodes})
    cache=dict(policy='required',provider='sglang_canonical_full_weight_v1',support_profile='glm4moe-canonical-v1',
        on_miss='build',node_specs=specs,usage_logs=usage,hicache_reserve_bytes_per_node={node:0 for node in nodes})
    config=dict(id=run.name,nodes=nodes,jobs=jobs,cases=cases,arms=list(arms),groups={'a100_pair':dict(nodes=nodes,arms=list(arms))},
        source_worktree=str(SG),source_commit=SG_COMMIT,source_tag=SG_TAG,model_path=str(MODEL),
        model_revision='local-config-sha256:'+sha(MODEL/'config.json'),python=PYTHON,
        humming_backend_path=str(HUMMING_ROOT),humming_seed_dir=str(SEED),
        deepep_native=dict(path=str(NATIVE),sha256=NATIVE_SHA),deepep_source=dict(worktree=str(DEEP),commit=DEEP_COMMIT,tag=DEEP_TAG),
        model_cache=cache,arm_timeout_s=3600,workloads=workloads,benchmark=dict(warmup_trials=1,formal_trials=1,profiler_enabled=False),
        caveats=['Same two A100 nodes, sequential arms, 8 GPUs, identical frozen Prefill C4 inputs, 8192 input and forced one output token.',
            'Attention TP4 per node: PP2 EP4 has two pipeline stages; EP8 has Attention DP2, TP4 within each node.',
            'CE native/source frozen by exact SHA/commit; same candidate can optionally run cross-pair RDMA controls by setting threshold=0.',
            'CE only handles local cross-NVLink-pair traffic; xor1 uses NVLink; cross-node remains NVSHMEM RDMA.',
            'Equal local scheduler Prefill chunk1024 and source capacity256 in both arms; PP2 global1024, EP8 AttentionDP2 global2048. Matches the five-way Prefill policy comparison.',
            'AWQ INT4 Humming grouped-masked on SM80, unchanged kernels/package in both arms.',
            'Prefill and Decode CUDA Graph disabled because CE uses dynamic host count preparation.',
            'Radix/HiCache/overlap/shared-expert fusion/custom-AR disabled; frozen cold-prefix API workload.',
            'PP2 boundary gathers hidden and residual separately; stage0 43 MoE layers, stage1 46.',
            'One warmup batch and one formal batch; request mean/median/P99 describe four requests, not a repeated-service tail.',
            'CPU weight cache required on both nodes, on_miss build and exact per-arm usage ranks; startup and correctness excluded from timing.',
            'Selected remote API/dist ports are new values, not asserted free; normal launch detects bind failure.'])
    save(run/'config.json',config)
    save(run/'kv_cache_policy.json',dict(policy='cold_prefix_no_hicache',restore_used=False,reason='Matched frozen Prefill8192 to one-token API workload; no prefix reuse'))
    paths+=list(ROOT.glob('*.py'))+[run/'config.json',run/'kv_cache_policy.json',MODEL/'config.json',MODEL/'model.safetensors.index.json',NATIVE,SHIM]
    paths+=[NATIVE.parents[1]/'build_result.json',DEEP/'csrc/kernels/internode_ll.cu']
    suffixes={'.py','.h','.hpp','.c','.cc','.cpp','.cu','.cuh'}
    paths+=[p for p in HUMMING_ROOT.rglob('*') if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts]
    paths+=[p for p in SEED.rglob('*') if p.is_file() and p.name!='lock' and not p.name.endswith('.lock')]
    paths+=list((run/'inputs').glob('*'))+[DEEP/'deep_ep'/name for name in ['__init__.py','buffer.py','utils.py']]
    for tree in [SG,DAEMON_SG]:paths+=list((tree/'python/sglang/srt/weight_cache').glob('*.py'))
    paths+=[SG/'python/sglang/srt'/p for p in ['models/glm4_moe.py','distributed/parallel_state.py','layers/quantization/humming.py','layers/moe/moe_runner/humming.py','layers/moe/token_dispatcher/deepep.py','model_loader/loader.py']]
    paths+=[GUARD_SCRIPTS/name for name in ['dashboard_guard.py','model_cache_adapter.py','cpu_model_cache_daemon.py','cpu_budget_migration.py','gpu_registry.py','artifact_visibility.py']]
    for path in ROOT.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    frozen=[dict(path=str(p.resolve()),sha256=sha(p)) for p in sorted(set(paths))]
    save(run/'frozen_inputs.json',dict(files=frozen,timestamp=time.time()))
    spec=dict(schema_version=1,id=run.name,title='GLM A100 x8: CE Prefill EP8 versus PP2 EP4, C4',engine='sglang',protocol='task-events-v2',
        template='parallel-tasks-v5',features=['parallel-tasks','task-progress','task-details','gpu-tab','gpu-telemetry','guard'],
        gpu_telemetry=dict(enabled=True,poll_interval_s=10,timeout_s=10),port=args.dashboard_port,max_active_nodes=2,gpu_lease_ttl_s=90,cleanup_wait_s=60,
        runtime_dir=str(run/'dashboard_guard'),sources=sources,model_cache=cache,frozen_inputs=frozen,
        tasks=[dict(id='a100_pair',label='PP2 EP4 / EP8: CE Prefill C4',root=str(run/'a100_pair/task'),expected=len(arms),nodes=nodes,controller_script=str(ROOT/'controller.py'))])
    save(run/'dashboard_spec.json',spec)
    save(run/'a100_pair/task/progress.json',dict(timestamp=time.time(),stage='queued',valid_points=0,planned_requests=4*len(arms)*2,completed_requests=0,note='Guard admission and exact CPU bank reuse; no GPU process launched'))
    print(json.dumps(dict(prepared=True,run=str(run),frozen_files=len(frozen),gpu_execution=False)))

if __name__=='__main__':main()
