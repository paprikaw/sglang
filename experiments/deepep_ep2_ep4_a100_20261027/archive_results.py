"""Archive only validated small evidence, then update the communication index."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/a100065_ep2_ep4_r4'
DATA=RUN/'analysis_v1'
REPORT=DATA/'report'
ID='deepep_ep2_ep4_a100_20261027'
DEST=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')/ID
REG=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
INDEX=ROOT.parents[0]/'experiment_summary/README.md'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def cp(source,target):target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)

def update_once(path,text,anchor=None):
    marker=f'<!-- begin {ID} -->'
    prior=path.read_text()
    if marker in prior:return False
    if anchor is None:after=prior.rstrip()+'\n\n'+text+'\n'
    else:
        assert anchor in prior
        after=prior.replace(anchor,text+'\n'+anchor,1)
    assert path.read_text()==prior,'Concurrent index change; rerun indexing'
    tmp=path.with_name(path.name+f'.{ID}.{os.getpid()}.tmp');tmp.write_text(after);tmp.replace(path)
    return True


def main():
    audit=load(REPORT/'audit.json');assert audit['passed'] and audit['checked_points']==72 and audit['checked_rank_samples']==6480
    cleanup=load(RUN/'task/cleanup.json');assert cleanup['passed'] and len(cleanup['evidence'])==8
    cfg=load(RUN/'config.json');spec=load(RUN/'dashboard_spec.json')
    assert all(e['returncode']==0 and e['gpu_idle'] and e['owned_processes_exited'] for e in cleanup['evidence'])
    if (DEST/'manifest.json').exists():assert load(DEST/'manifest.json')['original_run']==str(RUN)
    DEST.mkdir(parents=True,exist_ok=True)
    for p in REPORT.iterdir():
        if p.is_file():cp(p,DEST/p.name)
    cp(REPORT/'REPORT.md',DEST/'summary.md')
    for name in ('full_metrics.csv','raw_samples.csv','comparison.json'):cp(DATA/name,DEST/name)
    for name in ('config.json','frozen_inputs.json','cpu_tests.json','launch.json','dashboard_spec.json'):cp(RUN/name,DEST/name)
    cp(RUN/'task/cleanup.json',DEST/'cleanup.json')
    cp(RUN/'spartan-gpgpu065_nic.json',DEST/'topology/nic.json')
    for phase in cfg['phases']:
        for rank in range(phase['world']):cp(RUN/phase['id']/f'result_rank{rank}.json',DEST/'raw_rank_results'/phase['id']/f'result_rank{rank}.json')
        cp(RUN/phase['id']/'spartan-gpgpu065_topology.json',DEST/'topology'/f'{phase["id"]}.json')
    for p in (RUN/'frozen/harness').glob('*.py'):cp(p,DEST/'source/frozen_harness'/p.name)
    for p in (ROOT/'harness').glob('*.py'):cp(p,DEST/'source/current_harness'/p.name)
    for name in ('build_report.py','archive_results.py'):cp(ROOT/name,DEST/'source'/name)
    failures=[]
    for label,reason in [('r1','pair_only rejects EP2'),('r2','capacity requires larger NVSHMEM QP depth'),('r3','oversized QP depth caused native NIC queue creation failure'),('r4','all GPU phases passed; CPU analyzer duplicate iteration keyword failed')]:
        r=ROOT/f'runs/a100065_ep2_ep4_{label}'
        post=list((r/'dashboard_guard/checks').glob('*post.json'))
        failures.append(dict(run=str(r),reason=reason,controller_failure=load(r/'controller_failure.json'),
            cleanup=load(r/'task/cleanup.json'),original_postchecks=[dict(path=str(p),sha256=sha(p)) for p in post]))
    save(DEST/'failure_history.json',failures)
    source=spec['sources'][0]
    branch=subprocess.check_output(['git','-C',source['worktree'],'branch','--show-current'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',source['worktree'],'status','--porcelain'],text=True).strip()
    assert not dirty
    save(DEST/'manifest.json',dict(schema_version=1,id=ID,
        title='A100 DeepEP fixed-global-data EP2 versus EP4: small and large messages',
        created_at=datetime.now(timezone.utc).isoformat(),status='validated_offline_after_cpu_aggregation_failure',
        original_run=str(RUN),original_guard_postcheck='failed_cpu_analyzer_duplicate_iteration_keyword',
        offline_analysis=str(DATA),gpu_phases_all_passed=True,cleanup_passed=True,lease_release_observed=True,
        implementation=dict(engine='DeepEP',worktree=source['worktree'],branch=branch,commit=source['commit'],tag=source['tag'],dirty=False,old_implementation=False,native=cfg['native']),
        deployment=dict(id=cfg['id'],slurm_job_id=cfg['booked_job_id'],nodes=cfg['nodes'],server_endpoint=None),
        model_testbed=dict(model_loaded=False,gpu='NVIDIA A100 80GB PCIe',gpus_per_node=4,nodes_used=1,nic='ConnectX-6 Dx mlx5_bond_0',topology='NVLink pairs GPU0/1 and GPU2/3; cross-pair RDMA'),
        workload=dict(kind='communication-only identity Expert',hidden=5120,experts=160,topk=8,dtype='BF16',global_tokens=cfg['token_sweep'],global_routes_fixed=True,expert_compute=False,attention_compute=False,kv_cache=False),
        engine_parameters=dict(EP_degrees=[2,4],PP=None,TP=None,DP=None,backend='same low_latency BF16 API all sizes',CE=False,FP8=False,capacity_per_rank=1024,NVSHMEM_QP_DEPTH=4096,small_modes=['eager','graph'],large_modes=['eager'],warmup=10,timed_samples=30,opposite_order_rounds=2),
        validation=dict(points=72,rank_samples=6480,independent_audit='audit.json',raw_rank_records=24,hash_checks_passed=True),
        key_results=dict(practical_EP4_over_EP2_small_graph=[1.6433,2.2200,3.1928],practical_EP4_over_EP2_large_eager=[9.9137,17.6750,21.6436],RDMA_only_large_EP4_over_EP2=[1.4905,1.4691,1.4961]),
        interpretation='Practical 2-to-4 expansion crosses NVLink pairs. RDMA-only large-message ratios track the 50% increase in remote assignments, not measured wire bytes.',
        artifacts=dict(report='REPORT.md',summary='summary.md',metrics='full_metrics.csv',raw='raw_samples.csv',aggregate='aggregate.csv',comparisons='comparisons.csv',plot='latency.png',components='components.png',sources='source',raw_rank_results='raw_rank_results',cleanup='cleanup.json',failure_history='failure_history.json'),
        scope='No full-model TTFT/TPOT, normal-mode Prefill or cross-node experiment; no weights, KV payloads, authorization packets, lease tokens, credentials or native binaries copied'))
    (DEST/'REPRODUCE.md').write_text(f'''# Reproduction\n\nOriginal run: `{RUN}`. The original frozen GPU worker and input hashes are retained. The Guard postcheck failed only in CPU aggregation after all eight phases exited successfully. `source/current_harness/analyze.py` removes a duplicated `iteration` keyword; `analysis_v1` contains the independently validated offline outputs.\n\nOffline from the original run:\n\n```bash\npython3 {ROOT}/harness/analyze.py --run {RUN} --out /tmp/deepep_ep2_ep4_offline\n/home/bxb1/conda/bin/python {ROOT}/build_report.py --run {RUN} --out /tmp/deepep_ep2_ep4_report\n```\n\nFor a fresh GPU rerun use `source/current_harness/prepare.py` with a new run path, current booked node/job and the Experiment Guard skill. Never rerun into or edit an old frozen run. The repaired current harness is distinct from the archived original frozen harness.\n''')
    files=sorted(p for p in DEST.rglob('*') if p.is_file() and p.name!='checksums.sha256')
    (DEST/'checksums.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(DEST)}\n' for p in files))
    marker=f'<!-- begin {ID} -->';end=f'<!-- end {ID} -->'
    entry=f'''{marker}
### {ID}

Snapshot: {DEST}

- Completed communication-only identity Expert on A100065, NVIDIA A100 80GB PCIe, fixed global8/16/32/256/1024/2048 BF16 H5120 tokens,160experts,top8. EP2 NVLink versus EP4 paired-NVLink+RDMA, plus both degrees forced RDMA. Small eager/Graph, large eager, same low-latency API; no model compute.
- Practical EP4/EP2 D+C: small Graph1.64/2.22/3.19x, large9.91/17.67/21.64x. RDMA-only large ratios1.47-1.50x track global remote assignments+50%; no measured wire-bandwidth or full-PP speedup claim.
- Eight GPU phases passed,72 points/6480 rank samples independently recomputed, payload/source/Graph/weighted/skew checks and frozen hashes pass. Owned cleanup and lease release verified. Original Guard postcheck remains failed because CPU analyzer duplicated an iteration keyword; separate offline analysis repaired aggregation without changing GPU data. Three no-sample startup failures preserved.

Artifacts: [report]({DEST}/REPORT.md), Dispatch/Combine CSVs, two scientific PNG/PDF figures,24 raw rank records, exact source/native lineage, audit, cleanup, immutable failure history and checksums. No private authorization or model/cache payload copied.

{end}'''
    reg_changed=update_once(REG,entry)
    indexrow=f'''| [T02：A100 固定总量 EP2→EP4]({DEST}/REPORT.md) · [通信图]({DEST}/latency.png)／[D、C 分项]({DEST}/components.png) {marker}{end} | A100065，固定全组 8/16/32/256/1024/2048 token，BF16 H5120、160 experts、top8；双卡 NVLink／四卡混合、RDMA-only 对照；小消息 eager/Graph、大消息 eager，同一 low-latency API。72 点，两轮每点 30 次。实际扩卡小消息 1.64–3.19 倍，大消息 9.91–21.64 倍；统一 RDMA 后大消息约 1.47–1.50 倍。 | 只有 Dispatch／Combine，无 Expert GEMM、Attention、模型或请求 TPOT。四卡跨出 NVLink 对。GPU 测量及清理均完成；原 Guard 汇总失败原样保留，离线重算 6480 rank samples 通过。 |'''
    index_changed=update_once(INDEX,indexrow,anchor='| [H100 EP4/EP8 纯通信表]')
    assert not update_once(REG,entry) and not update_once(INDEX,indexrow,anchor='| [H100 EP4/EP8 纯通信表]')
    print(json.dumps(dict(archived=True,path=str(DEST),files=len(files),registry_updated=reg_changed,index_updated=index_changed,idempotent=True),ensure_ascii=False))


if __name__=='__main__':main()
