"""Archive validated H100 fixed-data EP2/EP4 communication results, without private lease data."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/h100178_ep2_ep4_r1'
ID='deepep_ep2_ep4_h100_20261027'
DEST=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')/ID
A100=DEST.parent/'deepep_ep2_ep4_a100_20261027'
REG=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
INDEX=ROOT.parent/'experiment_summary/README.md'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())
def save(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def cp(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

def update_once(path,text,anchor=None):
    marker=f'<!-- begin {ID} -->';prior=path.read_text()
    if marker in prior:return False
    if anchor is None:after=prior.rstrip()+'\n\n'+text+'\n'
    else:
        assert anchor in prior;after=prior.replace(anchor,text+'\n'+anchor,1)
    assert path.read_text()==prior,'Concurrent index update; rerun indexing'
    tmp=path.with_name(path.name+f'.{ID}.{os.getpid()}.tmp');tmp.write_text(after);tmp.replace(path)
    return True


def main():
    report=RUN/'report';audit=load(report/'audit.json');cfg=load(RUN/'config.json');spec=load(RUN/'dashboard_spec.json')
    cleanup=load(RUN/'task/cleanup.json');scientific=load(RUN/'comparison.json')
    assert audit['passed'] and audit['checked_points']==72 and audit['checked_rank_samples']==6480
    assert audit['A100_same_global_routes_verified'] and audit['H100_full_NVLink_route_policy_verified']
    assert cleanup['passed'] and len(cleanup['evidence'])==8 and scientific['passed']
    assert all(e['returncode']==0 and e['gpu_idle'] and e['owned_processes_exited'] for e in cleanup['evidence'])
    posts=list((RUN/'dashboard_guard/checks').glob('*post.json'));assert len(posts)==1
    post=load(posts[0]);assert post['passed'] and post['cleanup_verified'] and post['scientific_success']
    if (DEST/'manifest.json').exists():assert load(DEST/'manifest.json')['original_run']==str(RUN)
    DEST.mkdir(parents=True,exist_ok=True)
    for p in report.iterdir():
        if p.is_file():cp(p,DEST/p.name)
    cp(report/'REPORT.md',DEST/'summary.md')
    for name in ('full_metrics.csv','raw_samples.csv','comparison.json','config.json','frozen_inputs.json','cpu_tests.json','launch.json','dashboard_spec.json'):cp(RUN/name,DEST/name)
    cp(RUN/'task/cleanup.json',DEST/'cleanup.json')
    cp(RUN/'spartan-gpgpu178_nic.json',DEST/'topology/nic.json')
    for phase in cfg['phases']:
        for rank in range(phase['world']):cp(RUN/phase['id']/f'result_rank{rank}.json',DEST/'raw_rank_results'/phase['id']/f'result_rank{rank}.json')
        cp(RUN/phase['id']/'spartan-gpgpu178_topology.json',DEST/'topology'/f'{phase["id"]}.json')
    for p in (RUN/'frozen/harness').glob('*.py'):cp(p,DEST/'source/frozen_harness'/p.name)
    for name in ('build_report.py','archive_results.py'):cp(ROOT/name,DEST/'source'/name)
    for name in ('manifest.json','config.json','full_metrics.csv','comparisons.csv'):cp(A100/name,DEST/'a100_reference'/name)
    allowed=('timestamp','phase','experiment_id','state','valid','expected','exit_code','cleanup_verified','scientific_success','passed')
    save(DEST/'guard_postcheck.json',dict(original_path=str(posts[0]),original_sha256=sha(posts[0]),**{k:post[k] for k in allowed if k in post}))
    source=spec['sources'][0]
    branch=subprocess.check_output(['git','-C',source['worktree'],'branch','--show-current'],text=True).strip()
    assert not subprocess.check_output(['git','-C',source['worktree'],'status','--porcelain'],text=True).strip()
    save(DEST/'manifest.json',dict(schema_version=1,id=ID,title='H100 DeepEP fixed-global-data EP2 versus EP4: NVLink and RDMA',
        created_at=datetime.now(timezone.utc).isoformat(),status='complete_validated',original_run=str(RUN),
        implementation=dict(engine='DeepEP',worktree=source['worktree'],branch=branch,commit=source['commit'],tag=source['tag'],dirty=False,native=cfg['native'],architecture='SM90',old_implementation=False),
        deployment=dict(id=cfg['id'],slurm_job_id=cfg['booked_job_id'],nodes=cfg['nodes'],guard_postcheck_passed=True,cleanup_passed=True,lease_release_observed=True,server_endpoint=None),
        model_testbed=dict(model_loaded=False,gpu='NVIDIA H100 80GB HBM3',gpus_per_node=4,nodes_used=1,nic='mlx5_bond_0 PCI 15b3:1021',topology='four GPUs pairwise NV6; exact topology captured in every phase'),
        workload=dict(kind='communication-only identity Expert',hidden=5120,experts=160,topk=8,dtype='BF16',global_tokens=cfg['token_sweep'],global_routes_fixed=True,expert_compute=False,attention_compute=False,kv_cache=False),
        engine_parameters=dict(EP_degrees=[2,4],PP=None,TP=None,DP=None,backend='same H100 SM90 low_latency BF16 API for all H100 arms',pair_only=False,CE=False,FP8=False,capacity_per_rank=1024,NVSHMEM_QP_DEPTH=4096,small_modes=['eager','graph'],large_modes=['eager'],warmup=10,timed_samples=30,opposite_order_rounds=2),
        validation=dict(points=72,rank_samples=6480,raw_rank_records=24,independent_audit='audit.json',hash_checks_passed=True,A100_global_routes_and_protocol_matched=True),
        key_results=dict(NVLink_small_EP4_over_EP2=[1.2330,1.2362,1.0726],NVLink_large_EP4_over_EP2=[0.5781,0.3938,0.3614],RDMA_large_EP4_over_EP2=[1.2650,1.2511,1.2347],large2048_EP2_us=931.3680,large2048_EP4_us=336.6400,large2048_latency_reduction_pct=63.8553),
        A100_reference=dict(snapshot=str(A100),new_A100_measurement=False,difference='A100 pair-local NVLink plus cross-pair RDMA versus H100 full local NVLink; SM80/SM90 wrapper/native builds differ'),
        artifacts=dict(report='REPORT.md',metrics='full_metrics.csv',samples='raw_samples.csv',aggregate='aggregate.csv',comparisons='comparisons.csv',cross_platform='a100_h100_comparison.csv',plots=['latency.png','components.png','rdma_controls.png','a100_h100_domains.png'],raw_rank_results='raw_rank_results',source='source',cleanup='cleanup.json',guard='guard_postcheck.json'),
        scope='Communication latency only; no full-model TTFT/TPOT, normal-mode Prefill, cross-node test or measured NIC wire bandwidth. No weights, KV payloads, authorization packets, lease tokens, native binaries or credentials copied.'))
    (DEST/'REPRODUCE.md').write_text(f'''# Reproduction\n\nOriginal completed run: `{RUN}`. All eight phases, scientific validation, Guard postcheck, cleanup and lease release passed.\n\nCPU report rebuild:\n\n```bash\n/home/bxb1/conda/bin/python {ROOT}/build_report.py --run {RUN} --out /tmp/deepep_h100_ep2_ep4_report\n```\n\nA GPU rerun needs a fresh run ID, a current four-H100 booking, the pinned SM90 native and the Experiment Guard workflow. Use `source/frozen_harness/prepare.py` to create a new frozen run; do not rerun into the old completed run. All global routes and workload/timing parameters match the saved A100 reference, while hardware topology and native architecture differ.\n''')
    files=sorted(p for p in DEST.rglob('*') if p.is_file() and p.name!='checksums.sha256')
    (DEST/'checksums.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(DEST)}\n' for p in files))
    marker=f'<!-- begin {ID} -->';end=f'<!-- end {ID} -->'
    entry=f'''{marker}
### {ID}

Snapshot: {DEST}

- Completed communication-only identity Expert on existing H100178 booking33646188; four H10080GB HBM3 GPUs verified pairwise NV6. Fixed global8/16/32/256/1024/2048 BF16 H5120/E160/top8, identical routes to saved A100. EP2 versus EP4 within NVLink, plus both degrees forced RDMA; same pinned SM90 native in all H100 arms. Small eager/Graph and large eager low-latency API.
- NVLink EP4/EP2: small Graph1.233/1.236/1.073x, large eager0.578/0.394/0.361x. At2048 tokens D+C931.368 to336.640us(-63.9%). RDMA-only large ratios1.265/1.251/1.235x. A100 practical expansion crossed NVLink pairs; H100 expansion stays within NVLink. Cross-platform topology/native differences retained.
- All72 points/6480 rank samples independently recomputed from24 rank results; exact source/count/payload/weighted identity, changed-payload Graph, distinct-expert and skew checks, frozen hashes and matched A100 route/protocol checks pass. Eight owned cleanups, Guard scientific/postcheck and lease release pass.

Artifacts: [report]({DEST}/REPORT.md), Dispatch/Combine CSVs, NVLink/component/RDMA/domain PNG/PDF figures, raw rank JSONs, source/native lineage, audit, sanitized Guard proof and checksums. No model/cache payload or private authorization copied.

{end}'''
    reg=update_once(REG,entry)
    row=f'''| [T03：H100 固定总量 EP2→EP4]({DEST}/REPORT.md) · [NVLink 图]({DEST}/latency.png)／[D、C 分项]({DEST}/components.png)／[A100 对照]({DEST}/a100_h100_domains.png) {marker}{end} | H100178 四卡两两 NV6；全局 token、路由和协议匹配 T02。小消息 Graph EP4/EP2 为 1.233/1.236/1.073 倍；大消息 eager 为 0.578/0.394/0.361 倍。统一 RDMA 后大消息为 1.265/1.251/1.235 倍。72 点，两轮每点 30 次。 | 纯 Dispatch／Combine，无模型或 Expert GEMM。H100 扩卡仍在 NVLink 域内，A100 T02 跨 NVLink 对；SM90/SM80 native 不同。独立审计、Guard postcheck、清理及释放通过。 |'''
    idx=update_once(INDEX,row,anchor='| [H100 EP4/EP8 纯通信表]')
    assert not update_once(REG,entry) and not update_once(INDEX,row,anchor='| [H100 EP4/EP8 纯通信表]')
    print(json.dumps(dict(archived=True,path=str(DEST),files=len(files),registry_updated=reg,index_updated=idx,idempotent=True),ensure_ascii=False))


if __name__=='__main__':main()
