"""Archive validated small summaries; retain large logs in their original runs."""
from pathlib import Path
import datetime,fcntl,hashlib,json,re,shutil,subprocess
ROOT=Path(__file__).resolve().parent
WS=ROOT.parents[1]
EM=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager')
ID='qwen35b_h100_five_layouts_hicache_20261028'

def read(p):return json.loads(p.read_text())
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def main():
    summary=read(ROOT/'summary.json');validation=read(ROOT/'validation.json')
    assert summary['status']=='complete' and validation['passed'] is True and not validation['partial']
    dest=EM/'experiments'/ID;dest.mkdir(parents=True,exist_ok=True)
    for name in ['SUMMARY.md','REPORT.md','summary.json','validation.json','full_metrics.csv','management_api_validation.json','pp_cpu_validation.json','source_provenance.json']:
        p=ROOT/name
        if p.exists():shutil.copy2(p,dest/name)
    implementation=[];deployments=[]
    for check in summary['runs']:
        run=ROOT/'runs'/check['run'];cfg=read(run/'config.json')
        target=dest/'provenance'/run.name;target.mkdir(parents=True,exist_ok=True)
        for name in ['config.json','dashboard_spec.json','launch_manifest.json','kv_checkpoint_discovery.json','task/cleanup.json']:
            p=run/name;t=target/name;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
        post=Path(check['guard_postcheck']);shutil.copy2(post,target/'guard_postcheck.json')
        implementation.append(dict(run=run.name,engine='sglang',worktree=cfg['engine_root'],commit=cfg['engine_commit'],branch=read(ROOT/'source_provenance.json')['branch'],source_tag='experiment/sglang/glm47-ep16-canonical-profile-20261004-cpu-validated',old_implementation=False,dirty_state=read(ROOT/'source_provenance.json'),frozen_inputs=read(run/'dashboard_spec.json')['frozen_inputs']))
        deployments.append(dict(id=cfg['id'],tag=cfg['id'],job=cfg['job_id'],node=cfg['node'],gpu_type='H100 80GB',gpu_count=4,layouts=cfg['layouts'],endpoint='Node-local loopback endpoints recorded by frozen runner/config; all retired',gateway_route=None,rollback_endpoint=None,model_path=cfg['model_path'],model_revision=cfg['model_revision'],source_run=str(run),launch_manifest=str(run/'launch_manifest.json'),validation=str(post),cleanup=check['cleanup'],lease_released=True))
    ledger=[]
    for run in sorted((ROOT/'runs').iterdir()):
        if run.name in [r['run'] for r in summary['runs']]:continue
        p=run/'task/progress.json'
        if not p.exists():continue
        progress=read(p)
        if progress.get('stage')=='complete':continue
        cleanup=read(run/'task/cleanup.json') if (run/'task/cleanup.json').exists() else None
        ledger.append(dict(run=run.name,status='failed_or_interrupted_excluded',progress=progress,cleanup=cleanup,manual_recovery=read(run/'manual_recovery.json') if (run/'manual_recovery.json').exists() else None,original_logs=str(run)))
    save(dest/'excluded_runs.json',ledger)
    manifest=dict(id=ID,title='Qwen35B: EP4 / 2xEP2 / 4xEP1 / 2xPP2 / PP4 with native GPU and CPU HiCache',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='complete',implementation=implementation,deployment=deployments,workload=dict(dataset='Deterministic synthetic private-prefix token IDs; hashes in raw workload.json',request_count=64,P=[8256,130304],prefill=dict(Q=512,output=1),decode=dict(Q=1,output=128),formal_cohorts=3,ordering='Concurrent client barrier; fixed deterministic request-to-replica/DP mapping',thinning=None,request_rate='closed B64 cohort',cache_modes=['natural_replay','cpu_replay']),engine_parameters=dict(dtype='BF16',KV='BF16',GDN='FP32',page_size=64,context_length=262144,hicache=True,hicache_write_policy='write_through',hicache_io='kernel',host_pools_per_GPU_bytes=dict(KV=64000000000,GDN=8000000000),EP_mem_fraction_static=.94,PP_mem_fraction_static=.92,PP_decode_graph_max_bs=16,PP_candidates='Valid native/early-send with workload-specific microbatch selection; lookahead/experimental shape transport failed qualification, tail-local2 failed long-context slot reuse; both excluded',CPU_KV_checkpoint_replay=False),artifacts=dict(snapshot=str(dest),original_run_root=str(ROOT),summary='SUMMARY.md',report='REPORT.md',round_metrics='full_metrics.csv',validation='validation.json',large_logs='Referenced, not copied',secrets='API key values, weights and KV bytes excluded'),key_results=summary['metrics'],caveats=['Synthetic prefix reuse is not a production trace hit-rate estimate.','Natural replay mixes actual GPU and CPU hits; controlled CPU replay includes host-to-device restoration inside client timing.','All five formal layouts use the same node178; node171 exploratory candidates are excluded.','PP selection means best measured valid candidate, not an exhaustive global optimum.','Old no-HiCache results are a separate cohort.'])
    save(dest/'manifest.json',manifest)
    shutil.copy2(ROOT/'SUMMARY.md',dest/'summary.md')
    entry=f'\n<!-- begin {ID} -->\n### {ID}\n\nFive four-GPU H100 layouts, Qwen3.6-35B-A3B BF16, native GPU/CPU HiCache enabled throughout. B64, P8256/P130304; new Prefill Q512->1 and Decode Q1->128; natural replay and controlled CPU replay, three formal cohorts, phase-specific PP pilot selection. Exact native tier hits, request timings, capacities, frozen inputs, canonical CPU weight-cache usage, owned cleanup and released Guard leases audited. All five formal layouts are on node178; node171 exploratory runs are excluded. Additional B64 capacity cohorts do not change the standard132-cell benchmark count.\n\n[Simple report]({dest}/SUMMARY.md) · [Full report]({dest}/REPORT.md) · [Round data]({dest}/full_metrics.csv)\n\nSnapshot: {dest}\n<!-- end {ID} -->\n'
    for registry in [Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md'),WS/'experiments/experiment_summary/README.md']:
        with registry.open('r+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);s=f.read();s=re.sub(r'\n?<!-- begin '+re.escape(ID)+r' -->.*?<!-- end '+re.escape(ID)+r' -->\n?', '\n',s,flags=re.S)
            f.seek(0);f.write(s.rstrip()+'\n'+entry);f.truncate()
    print(json.dumps(dict(snapshot=str(dest),archived_status='complete',excluded_runs=len(ledger)),ensure_ascii=False))

if __name__=='__main__':main()
