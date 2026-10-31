"""Record final read-only evidence and live cleanup state without changing run receipts."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WS=ROOT.parents[1]
def read(p):return json.loads(p.read_text())
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main():
    outcomes=[];allrows=[];later=[]
    regpath=Path('/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/gpu_registry/registry.json')
    registry=read(regpath)
    registry_rows=registry['leases']+registry['history']
    for run in sorted((ROOT/'runs').iterdir()):
        cfg=read(run/'config.json')
        audit=read(run/'analysis_audit.json') if (run/'analysis_audit.json').exists() else None
        superseded=read(run/'superseded.json') if (run/'superseded.json').exists() else None
        leases=[{k:x.get(k) for k in ('experiment_id','lease_id','state','cleanup_verified','ended_at','end_reason','nodes')} for x in registry_rows if x.get('experiment_id')==run.name]
        if (run/'dashboard_guard/gpu_registry_lease.json').exists():assert len(leases)==1 and leases[0]['state'] in ('released','expired'), 'Task lease still active or missing'
        cleanup=[dict(**ref(p),verified=read(p).get('cleanup_verified')) for p in run.glob('evidence/*/cleanup.json')]
        checks=[dict(**ref(p),passed=read(p).get('passed')) for p in sorted(run.glob('dashboard_guard/checks/*-post.json'))]
        completed=list(run.glob('results/*/*/completed.json'))
        if completed:
            assert not superseded and audit['frozen_inputs_match']
            assert len(audit['completed_arm_loader_checks'])==2
            for cp in completed:
                assert cp.parent.parent.name=='colocated'
                for p in sorted(cp.parent.glob('formal.r*.json')):
                    rows=read(p)['requests'];allrows.extend(rows);later.extend(x for x in rows if not x['primer'])
        errors=[]
        for p in run.glob('cache/pd.r*.log'):
            for n,line in enumerate(p.read_text(errors='replace').splitlines(),1):
                if 'ValueError: --disaggregation-decode-enable-radix-cache is incompatible with Mamba/SSM models' in line:
                    errors.append(dict(path=str(p),line=n,error=line.strip()))
        if run.name.endswith('_r4'):assert errors and not completed
        outcomes.append(dict(id=run.name,config=ref(run/'config.json'),superseded=superseded,
            completed_performance_points=len(completed),audit=ref(run/'analysis_audit.json') if audit else None,
            cleanup=cleanup,leases=leases,original_guard_postchecks=checks,decode_cache_failures=errors))
    assert len(allrows)==480 and len(later)==384
    tiers={k:sum((x['meta_info'].get('cached_tokens_details') or {}).get(k,0) for x in later) for k in ('device','host','storage')}
    sources=[]
    for base in (WS/'.engine-worktrees/qwen35b-decode-ep-overlap-20261027',WS):
        p=base/'python/sglang/srt/mem_cache/kv_cache_builder.py'
        content=p.read_text();lines=content.splitlines()
        hits=[n for n,s in enumerate(lines,1) if 'with Mamba/SSM models' in s]
        assert len(hits)==1
        n=hits[0]
        sources.append(dict(**ref(p),line=n-2,excerpt='\n'.join(lines[n-5:n+1])))
    backfill_path=WS/'experiments/cluster_backfill_20261020/state.json'
    state=read(backfill_path)
    backfill={n:{k:state['nodes'][n].get(k) for k in ('state','reason','experiment_id','active_experiment_id')} for n in ('spartan-gpgpu011','spartan-gpgpu171')}
    findings=dict(schema_version=1,status='feasibility_audit_complete_pd_cache_unsupported',
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        valid_colocated_performance_points=4,formal_repetitions=12,formal_requests=len(allrows),later_turn_requests=len(later),
        later_turn_cache_tokens=tiers,mean_later_turn_gpu_cached_tokens=tiers['device']/len(later),
        pd_performance_points=0,pd_output_equivalence_verified=False,dynamic_allocation_gain_verified=False,
        blocker='Pinned and workspace engines reject radix prefix cache on PD Decode for Qwen35B GDN/Mamba hybrid.',
        blocker_sources=sources,runs=outcomes,
        backfill=dict(source=str(backfill_path),observed_snapshot_timestamp=state['timestamp'],nodes=backfill),
        limitations=['Only completed colocated arms independently audited; original whole-run Guard postchecks remain failed.',
            'Short AgentX-derived no-subagent windows, not the full official benchmark or DAG.',
            'EP2 and PP2 baseline arms ran on different nodes; no direct causal ranking.',
            'GPU cache hits were observed; host hits were zero, so CPU loadback was not exercised.',
            'No performance conclusion about PD or dynamic allocation can be drawn from a startup compatibility failure.'])
    (ROOT/'final_findings.json').write_text(json.dumps(findings,indent=2)+'\n')
    print(json.dumps({k:findings[k] for k in ('status','valid_colocated_performance_points','formal_requests','later_turn_requests','mean_later_turn_gpu_cached_tokens','pd_performance_points')}))

if __name__=='__main__':main()
