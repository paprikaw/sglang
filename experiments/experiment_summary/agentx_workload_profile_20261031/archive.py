"""Archive the CPU profile and the separately labeled 2P6D pilot topic."""
import csv
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
STORE=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')
REGISTRY=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
PILOT=HERE.parent/'agentx_2p6d_20261031'
RUN=WORK/'experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_ep4ep2_vs_3ep2_e8_v4_gdn256_20261031_pilot'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def copy_file(source,dest):
    if source.suffix=='.md':
        def link(match):
            target=match.group(1)
            if target.startswith(('/', '#')) or re.match(r'\w+://',target):
                return match.group(0)
            return ']('+str((source.parent/target).resolve())+')'
        dest.write_text(re.sub(r'\]\(([^)]+)\)',link,source.read_text()))
    else:
        shutil.copyfile(source,dest)
    return dict(source=str(source),source_sha256=sha(source),snapshot=dest.name,sha256=sha(dest))


def registry_entry(identifier, body):
    start,end=f'<!-- begin {identifier} -->',f'<!-- end {identifier} -->'
    entry=f'{start}\n### {identifier}\n\n{body}\n\n{end}'
    with REGISTRY.open('r+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        text=stream.read()
        if start in text:
            left,rest=text.split(start,1)
            _,right=rest.split(end,1)
            text=left+entry+right
        else:
            text+='\n\n'+entry+'\n'
        stream.seek(0);stream.write(text);stream.truncate()


def main():
    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    profile=json.loads((HERE/'profile_manifest.json').read_text())
    assert profile['analyzer']['sha256']==sha(HERE/'analyze.py')
    identifier=profile['id'];dest=STORE/identifier;dest.mkdir(parents=True,exist_ok=True)
    files=[]
    for source in [HERE/'REPORT.md',*sorted(HERE.glob('*.csv')),HERE/'profile_manifest.json',HERE/'analyze.py',Path(__file__).resolve()]:
        files.append(copy_file(source,dest/('summary.md' if source.name=='REPORT.md' else source.name)))
    shutil.copyfile(HERE/'summary.csv',dest/'full_metrics.csv')
    manifest=dict(id=identifier,title='AgentX C8/C128 request and conversation profiles with P-side cache accounting',
        created_at=timestamp,status='complete_cpu_analysis_no_gpu_launch',
        implementation='Saved E5/V3 colocated and E7/V3 PD frozen episodes; cohorts kept separate',
        model='Qwen3.6-35B-A3B BF16',hardware='Historical two-node eight-H100 runs',
        workload=profile,artifacts=files,key_results='C8 prefix hit91.46%; C128 EP4/PP2/single84.05/88.62/52.68%; historical PD P-side43.30%.',
        limitations='Three repeats of selected eight source families, cloned at C128; V3 branch timing, synthetic tokens and finite compressed replay remain.')
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    registry_entry(identifier,f'Snapshot: {dest}\n\nOffline reanalysis of21 accepted episodes/34371 requests. Per-request and per-conversation means and P10-P99 distributions for context, new Prefill, Decode, cache locations and latency. PD uses P-side prefill_meta. V3 and engine cohorts remain labeled. Source hashes and reproducible script included.')
    identifier='qwen35b_h100_agentx_2p6d_gdn256_20261031';dest=STORE/identifier;dest.mkdir(parents=True,exist_ok=True)
    files=[copy_file(PILOT/n,dest/('summary.md' if n=='REPORT.md' else n)) for n in ('REPORT.md','STATUS.md')]
    for name in ('full_metrics.csv','slot_allocations.csv','runtime_markers.csv','instance_metrics.csv','forward_batches.csv','pilot_analysis.json','analyze_pilot.py','checkpoint_lookup.json','diagnostic_snapshot.json','failure_diagnostics.json','placement_audit.json','initial_placements.csv','placement_audit.py','continue_controls.py','continuation.json','backfill_reconciliation.json','current_control_observation.json','C8_COLOCATED_COMPARISON.md','c8_comparison.csv','c8_comparison_evidence.json','compare_colocated_c8.py'):
        p=PILOT/name
        if p.exists():
            files.append(copy_file(p,dest/name))
    for name in ('SINGLE_VS_PD_DIAGNOSIS.md','single_pd_decomposition.csv','single_pd_diagnosis.json','analyze_single_pd.py'):
        p=PILOT/name
        if p.exists():
            files.append(copy_file(p,dest/name))
    analysis=json.loads((PILOT/'pilot_analysis.json').read_text())
    progress=analysis['progress']
    launch=json.loads((RUN/'launch_manifest.json').read_text())
    for name in ('config.json','launch_manifest.json','preflight.json'):
        files.append(copy_file(RUN/name,dest/name))
    for source in analysis['source_runs']:
        run=Path(source['run']);run_dest=dest/'source_runs'/run.name
        run_dest.mkdir(parents=True,exist_ok=True)
        for name in ('config.json','launch_manifest.json','preflight.json','task/progress.json','task/cleanup.json'):
            p=run/name
            if p.exists():
                entry=copy_file(p,run_dest/Path(name).name)
                entry['snapshot']=str((run_dest/Path(name).name).relative_to(dest))
                files.append(entry)
    manifest=dict(id=identifier,title='AgentX2P6D PP2 P with EP4+EP2 or3 EP2 D, GDN256 pilot',
        created_at=timestamp,status=progress['stage'],valid_points=analysis['accepted_primary_episodes'],expected_points=4,
        diagnostic_complete_points=analysis['accepted_diagnostic_episodes'],
        deployment=launch,raw_run=str(RUN),progress=str(RUN/'task/progress.json'),
        source_runs=analysis['source_runs'],partials=analysis['partials'],
        validation=str(RUN/'evidence'),cleanup=str(RUN/'task/cleanup.json'),
        artifacts=files,key_results='See current status; no claim of PD/dynamic gain before matched controls.',
        limitations='One pilot per point; partial failed episodes are not promoted. Historical E5/V3 and E7/V3 are separate from E8/V4. Same-node and cross-node handoff coexist.')
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    registry_entry(identifier,f'Snapshot: {dest}\n\nFixed2P6D pilot:PP2 P + EP4/EP2 D versus PP2 P +3 EP2 D; C8/C128 one episode each. Nodes011/013 eight H100; E8 enginefa52c3cd fixes sparse Hybrid MHA PP KV mapping on top of E7;28 CPU checks passed. Corrected V4 branch timing, HiCache32 decimal GB/rank. GDN256/rank explicitly charged before native KV sizing. Initial equal-instance EP4+EP2 attempt is diagnostic; corrected per-owner placement has6 CPU checks, with uniform3xEP2/P routing unchanged. Current stage:{progress["stage"]}; accepted primary points:{analysis["accepted_primary_episodes"]}/4; diagnostic complete points:{analysis["accepted_diagnostic_episodes"]}. Detailed status, failed partial episodes and per-run cleanup evidence are in summary.md and STATUS.md.')
    print(json.dumps(dict(profile_snapshot=str(STORE/profile['id']),pilot_snapshot=str(dest),progress=progress)))


if __name__=='__main__':
    main()
