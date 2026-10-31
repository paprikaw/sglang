"""Save the offline audit as a small experiment-manager snapshot."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
SOURCE = WORK/'experiments/agentx_pd_dynamic_20261030'
ID = 'qwen35b_h100_agentx_data_audit_20261031'
DEST = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')/ID
REGISTRY = Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    audit = json.loads((HERE/'audit.json').read_text())
    assert audit['selected_accepted_episodes'] == 21 and audit['selected_requests'] == 34371
    assert audit['selected_completion_integrity'] == 'passed'
    assert sha(HERE/'recompute_audit.py') == audit['analyzer_sha256']
    DEST.mkdir(parents=True, exist_ok=True)
    own_files = ['REPORT.md','audit.json','accepted_means.csv','accepted_episodes.csv',
                 'output_agreement.csv','latency_scope_slo.csv','run_inventory.csv',
                 'formal_inventory.csv','historical_episodes.csv','historical_means.csv',
                 'recompute_audit.py','archive_audit.py']
    copied = []
    for name in own_files:
        source = HERE/name
        shutil.copyfile(source, DEST/('summary.md' if name == 'REPORT.md' else name))
        copied.append(dict(source=str(source), sha256=sha(source),
                           snapshot='summary.md' if name == 'REPORT.md' else name))
    shutil.copyfile(HERE/'accepted_means.csv', DEST/'full_metrics.csv')
    evidence = [SOURCE/name for name in ['PROTOCOL_AUDIT.md','TIMING_AUDIT.md',
        'WORKLOAD_COVERAGE.md','POLICY_AUDIT.md','CACHE_ROUTING_DIAGNOSIS.md',
        'e7_v4_b7200_analysis/audit.json','wait_for_e7_v4_controls_r1.json',
        'runs/eight_gpu_ep_r6/analysis/TRADEOFFS.md','runs/eight_gpu_ep_r6/analysis/EXPOSURE.md']]
    early = WORK/'experiments/agentx_pd_feasibility_20261030'
    evidence += [early/'final_findings.json',
                 early/'runs/011_ep2_hicache_r2/analysis_audit.json',
                 early/'runs/171_pp2_hicache_r2/analysis_audit.json',
                 WORK/'experiments/experiment_summary/agentx_pd_feasibility_20261030/REPORT.md']
    original_evidence = [dict(path=str(p), sha256=sha(p)) for p in evidence]
    with (HERE/'accepted_episodes.csv').open() as stream:
        runs = sorted({r['run'] for r in csv.DictReader(stream)})
    deployments = []
    for name in runs:
        run = SOURCE/'runs'/name
        m = json.loads((run/'launch_manifest.json').read_text())
        deployments.append(dict(id=name, engine='sglang', worktree=m['engine_root'],
            exact_commit=m['engine_commit'], immutable_source_tag=m['engine_tag'],
            model_revision=m['model_revision'], nodes=m['nodes'], arms=m['arms'],
            historical_launch_manifest=str(run/'launch_manifest.json'),
            config=str(run/'config.json'),
            dirty_state='Historical frozen manifest and inputs retained; current worktree state not substituted',
            new_deployment=False))
    manifest = dict(id=ID, title='AgentX saved-data inventory and benchmark protocol audit',
        created_at=audit['updated_at'], status='complete', kind='offline_artifact_audit',
        source_root=str(HERE), report=str(HERE/'REPORT.md'), new_gpu_runs=0,
        verification='21 completed episodes rechecked by request identity, shape, completion, metric recomputation and import SHA',
        selected_requests=34371, selected_episodes=21,
        separate_historical_points=audit['separate_historical_points'],
        separate_historical_episodes=audit['separate_historical_episodes'],
        historical_deployments=deployments,
        model=dict(name='Qwen3.6-35B-A3B', weights='BF16', kv='BF16', recurrent_state='FP32'),
        testbed=dict(gpus=8, gpu='H100 80GB', nodes=['spartan-gpgpu011','spartan-gpgpu171']),
        workload=dict(source='AgentX Weka 256k hashed traces', source_families=8,
            calls_per_copy=171, isolated_copies_c8=1, isolated_copies_c128=16,
            policy='whole-family filtered finite replay, fixed affinity, V3 post branches',
            idle_compression='global idle only, cap 10 seconds', official_score=False),
        cache=dict(hicache=True, host_decimal_gb_per_rank=32, gpu_static_fraction=.95,
            reset='between independent episodes; retained within formal sessions',
            caveat='different native KV capacities and explicit recurrent state budgets'),
        key_results='E5/V3 C8 single best mean request E2E; C128 PP2 best. E7/V3 EP4 P/D completed independently; PD and dynamic gains unproven.',
        limitations=['V3 branch offset/overlap differs from official harness; V4 no GPU episodes',
            'timed output differences not causally diagnosed', 'synthetic tokens, eight families and clones',
            'finite compressed trace throughput is not steady service capacity',
            'engine/cache/capacity/routing differences require controlled attribution',
            'HTTP-start means and ready-start SLO must be labeled separately'],
        copied_artifacts=copied, referenced_evidence=original_evidence,
        raw_policy='Original large logs and request outputs stay in source directories; no weights, KV, or auth files copied')
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    start, end = f'<!-- begin {ID} -->', f'<!-- end {ID} -->'
    entry = f'''{start}
### {ID}

Snapshot: {DEST}

- Completed offline inventory/audit, no new GPU launch:21 accepted E5/V3 and E7/V3 episodes,34371 request rows rechecked;74 main run directories,46 formal artifacts,42 unique content hashes. Four imported copies are not independent repeats.
- E5/V3:6 colocated points x3 repeats; E7/V3:one EP4 4P4D C128 point x3 repeats, kept separate. Seven other historical points comprise21 unique episodes, including the earlier same-source EP4 colocated-vs-PD C2/C8 comparison. Earlier480-call feasibility fragments stay separate. V4 GPU results remain0.
- Documents exact replay, synthetic hash prompts, filtered/cloned workload, branch offsets, global idle compression, output disagreement, engine/capacity differences, cache-owner placement, and HTTP-vs-ready timing/SLO. Does not establish PD or dynamic-layout gains.

Artifacts: summary.md,manifest.json,full_metrics.csv,accepted_episodes.csv,historical_means.csv,historical_episodes.csv,run_inventory.csv,formal_inventory.csv,output_agreement.csv,latency_scope_slo.csv,audit.json and reproducible CPU-only scripts. Original logs stay at recorded paths.

{end}'''
    text = REGISTRY.read_text()
    if start in text:
        left, rest = text.split(start,1)
        _, right = rest.split(end,1)
        text = left+entry+right
    else:
        text += '\n\n'+entry+'\n'
    REGISTRY.write_text(text)
    print(json.dumps(dict(snapshot=str(DEST), files=len(copied)+2, selected_episodes=21, gpu_launches=0)))


if __name__ == '__main__':
    main()
