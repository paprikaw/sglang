"""Archive qualified baselines and explicitly unfinished P/D progress on CPU."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

ROOT = Path(__file__).resolve().parent
ID = 'qwen35b_h100_agentx_pd_dynamic_20261030'
DEST = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')/ID
REGISTRY = Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def absolute_links(text, source):
    def replace(match):
        target = match.group(1)
        if target.startswith(('/', '#')) or '://' in target:
            return match.group(0)
        return ']('+str((source.parent/target).resolve())+')'
    return re.sub(r'\]\(([^)]+)\)', replace, text)


def main():
    DEST.mkdir(exist_ok=True)
    if (DEST/'manifest.json').exists():
        assert read(DEST/'manifest.json')['source_root'] == str(ROOT)
    baseline_names = ['eight_gpu_col_ep4_durable_r3',
        'eight_gpu_col_pp2_durable_m16_r3', 'eight_gpu_col_single_durable_kv40_r3']
    deployments = []
    count = 0
    for name in baseline_names:
        run = ROOT/'runs'/name
        progress, clean = read(run/'task/progress.json'), read(run/'task/cleanup.json')
        posts = sorted((run/'dashboard_guard/checks').glob('*-post.json'))
        assert progress['stage'] == 'complete' and clean['passed'] and clean['gpu_idle']
        assert posts and read(posts[-1])['passed']
        cfg, m = read(run/'config.json'), read(run/'launch_manifest.json')
        markers = sorted(run.glob('results/*/*/*/completed.r*.json'))
        assert len(markers) == cfg['expected_points'] == 6
        count += len(markers)
        deployments.append(dict(id=name, engine='sglang', engine_root=m['engine_root'],
            engine_commit=m['engine_commit'], source_tag=m['engine_tag'],
            model_revision=m['model_revision'], nodes=m['nodes'], arms=m['arms'],
            original_launch_manifest=str(run/'launch_manifest.json'),
            original_config=str(run/'config.json'),
            qualification_status='accepted', completed_episodes=len(markers),
            cleanup=dict(path=str(run/'task/cleanup.json'), sha256=sha(run/'task/cleanup.json')),
            postcheck=dict(path=str(posts[-1]), sha256=sha(posts[-1]))))
    assert count == 18
    sources = []
    selected = [p for p in (ROOT/'matrix_analysis').iterdir()
                if p.is_file() and p.suffix in {'.md', '.csv', '.json'} and p.stat().st_size < 1_000_000]
    selected += [ROOT/name for name in ('PROTOCOL_AUDIT.md','WORKLOAD_COVERAGE.md',
        'TIMING_AUDIT.md','POLICY_AUDIT.md','DECODE_HICACHE_RESTORE_PRESSURE.md',
        'DECODE_CHECKPOINT_PRESSURE.md','PP_RESTORE_DIAGNOSIS.md','e7_failure_audit.json',
        'E7_T300_RUN_PLAN.json','NEXT_RUNS.md','analyze_pd_waits.py',
        'pd_queue_split_validation.json','EPISODE_TIMEOUT_AUDIT.md',
        'episode_deadline_failure_audit.json','episode_capture_tests.log',
        'E7_T300_B7200_RUN_PLAN.json','e7_b7200_cohort_preflight.json',
        'E7_V4_RUN_PLAN.json','e7_v4_cohort_preflight.json','v4_bundle_tests_native_ast.log',
        'validate_v4_bundle.py','audit_v4_cohort.py','completed_episode_files.py',
        'completion_gate_validation.json','completion_gate_baseline_refresh.json',
        'analyze_forward_batches.py','analyze_cache_owner_pressure.py',
        'CACHE_ROUTING_DIAGNOSIS.md','cache_owner_first_episode_audit.json')]
    selected += [ROOT/'runs/eight_gpu_ep4_queue32_e7_v4_b7200_r1/v4_cpu_validation.json']
    current_audit = ROOT/'e7_v4_b7200_analysis/audit.json'
    pressure_audit = ROOT/'e7_t300_b7200_analysis/audit.json'
    current = read(current_audit) if current_audit.exists() else None
    for audit in [current_audit, pressure_audit]:
        if audit.exists():
            selected += [p for p in audit.parent.iterdir()
                         if p.is_file() and p.suffix in {'.md','.csv','.json'}
                         and p.stat().st_size < 1_000_000]
    selected += [ROOT/'output_agreement_e5/OUTPUT_AGREEMENT.md']
    for source in selected:
        assert source.exists(), source
        target = DEST/source.relative_to(ROOT)
        target.parent.mkdir(exist_ok=True, parents=True)
        if source.suffix == '.md':
            target.write_text(absolute_links(source.read_text(), source))
        else:
            shutil.copyfile(source, target)
        sources.append(dict(original=str(source), original_sha256=sha(source),
            snapshot=str(target.relative_to(DEST)), snapshot_sha256=sha(target)))
    report = ROOT.parent/'experiment_summary/agentx_pd_dynamic_20261030/REPORT.md'
    (DEST/'summary.md').write_text(absolute_links(report.read_text(), report))
    manifest = dict(id=ID, title='AgentX-derived eight-H100 P/D and EP/PP validation',
        created_at=read(DEST/'manifest.json').get('created_at',time.time())
            if (DEST/'manifest.json').exists() else time.time(), updated_at=time.time(),
        status='in_progress', source_root=str(ROOT),
        source_report=str(report), source_report_sha256=sha(report),
        completed_baseline_episodes=18, baseline_deployments=deployments,
        ongoing_plan=str(ROOT/'E7_V4_RUN_PLAN.json'),
        ongoing_status=str(ROOT/'STATUS.json'),
        current_cohort_snapshot=dict(
            audit=str(current_audit), audit_sha256=sha(current_audit),
            engine_commit=current['engine_commit'],
            completed_episodes=sum(x['episodes'] for x in current['runs']),
            accepted_episodes=sum(x['episodes'] for x in current['runs'] if x['accepted']),
            all_runs_accepted=current['all_runs_accepted'],
            warning='Corrected V4 branch timing; complete episodes from unaccepted runs are diagnostic progress only; not merged with E5 or V3 pressure qualification') if current else None,
        pressure_qualification_snapshot=dict(
            audit=str(pressure_audit), audit_sha256=sha(pressure_audit),
            runs=read(pressure_audit)['runs'],
            warning='V3 engine-pressure qualification only; separate from V4 performance cohort') if pressure_audit.exists() else None,
        model=dict(name='Qwen3.6-35B-A3B', weight_dtype='BF16', kv_dtype='BF16', gdn_dtype='FP32'),
        testbed=dict(nodes=['spartan-gpgpu011','spartan-gpgpu171'],gpus=8,gpu='H100 80GB'),
        workload=dict(source='AgentX Weka 256k hashed traces',source_families=8,
            calls_per_source_copy=171,c128_isolated_copies=16,c128_calls=2736,
            replay='Accepted E5 baselines use V3; new matched controls use V4 corrected branch offsets; both use custom compressed finite replay, not official AgentX score'),
        cache=dict(hicache=True,host_pool_decimal_gb_per_rank=32,gpu_static_fraction=.95,
            gpu_kv_capacity='native auto-sized; no artificial max-total-tokens cap',
            flush='untimed warmup / independent repeat boundaries only',affinity='conversation and Attention-DP'),
        key_results='E5 accepted baselines: single-GPU replicas best C8 E2E; PP2 replicas best C128 E2E. EP4 wins some strict SLO cells. P/D and dynamic gains not established.',
        limitations=['different explicit GDN budgets; native KV capacity differs',
            'synthetic hash-token prefix reconstruction, not semantic prompts',
            'V3 idle compression and branch timing differ from official harness',
            'V4 corrects branch rules but retains custom root admission, synthetic tokens and global idle compression',
            'timed outputs are not all token-identical',
            'failed/incomplete P/D attempts excluded from performance means'],
        copied_artifacts=sources,
        raw_artifacts='Large logs, request traces, weights and KV bytes are not copied; original paths retained. No auth files or private lease/binding contents copied.')
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    start, end = f'<!-- begin {ID} -->', f'<!-- end {ID} -->'
    entry=f'''{start}
### {ID}

Snapshot: {DEST}

- In progress: accepted E5 colocated baselines comprise18 full episodes, two existing4xH100 nodes, Qwen3.6-35B-A3B BF16, all roles HiCache ON and native .95 GPU KV auto-sizing.
- C128 request E2E means:2xEP4 22624.015ms;4xPP2 13128.898ms;8xsingle33312.386ms. C8 single is best E2E; some strict SLO cells favor EP4. Different state budgets/capacities retained.
- E7 HiCache capacity retries observed on real GPUs. Earlier attempts hit120-second handoff or2400-second episode bounds, or a shared lease interruption; incomplete samples are excluded. V3 pressure qualification completed3/3 C128 episodes with postcheck and cleanup passed, kept separate from E5 performance means. The former24-episode V4 controls remain prepared with0 GPU episodes; their automatic interleaved queue was cancelled before admission per user steering. Next order is colocated layouts first, then P:D ratios and layout candidates including single-GPU Decode. No claim of P/D or dynamic-allocation gain.
- Eight source families and cache-isolated cloning; synthetic hash-token prefixes and compressed finite replay. V4 corrects pinned branch issue/return offsets but is not the official harness or AgentX score. Failed attempts and output-agreement limitations preserved.

Artifacts: manifest, summary,18-episode metric tables and source audits, protocol/coverage/timing notes, fixes and current run-plan pointers. Large/private originals remain in source paths.

{end}'''
    text = REGISTRY.read_text()
    if start in text:
        assert text.count(start) == text.count(end) == 1
        left, rest = text.split(start,1)
        _, right = rest.split(end,1)
        text = left+entry+right
    else:
        text += '\n\n'+entry+'\n'
    REGISTRY.write_text(text)
    print(json.dumps(dict(snapshot=str(DEST), accepted_baseline_episodes=18,
        copied_artifacts=len(sources), status='in_progress')))


if __name__ == '__main__':
    main()
