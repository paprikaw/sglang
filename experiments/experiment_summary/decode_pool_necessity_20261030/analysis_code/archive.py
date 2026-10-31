"""Archive public summaries, provenance and code; keep credentials/logs in place."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[1]
EXPERIMENT_ID = 'qwen35b_h100_decode_pool_necessity_20261029'


def read(path):
    return json.loads(path.read_text())


def desc(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    human = args.human.resolve(); snapshot = args.snapshot.resolve()
    if snapshot.exists():
        assert read(snapshot / 'manifest.json')['id'] == EXPERIMENT_ID
    decision = read(human / 'decision.json')
    arm_audit = read(human / 'accepted_arm_audit.json')
    run_manifests = []
    for name in decision['accepted_runs']:
        run = ROOT / 'runs' / name
        cfg = read(run / 'config.json')
        target = human / 'run_manifests' / name
        target.mkdir(parents=True, exist_ok=True)
        for rel in ['config.json', 'launch_manifest.json', 'dashboard_spec.json', 'topology.txt',
                    'kv_checkpoint_discovery.json', 'task/cleanup.json', 'task/progress.json']:
            source = run / rel
            dest = target / rel; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
        for source in (run / 'dashboard_guard/checks').glob('*-post*.json'):
            shutil.copyfile(source, target / source.name)
        code = target / 'code'; code.mkdir(exist_ok=True)
        for source in (run / 'code').glob('*.py'):
            shutil.copyfile(source, code / source.name)
        run_manifests.append(dict(id=name, original_run=str(run),
                                  launch_manifest=desc(run / 'launch_manifest.json'),
                                  config=desc(run / 'config.json'), node=cfg['node'], job=cfg['job_id'],
                                  repeats=cfg['repeats'], layouts=cfg['layouts'], profiles=cfg['profiles'],
                                  whole_run_postpassed=next(a['whole_run_guard_postpassed'] for a in arm_audit['runs'] if Path(a['run']).name == name)))
    shutil.copyfile(ROOT / 'PROTOCOL.md', human / 'PROTOCOL.md')
    code = human / 'analysis_code'; code.mkdir(exist_ok=True)
    for source in ROOT.glob('*.py'):
        shutil.copyfile(source, code / source.name)
    state_path = WS / 'experiments/cluster_backfill_20261020/state.json'
    state = read(state_path)
    nodes = {n: {k: state['nodes'][n].get(k) for k in ('state', 'reason')}
             for n in ('spartan-gpgpu011', 'spartan-gpgpu171')}
    assert all(v['state'] == 'benchmark' and v['reason'] == 'verified_live_worker' for v in nodes.values())
    recovery = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),
                    coordinator_source=str(state_path), coordinator_timestamp=state['timestamp'], nodes=nodes,
                    foreground_cleanup_verified=all(a['cleanup']['passed'] for a in arm_audit['runs']),
                    policy='Original background coordinator rediscovered and resumed verified benchmark workers; no CPU weight eviction or new Slurm allocation.')
    save(human / 'RESOURCE_RECOVERY.json', recovery)
    first = read(ROOT / 'runs/011_screen_r3/config.json')
    source_manifest = read(ROOT / 'runs/011_screen_r3/launch_manifest.json')
    branch = subprocess.run(['git', '-C', first['engine_root'], 'symbolic-ref', '-q', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip() or 'detached HEAD'
    manifest = dict(id=EXPERIMENT_ID, title='Qwen35B single-node Decode PP/EP pool necessity audit',
                    created_at=datetime.now(timezone.utc).isoformat(), report_date_local='2026-10-30',
                    status='complete_with_documented_partial_run',
                    implementation=dict(engine='sglang', worktree=first['engine_root'], branch=branch,
                                        commit=first['engine_commit'], source_tag=first['engine_tag'],
                                        dirty_at_freeze=False, runtime_adapter_versions='run_manifests/*/code; frozen individually',
                                        cpu_weight_daemon=source_manifest['cpu_weight_daemon_source']),
                    model=dict(path=first['model_path'], weight_dtype='BF16', kv_dtype='BF16', gdn_dtype='FP32'),
                    testbed=dict(nodes=['spartan-gpgpu011', 'spartan-gpgpu171'], gpu='H100 80GB HBM3',
                                 gpus_per_candidate=4, cross_node_layouts=False,
                                 slurm_jobs=['33758541', '33646189']),
                    deployment=dict(type='temporary benchmark servers', gateway_modified=False,
                                    retained_serving_endpoint=None, rollback_endpoint=None,
                                    final_state='foreground servers stopped; original background benchmarks recovered'),
                    workload=dict(source='Deterministic model-tokenized repeated and eight-topic text; source in each frozen verify_workload.py',
                                  cases=first['workloads'], Q=1, output_tokens=128,
                                  cache='real model-produced GPU-hot prefix; no CPU KV checkpoint restore or HiCache',
                                  metric='Per-request TPOT=(last-first)/127; request E2E and TTFT separate; arithmetic means across all complete formal cohorts'),
                    runs=run_manifests, qualification=desc(human / 'accepted_arm_audit.json'),
                    key_results=decision, resource_recovery=desc(human / 'RESOURCE_RECOVERY.json'),
                    artifacts=dict(human_report=str(human / 'REPORT.md'), canonical_snapshot=str(snapshot),
                                   primary_means=str(human / 'primary/mean_metrics.csv'),
                                   full_means=str(human / 'full_metrics.csv'),
                                   formal_cohorts=str(human / 'formal_rounds.csv'),
                                   raw_logs='Original run directories, referenced by path and SHA256; not copied',
                                   credentials='Not copied: admin keys, lease tokens, binding payloads and control files remain in original protected run directories'))
    save(human / 'manifest.json', manifest)
    for document in ('REPORT.md', 'RUN_AUDIT.md'):
        path = human / document
        def absolute_link(match):
            target = match.group(1)
            if target.startswith(('/', '#', 'http:', 'https:')):
                return match.group(0)
            destination = (human / target).resolve()
            assert destination.exists(), str(destination)
            return '](' + str(destination) + ')'
        path.write_text(re.sub(r'\]\(([^)]+)\)', absolute_link, path.read_text()))
    shutil.copyfile(human / 'REPORT.md', human / 'summary.md')
    shutil.copytree(human, snapshot, dirs_exist_ok=True)
    copied = [dict(relative_path=str(p.relative_to(snapshot)), sha256=desc(p)['sha256'])
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name != 'SNAPSHOT_FILES.json']
    save(snapshot / 'SNAPSHOT_FILES.json', dict(id=EXPERIMENT_ID, files=copied))
    print(json.dumps(dict(snapshot=str(snapshot), files=len(copied), recovered_nodes=nodes,
                          measured_points=decision['measured_points'], formal_cohorts=decision['formal_cohorts'])))


if __name__ == '__main__':
    main()
