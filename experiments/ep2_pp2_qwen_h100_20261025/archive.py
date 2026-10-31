"""Archive only validated, small artifacts; never weights, KV, or credentials."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def archive(run, destination):
    evidence = json.loads((run / 'analysis/validated_metrics.json').read_text())
    assert evidence['passed'] and len(evidence['rows']) == 14
    cfg = json.loads((run / 'config.json').read_text())
    post = json.loads(Path(evidence['guard_postcheck']).read_text())
    assert post['passed'] and post['scientific_success'] and post['cleanup_verified']
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    def copy(source, target):
        target = destination / target
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(source, target)
        copied.append(target)
    for name in ('full_metrics.csv', 'validated_metrics.json', 'prefill_comparison.png',
                 'prefill_comparison.pdf', 'decode_comparison.png', 'decode_comparison.pdf',
                 'decode_graph_summary.csv', 'decode_graph_inventory.json', 'GRAPH_PROFILE.md'):
        copy(run / 'analysis' / name, name)
    text = (run / 'analysis/REPORT.md').read_text()
    text = text.replace('](../../../README.md)', '](protocol.md)')
    text = text.replace('](../config.json)', '](' + str(run / 'config.json') + ')')
    text = text.replace('](../capacity_matrix.json)', '](capacity_matrix.json)')
    summary = destination / 'summary.md'
    summary.write_text(text)
    copied.append(summary)
    copy(Path(__file__).with_name('README.md'), 'protocol.md')
    copy(run.parents[2] / 'experiment_summary/component_reports/Q03.md', 'overview.md')
    for name in ('devices.csv', 'topology.txt', 'capacity_matrix.json', 'worker_cleanup.json'):
        copy(run / name, name)
    for arm in ('ep2', 'pp2'):
        for name in ('tuning.json', 'cache_residency_qualification.json'):
            copy(run / arm / name, arm + '_' + name)
    for name in ('runner.py', 'protocol.py', 'runtime_adapter.py', 'pp_ready_send.py',
                 'base_runner.py', 'process_helpers.py', 'runtime_setup.py'):
        copy(run / 'code' / name, 'code/' + name)
    for name in ('analyze.py', 'plot.py', 'archive.py', 'trace_analysis.py', 'publish_summary.py', 'test_protocol.py'):
        copy(Path(__file__).with_name(name), 'code/' + name)
    for name in ('native_cache_cpu_tests.log', 'protocol_cpu_tests.log'):
        copy(Path(__file__).with_name(name), name)
    engine = cfg['engine_root']
    commit = subprocess.check_output(['git', '-C', engine, 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == cfg['engine_commit']
    tags = subprocess.check_output(['git', '-C', engine, 'tag', '--points-at', 'HEAD'], text=True).splitlines()
    manifest = dict(id=destination.name, title='H100 Qwen3.5-122B: EP2 versus two-slot PP2',
        created_at=datetime.now(timezone.utc).isoformat(), status='complete',
        implementation=dict(engine='SGLang', worktree=engine, commit=commit, immutable_tags=tags,
            private_runtime_adapter=str(run / 'code/runtime_adapter.py')),
        deployment=dict(id=cfg['id'], node=cfg['node'], slurm_job=cfg['job_id'],
            rank_mapping={r['id']:r['gpu_groups'] for r in cfg['layouts']}, retired=True),
        model=dict(path=cfg['model_path'], revision=cfg['model_revision'], precision=cfg['precision']),
        workload=json.loads((run / 'completed.json').read_text()),
        parameters=dict(layouts=[{k:r[k] for k in ('id','tp','dp','ep','pp','gpu_groups')}
            for r in cfg['layouts']], graph=True, max_running_requests=16,
            mamba_states_global=dict(ep2=88, pp2=84), mamba_max_states_per_path=1,
            static_memory_fraction=.94,
            hicache=False, mtp=False, preparation='real GPU prefixes; bounded untimed preparation',
            formal_request_arrivals='entire cohort concurrently; no preparation limit in timed traffic'),
        communication=dict(wrapper=cfg['deepep_wrapper'], native=cfg['deepep_native'],
            native_sha256=hashlib.sha256(Path(cfg['deepep_native']).read_bytes()).hexdigest()),
        validation=dict(guard_postcheck=evidence['guard_postcheck'], cache=post['model_cache']['passed'],
            cleanup=True, actual_graph=True, raw_metrics_recomputed=True),
        artifacts=dict(original_run=str(run), summary='summary.md', overview='overview.md',
            protocol='protocol.md', metrics='full_metrics.csv',
            profiles=str(run / '{ep2,pp2}/traces'), private_config_sha256=hashlib.sha256(
                (run / 'config.json').read_bytes()).hexdigest()),
        key_results=evidence['rows'], limitations=['Three rounds, descriptive percentiles only',
            'Parallel disjoint GPU pairs share CPU and node resources',
            'PP and EP have different supported overlap scheduling',
            'H100 W8A8 versus historical A100 W8A16 is not a pure GPU comparison',
            'PP-only own-capacity row has different context from matched EP workloads'])
    path = destination / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    copied.append(path)
    checksums = {str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest() for p in copied}
    (destination / 'checksums.json').write_text(json.dumps(checksums, indent=2) + '\n')
    for name, sha in checksums.items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == sha
    print(json.dumps(dict(archived=True, files=len(copied), destination=str(destination))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    archive(args.run.resolve(), args.destination.resolve())
