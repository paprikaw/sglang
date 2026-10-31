"""Freeze a fresh run from the accepted native-FP8 experiment harness."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'h100_qwen122b_latest_decode_20261027/runs/pp2_lookahead_r6'


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def desc(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--cases', nargs='+', default=['b32_repeated'])
    ap.add_argument('--policies', nargs='+', type=int, default=[9, 10])
    a = ap.parse_args()
    run = a.run.resolve()
    assert not run.exists(), 'Use a fresh immutable run directory'
    run.mkdir(parents=True)
    for name in ('cache', 'task', 'pp2'):
        (run / name).mkdir()
    shutil.copytree(BASE / 'code', run / 'code', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(BASE / 'cache/node.json', run / 'cache/node.json')
    shutil.copyfile(ROOT / 'source/gap_schedule.py', run / 'code/gap_schedule.py')
    shutil.copyfile(ROOT / 'source/refine_benchmark.py', run / 'code/refine_benchmark.py')
    text = (run / 'code/runner.py').read_text()
    marker = "if __name__ == '__main__':"
    assert text.count(marker) == 1
    (run / 'code/runner.py').write_text(text.replace(marker, 'import refine_benchmark\nrefine_benchmark.install(sys.modules[__name__])\n\n' + marker))
    def read_rebound(path):
        return json.loads(path.read_text().replace(str(BASE), str(run)))
    cfg = read_rebound(BASE / 'config.json')
    cfg.update(id=run.name, variant='pp_refine', port=27821, dist_port=27822,
               gap_policies=a.policies, samples_per_policy=5, repeats=5,
               profile_steps=128, selected_forwards=32, drop_initial_graphs=40)
    cfg['workloads'] = [c for c in cfg['workloads'] if c['id'] in a.cases]
    assert {c['id'] for c in cfg['workloads']} == set(a.cases)
    cfg['expected_points'] = len(cfg['workloads']) * len(a.policies)
    cfg['protocol'] += ' PP refinement: policy9 control versus tail-local retirement; counterbalanced paired cohorts, unchanged Graph compute; aligned generated-position profiles.'
    cfg['model_cache']['node_specs'][cfg['node']] = desc(run / 'cache/node.json')
    cfg['layouts'][0]['usage_logs'][0]['id'] = run.name + '.pp2'
    cfg['model_cache']['usage_logs'][cfg['node']] = cfg['layouts'][0]['usage_logs']
    Path(cfg['admin_key_file']).write_text(secrets.token_hex(32) + '\n')
    Path(cfg['admin_key_file']).chmod(0o600)
    for field, value in [('policy_path', b'0'), ('diagnostic_eager_policy_path', b'0'),
                         ('fragment_policy_path', b'0'), ('gap_policy_path', b'0'),
                         ('fixed_cohort_control', b'\x00'), ('fixed_cohort_size', bytes([32])),
                         ('transfer_audit_control', b'0')]:
        Path(cfg[field]).write_bytes(value)
        Path(cfg[field]).chmod(0o600)
    (run / 'decode_overlap_verify.bin').write_bytes(b'0')
    save(run / 'config.json', cfg)
    qualification = json.loads((ROOT / 'source_cpu_qualification.json').read_text())
    assert qualification['passed']
    for source in qualification['frozen_sources']:
        assert desc(Path(source['path'])) == source
    source_commit = subprocess.check_output(['git', '-C', str(ROOT / 'source'), 'rev-parse', 'HEAD'], text=True).strip()
    assert source_commit == qualification['source_commit']
    assert not subprocess.check_output(['git', '-C', str(ROOT / 'source'), 'status', '--porcelain'], text=True).strip()
    save(run / 'cpu_qualification.json', qualification)
    spec = read_rebound(BASE / 'dashboard_spec.json')
    spec.update(id=run.name, title='H100 122B PP2 tail-local feedback refinement', port=27820,
                model_cache=cfg['model_cache'])
    spec['tasks'][0].update(expected=cfg['expected_points'], label='PP9 control vs PP10 tail-local feedback')
    inputs = [d for d in spec['frozen_inputs'] if not Path(d['path']).is_relative_to(run)]
    inputs += [desc(p) for p in sorted((run / 'code').glob('*.py'))]
    inputs += [desc(run / 'config.json'), desc(run / 'cache/node.json'), desc(run / 'cpu_qualification.json')]
    inputs += qualification['frozen_sources'] + [desc(ROOT / 'prepare.py')]
    spec['frozen_inputs'] = list({d['path']: d for d in inputs}.values())
    spec['sources'].append(dict(worktree=str(ROOT / 'source'), commit=source_commit,
                               tag=qualification['source_tag']))
    save(run / 'dashboard_spec.json', spec)
    manifest = read_rebound(BASE / 'launch_manifest.json')
    manifest.update(deployment_id=run.name, experiment_id=run.name, protocol=cfg['protocol'],
                    private_adapters=[desc(p) for p in sorted((run / 'code').glob('*.py'))],
                    scheduling_only=True, unchanged_compute_kernels=True, gap_policies=a.policies,
                    source_bundle_commit=source_commit,
                    source_bundle_tag=qualification['source_tag'])
    save(run / 'launch_manifest.json', manifest)
    save(run / 'task/progress.json', dict(experiment_id=run.name, task_id='comparison', stage='queued',
                                        valid_points=0, expected=cfg['expected_points']))
    for p in (run / 'code').glob('*.py'):
        p.chmod(0o600)
    print(json.dumps(dict(run=str(run), policies=a.policies, cases=a.cases)))


if __name__ == '__main__':
    main()
