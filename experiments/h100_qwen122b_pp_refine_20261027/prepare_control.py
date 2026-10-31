"""Freeze a static PP partition trial or a fresh, unchanged EP reference."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def desc(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--variant', choices=['pp_balance', 'ep_control'], required=True)
    ap.add_argument('--cases', nargs='+', default=['b32_repeated'])
    a = ap.parse_args()
    pp = a.variant == 'pp_balance'
    base = (ROOT / 'runs/pp_feedback_r3' if pp else
            ROOT.parent / 'h100_qwen122b_latest_decode_20261027/runs/ep2_native_r5')
    arm = 'pp2' if pp else 'ep2'
    run = a.run.resolve()
    assert not run.exists(), 'Use a fresh immutable run directory'
    run.mkdir(parents=True)
    for name in ('cache', 'task', arm):
        (run / name).mkdir()
    shutil.copytree(base / 'code', run / 'code', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(base / 'cache/node.json', run / 'cache/node.json')
    shutil.copyfile(base / 'cpu_qualification.json', run / 'cpu_qualification.json')

    def rebound(path):
        return json.loads(path.read_text().replace(str(base), str(run)))

    cfg = rebound(base / 'config.json')
    cfg.update(id=run.name, variant=a.variant, port=27831, dist_port=27832,
               repeats=5, profile_steps=128 if pp else 64,
               drop_initial_graphs=40 if pp else 20, selected_forwards=32 if pp else 16)
    cfg['workloads'] = [c for c in cfg['workloads'] if c['id'] in a.cases]
    assert {c['id'] for c in cfg['workloads']} == set(a.cases)
    cfg['expected_points'] = len(cfg['workloads']) * (2 if pp else 1)
    cfg['model_cache']['node_specs'][cfg['node']] = desc(run / 'cache/node.json')
    cfg['layouts'][0]['usage_logs'][0]['id'] = run.name + '.' + arm
    cfg['model_cache']['usage_logs'][cfg['node']] = cfg['layouts'][0]['usage_logs']
    if pp:
        cfg['launch_env']['SGLANG_PP_LAYER_PARTITION'] = '25,23'
        cfg['layouts'][0]['layer_partition'] = [25, 23]
        cfg['profile_policies'] = [10]
        cfg['protocol'] = cfg['protocol'].replace('PP2 24+24 layers', 'PP2 25+23 layers')
        cfg['protocol'] += ' Static measured-time PP partition: move GDN layer24 from tail to head, six full Attention layers remain on each rank.'
        shutil.copyfile(ROOT / 'balance_source/runtime_adapter.py', run / 'code/runtime_adapter.py')
    else:
        cfg['protocol'] += ' Fresh EP native control for the PP scheduling/partition refinement; no new EP compute or communication change.'
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
    spec = rebound(base / 'dashboard_spec.json')
    spec.update(id=run.name, title='H100 122B ' + a.variant, port=27830,
                model_cache=cfg['model_cache'])
    spec['tasks'][0].update(expected=cfg['expected_points'], label=a.variant)
    inputs = [d for d in spec['frozen_inputs'] if not Path(d['path']).is_relative_to(run)]
    inputs += [desc(p) for p in sorted((run / 'code').glob('*.py'))]
    inputs += [desc(run / name) for name in ('config.json', 'cache/node.json', 'cpu_qualification.json')]
    inputs += [desc(Path(__file__).resolve())]
    if pp:
        source = ROOT / 'balance_source'
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
        spec['sources'].append(dict(worktree=str(source), commit=commit,
                                    tag='experiment/pp-balance/20261027-cpu-r1'))
        inputs += [desc(source / 'runtime_adapter.py')]
        compile((source / 'runtime_adapter.py').read_text(), str(source / 'runtime_adapter.py'), 'exec')
        save(run / 'partition_qualification.json', dict(cpu_passed=True, source_commit=commit,
             partition=[25, 23], total_layers=48, full_attention_layers=[6, 6],
             linear_attention_layers=[19, 17], native_kernels_unchanged=True,
             baseline_stage_graph_ms=[15.717125442504882 / 2, 16.45132420349121 / 2],
             gpu_qualification_required=True))
        inputs += [desc(run / 'partition_qualification.json')]
    spec['frozen_inputs'] = list({d['path']: d for d in inputs}.values())
    save(run / 'dashboard_spec.json', spec)
    manifest = rebound(base / 'launch_manifest.json')
    manifest.update(deployment_id=run.name, experiment_id=run.name, protocol=cfg['protocol'],
                    private_adapters=[desc(p) for p in sorted((run / 'code').glob('*.py'))],
                    pp_layer_partition=[25, 23] if pp else None,
                    unchanged_compute_kernels=True)
    save(run / 'launch_manifest.json', manifest)
    save(run / 'task/progress.json', dict(experiment_id=run.name, task_id='comparison', stage='queued',
                                        valid_points=0, expected=cfg['expected_points']))
    for path in (run / 'code').glob('*.py'):
        path.chmod(0o600)
    print(json.dumps(dict(run=str(run), variant=a.variant, cases=a.cases)))


if __name__ == '__main__':
    main()
