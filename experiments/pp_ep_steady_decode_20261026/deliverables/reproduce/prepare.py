"""Freeze new exact-node runs; reuse accepted code and required CPU banks."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
A100 = ROOT.parent / 'ep2_pp2_qwen35b_a100_20261025'
H100 = ROOT.parent / 'ep2_pp2_qwen35b_h100_20261025/runs/h100_qwen35b_ep2_pp2_r3'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_prepared(path, value):
    """Update only a new preparation before its final input hashes are frozen."""
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def build(run, node, hardware):
    prep = load_module('a100_preparation', A100 / 'prepare.py')
    result = prep.build(run, node)
    run = Path(run).absolute()
    code = run / 'code'
    cfg = json.loads((run / 'config.json').read_text())
    spec = json.loads((run / 'dashboard_spec.json').read_text())
    cfg.update(hardware=hardware, port=20621 if hardware == 'A100' else 20821,
        dist_port=20622 if hardware == 'A100' else 20822,
        profile_repeats=3, profile_steps=64, selected_forwards=24,
        diagnostic_eager_repeats=3, requests=8, prefix_tokens=4096,
        repeats=3, output_tokens=128,
        reference_run=str(A100 / 'runs/a100_qwen35b_ep2_pp2_r2' if hardware == 'A100' else H100),
        protocol='Native Decode formal timing plus three independent longer native profiles; '
            'H100 also has separately labelled eager diagnostic module scopes. '
            'Same unique prompts across hardware; exact real 4096-token GPU cache; no CPU KV restore.')
    if hardware == 'H100':
        previous = json.loads((H100 / 'config.json').read_text())
        for key in ['launch_env', 'deepep_native', 'deepep_wrapper', 'attention_backend', 'precision', 'graph']:
            cfg[key] = previous[key]
        cfg['overlap_schedule'] = 'native EP overlap; PP overlap unsupported'
        cfg['policy_path'] = str(run / 'pp_ready_policy.bin')
        cfg['diagnostic_eager_policy_path'] = str(run / 'eager_policy.bin')
        for key in ['policy_path', 'diagnostic_eager_policy_path']:
            Path(cfg[key]).write_bytes(b'0')
            os.chmod(cfg[key], 0o600)
        for source in (H100 / 'code').glob('*.py'):
            destination = 'hopper_reference.py' if source.name == 'runner.py' else source.name
            if source.name not in ['runtime_adapter.py']:
                shutil.copyfile(source, code / destination)
        shutil.copyfile(H100 / 'code/runtime_adapter.py', code / 'h100_runtime_adapter.py')
    # Reinstall the current accepted A100 controller after the reference copy.
    runner = (A100 / 'runner.py').read_text()
    marker = "if __name__ == '__main__':"
    assert runner.count(marker) == 1
    runner = runner.replace(marker,
        'import sampling\nsampling.install(sys.modules[__name__])\n\n\n' + marker)
    runner = runner.replace("expected=2 * len(cfg['layouts'])", "expected=len(cfg['layouts'])")
    (code / 'runner.py').write_text(runner)
    shutil.copyfile(A100 / 'runtime_adapter.py', code / 'a100_runtime_adapter.py')
    for name in ['sampling.py', 'diagnostic_switch.py', 'runtime_adapter.py']:
        shutil.copyfile(ROOT / name, code / name)
    for source in code.glob('*.py'):
        os.chmod(source, 0o600)
    write_prepared(run / 'config.json', cfg)
    # Freeze every actual imported adapter and native; keep the clean engine tag.
    inputs = [r for r in spec['frozen_inputs']
        if not Path(r['path']).is_relative_to(code) and r['path'] != str(run / 'config.json')]
    inputs += [prep.old.descriptor(p) for p in code.glob('*.py')]
    inputs += [prep.old.descriptor(run / 'config.json'), prep.old.descriptor(Path(cfg['deepep_native']))]
    inputs += [prep.old.descriptor(p) for p in (Path(cfg['deepep_wrapper']) / 'deep_ep').rglob('*.py')]
    spec.update(title=f'{hardware} Qwen35B EP2/PP2 steady Decode components',
        port=20620 if hardware == 'A100' else 20820,
        frozen_inputs=list({r['path']: r for r in inputs}.values()))
    spec['tasks'][0].update(label=f'{hardware}: EP2 then PP2 on GPUs0/1', expected=2)
    prep.old.validate_contract(spec)
    write_prepared(run / 'dashboard_spec.json', spec)
    write_prepared(run / 'task/progress.json', dict(experiment_id=run.name,
        task_id='comparison', stage='queued', valid_points=0, expected=2))
    prep.old.write_json(run / 'launch_manifest.json', dict(deployment_id=run.name,
        source_tag=prep.old.TAG, source_commit=cfg['engine_commit'],
        private_adapters=[prep.old.descriptor(p) for p in code.glob('*.py')],
        hardware=hardware, node=node, job_id=cfg['job_id'],
        actual_gpu_pair=[0, 1], node_lease_gpus=4, sequential_arms=True,
        required_weight_cache=True, model_path=cfg['model_path'], model_revision=cfg['model_revision'],
        diagnostic_eager_is_not_performance=True))
    result['command'] = [cfg['python'], str(Path(cfg['guard_scripts']) / 'dashboard_guard.py'),
        'run', '--spec', str(run / 'dashboard_spec.json'), '--resource-choice', 'auto',
        '--exit-after-run', '--', cfg['python'], str(code / 'runner.py'), '--run', str(run)]
    write_prepared(run / 'preparation.json', dict(command=result['command'],
        required_environment=prep.old.shared_environment(), node=node, job_id=cfg['job_id']))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--hardware', choices=['A100', 'H100'], required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run, args.node, args.hardware), indent=2))
