"""Freeze a new run using the reviewed Qwen H100 loader and controller."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def descriptor(path):
    path = Path(path).absolute()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def build(run, mode, arms, repeats=3, validation_only=False):
    old = load('steady_prepare', ROOT.parent / 'pp_ep_steady_decode_20261026/prepare.py')
    old.build(run, 'spartan-gpgpu178', 'H100')
    run = Path(run).absolute()
    cfg = json.loads((run / 'config.json').read_text())
    spec = json.loads((run / 'dashboard_spec.json').read_text())
    code = run / 'code'
    for name in ('fragmentation.py', 'benchmark.py', 'route_gemv.py'):
        shutil.copyfile(ROOT / name, code / name)
    adapter = (code / 'runtime_adapter.py').read_text()
    adapter += '\n_previous_install = install\ndef install():\n    _previous_install()\n    import fragmentation\n    fragmentation.install()\n'
    (code / 'runtime_adapter.py').write_text(adapter)
    runner = (code / 'runner.py').read_text().replace(
        'sampling.install(sys.modules[__name__])', 'sampling.install(sys.modules[__name__])\nimport benchmark\nbenchmark.install(sys.modules[__name__])')
    runner = runner.replace('expected=len(cfg[\'layouts\'])', 'expected=2 * len(cfg[\'layouts\'])')
    runner = runner.replace("text=final.get('text', '')[-200:]", "text=final.get('text', '')")
    runner = runner.replace("output_ids=final.get('output_ids'),", "validation_logits={k:final.get('meta_info', {}).get(k) for k in ('output_token_logprobs','output_top_logprobs')}, output_ids=final.get('output_ids'),")
    (code / 'runner.py').write_text(runner)
    cfg.update(expert_mode=mode, arm_order=arms, port=24221, dist_port=24222,
        validation_only=validation_only,
        repeats=repeats, profile_repeats=1, profile_steps=40, selected_forwards=16,
        diagnostic_eager_repeats=0, fragment_policy_path=str(run / 'fragment_policy.bin'),
        protocol='Matched cold Prefill and GPU-cached Decode. Formal timing separate from routing diagnostics and native profiles.')
    Path(cfg['fragment_policy_path']).write_bytes(b'0')
    os.chmod(cfg['fragment_policy_path'], 0o600)
    for source in code.glob('*.py'):
        os.chmod(source, 0o600)
    cfg['launch_env']['SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED'] = '0'
    cfg['layouts'] = [r for r in cfg['layouts'] if r['id'] in arms]
    uses = [r['usage_logs'][0] for r in cfg['layouts']]
    cfg['model_cache']['usage_logs'][cfg['node']] = uses
    spec['model_cache'] = cfg['model_cache']
    spec['tasks'][0].update(expected=(1 if validation_only else 2) * len(arms), label='H100 Expert fragmentation: ' + mode)
    spec.update(title='H100 full-model PP Expert fragmentation: ' + mode, port=24220)
    write(run / 'config.json', cfg)
    inputs = [r for r in spec['frozen_inputs'] if not Path(r['path']).is_relative_to(code)
              and r['path'] != str(run / 'config.json')]
    inputs += [descriptor(p) for p in sorted(code.glob('*.py'))] + [descriptor(run / 'config.json')]
    spec['frozen_inputs'] = list({r['path']: r for r in inputs}.values())
    write(run / 'dashboard_spec.json', spec)
    manifest = json.loads((run / 'launch_manifest.json').read_text())
    manifest.update(expert_mode=mode, private_adapters=[descriptor(p) for p in sorted(code.glob('*.py'))],
                    no_route_changes=True, no_weight_precision_changes=True)
    write(run / 'launch_manifest.json', manifest)
    prep = json.loads((run / 'preparation.json').read_text())
    prep['command'] = [cfg['python'], str(Path(cfg['guard_scripts']) / 'dashboard_guard.py'),
        'run', '--spec', str(run / 'dashboard_spec.json'), '--resource-choice', 'auto',
        '--exit-after-run', '--', cfg['python'], str(code / 'runner.py'), '--run', str(run)]
    write(run / 'preparation.json', prep)
    return dict(run=str(run), mode=mode, arms=arms, command=prep['command'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--mode', default='native')
    ap.add_argument('--arms', nargs='+', default=['ep2', 'pp2'], choices=['ep2', 'pp2'])
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--validation-only', action='store_true')
    a = ap.parse_args()
    print(json.dumps(build(a.run, a.mode, a.arms, a.repeats, a.validation_only), indent=2))
