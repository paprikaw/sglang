"""Freeze an experiment-private optimization, reusing the qualified controller."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'ep2_pp2_qwen_a100_20261024'


def descriptor(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def build(run, node, depth):
    spec = importlib.util.spec_from_file_location('original_prepare', BASE / 'prepare.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    receipt = base.build(run, node, ('pp2',))
    run = Path(run).resolve()
    code = run / 'code'
    for name in ('pp_ready_send.py', 'benchmark.py'):
        shutil.copyfile(ROOT / name, code / name)
        (code / name).chmod(0o600)
    adapter = code / 'runtime_adapter.py'
    value = adapter.read_text()
    marker = '    _INSTALLED = True\n'
    assert value.count(marker) == 1
    adapter.write_text(value.replace(marker, '    import pp_ready_send\n    pp_ready_send.install(torch)\n' + marker))
    runner = code / 'runner.py'
    value = runner.read_text()
    value = value.replace('import argparse\n', 'import argparse\nimport atexit\nimport hashlib\n', 1)
    marker = 'def run_workloads(run, cfg, arm, key, health):\n'
    assert value.count(marker) == 1
    value = value.replace(marker, marker + '    import benchmark\n    return benchmark.run(sys.modules[__name__], run, cfg, arm, key, health)\n', 1)
    marker = "    for arm in cfg.get('arm_order', ['ep2', 'pp2']):\n"
    assert value.count(marker) == 1
    value = value.replace(marker, "    import pp_ready_send\n    policy = Path(cfg['launch_env']['PP_READY_POLICY_PATH'])\n    pp_ready_send.create_policy(policy)\n    atexit.register(lambda: policy.unlink(missing_ok=True))\n" + marker, 1)
    marker = "        output_ids=final.get('output_ids'), text=final.get('text', '')[-200:])"
    assert value.count(marker) == 1
    value = value.replace(marker, "        text_sha256=hashlib.sha256(final.get('text', '').encode()).hexdigest(),\n" + marker)
    value = value.replace("'127.0.0.1:19422'", "'127.0.0.1:19452'")
    runner.write_text(value)
    cfg = json.loads((run / 'config.json').read_text())
    cfg.update(pp_async_batch_depth=depth, pp_loop_slots=2 + depth,
               ready_send_optimization=True, policy_order=['baseline', 'ready', 'ready', 'baseline'],
               repeats=1, port=19451)
    cfg['launch_env']['PP_READY_POLICY_PATH'] = '/tmp/' + run.name + '-policy'
    cfg['model_cache']['on_miss'] = 'fail'
    (run / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    dashboard = json.loads((run / 'dashboard_spec.json').read_text())
    dashboard.update(title='PP2 producer-ready send A/B/B/A; full Qwen122B', port=19450,
                     model_cache=cfg['model_cache'])
    dashboard['tasks'][0].update(expected=4, label='PP2 same-process baseline/ready A/B/B/A')
    frozen = {item['path']: item for item in dashboard['frozen_inputs']}
    for path in [run / 'config.json', *sorted(code.glob('*.py'))]:
        frozen[str(path)] = descriptor(path)
    # Freeze the exact native PP send implementation and this preparation code.
    for rel in ('python/sglang/srt/managers/scheduler_pp_mixin.py',
                'python/sglang/srt/distributed/parallel_state.py'):
        path = Path(cfg['engine_root']) / rel
        frozen[str(path)] = descriptor(path)
    for path in (Path(__file__).resolve(), ROOT / 'prepare.py', ROOT / 'test_ready_send.py'):
        frozen[str(path)] = descriptor(path)
    dashboard['frozen_inputs'] = list(frozen.values())
    base.old.validate_contract(dashboard)
    (run / 'dashboard_spec.json').write_text(json.dumps(dashboard, indent=2) + '\n')
    receipt.update(pp_loop_slots=2 + depth, same_process=True, no_new_cache=True)
    (run / 'optimization_preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--node', default='spartan-gpgpu069')
    ap.add_argument('--depth', type=int, choices=(0, 2), default=0)
    args = ap.parse_args()
    print(json.dumps(build(args.run, args.node, args.depth), indent=2))
