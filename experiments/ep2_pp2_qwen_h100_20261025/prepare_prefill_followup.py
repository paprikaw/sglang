"""Freeze a NEW Prefill-only follow-up; never alter accepted r7 evidence."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent


def build(run, node):
    module_spec = importlib.util.spec_from_file_location('h100_preparation', ROOT / 'prepare.py')
    prep = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(prep)
    result = prep.build(run, node, reuse_bank=True)
    run = Path(run).absolute()
    cfg = json.loads((run / 'config.json').read_text())
    spec = json.loads((run / 'dashboard_spec.json').read_text())
    code = run / 'code'
    path = code / 'runner.py'
    text = path.read_text()
    old = 'worker(args.run, cfg) if args.worker'
    assert text.count(old) == 1
    text = 'import prefill_followup\n' + text.replace(old,
        'prefill_followup.worker(args.run, cfg, sys.modules[__name__]) if args.worker')
    text = text.replace("len(list(run.glob('*/metrics.*.json')))",
                        "len(list(run.glob('*/traces/*/manifest.json')))")
    text = text.replace('expected=14', 'expected=7')
    path.write_text(text)
    shutil.copyfile(ROOT / 'prefill_followup.py', code / 'prefill_followup.py')
    adapter = code / 'runtime_adapter.py'
    text = adapter.read_text()
    old = "            if label:\n                module.forward = instrument(module.forward, label, layer=layer)"
    assert text.count(old) == 1
    text = text.replace(old, old + "\n            if suffix == '' and hasattr(module, 'self_attention'):\n"
        "                module.self_attention = instrument(module.self_attention,\n"
        "                    'component.attention.full', layer=layer)")
    adapter.write_text(text)
    reference = ROOT / 'runs/h100_ep2_pp2_r7'
    capacity_path = reference / 'capacity_matrix.json'
    tuning_path = reference / 'pp2/tuning.json'
    cfg.update(protocol='Prefill-only diagnostic follow-up to accepted H100 r7; one shape warmup '
        'and one all-chunk trace per cell; original full-model/native kernel/cache/Graph settings; '
        'all prefix preparation outside profiler; no formal performance rerun',
        reference_capacities=json.loads(capacity_path.read_text())['logical_capacities'],
        prefill_ready_send=json.loads(tuning_path.read_text())['prefill_ready_send_selected'],
        reference_run=str(reference), port=19731, dist_port=19741)
    (run / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    spec['title'] = 'H100 EP2 versus PP2: missing Prefill component traces only'
    spec['port'] = 19730
    spec['tasks'][0].update(expected=7, label='Seven Prefill-only trace cells; same native H100 pairs')
    inputs = [r for r in spec['frozen_inputs'] if not Path(r['path']).is_relative_to(code)
              and r['path'] != str(run / 'config.json')]
    inputs += [prep.base.old.descriptor(p) for p in code.glob('*.py')]
    inputs += [prep.base.old.descriptor(p) for p in (run / 'config.json', capacity_path, tuning_path)]
    spec['frozen_inputs'] = list({r['path']:r for r in inputs}.values())
    prep.base.old.validate_contract(spec)
    (run / 'dashboard_spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    os.chmod(code / 'prefill_followup.py', 0o600)
    progress = run / 'task/progress.json'
    row = json.loads(progress.read_text()); row['expected'] = 7
    progress.write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--node', default='spartan-gpgpu178')
    args = p.parse_args()
    build(args.run, args.node)
