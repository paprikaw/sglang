"""Freeze a new small-model experiment; never modify an accepted run."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
LARGE = ROOT.parent / 'ep2_pp2_qwen_h100_20261025'


def build(run, node):
    loader = importlib.util.spec_from_file_location('hopper_preparer', LARGE / 'prepare.py')
    prep = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(prep)
    result = prep.build(run, node, reuse_bank=True)
    run = Path(run).absolute()
    cfg = json.loads((run / 'config.json').read_text())
    spec = json.loads((run / 'dashboard_spec.json').read_text())
    model, info, model_inputs = prep.base.old.model_identity('small')
    old_model = Path(cfg['model_path'])
    code = run / 'code'
    for name in ('runner.py', 'runtime_adapter.py', 'protocol.py', 'prefill_followup.py', 'decode_followup.py'):
        shutil.copyfile(ROOT / name, code / name)
        os.chmod(code / name, 0o600)
    node_path = run / 'cache/node.json'
    node_spec = json.loads(node_path.read_text())
    node_spec['model_path'] = str(model)
    node_path.write_text(json.dumps(node_spec, indent=2) + '\n')
    for contract in (cfg['model_cache'], spec['model_cache']):
        contract['node_specs'][node] = prep.base.old.descriptor(node_path)
    (run / 'single').mkdir(mode=0o700)
    use = dict(id=run.name+'.single', path=str(run/'cache/single.log'),
               expected_ranks=['TP0'], single_instance_tp1=True)
    prep.base.old.create_usage_log(use['path'])
    cfg['layouts'].append(dict(id='single',tp=1,ep=1,dp=1,pp=1,
        gpu_groups=[[0]],replicas=1,usage_logs=[use]))
    for contract in (cfg['model_cache'], spec['model_cache']):
        contract['usage_logs'][node].append(use)
    cfg.update(model_path=str(model), model_revision=info['revision'], model_cohort='small',
        num_hidden_layers=40, quantization=None, port=19931, dist_port=19941,
        max_global_requests=32,
        precision='Unquantized BF16 weights/activations/KV, FP32 GDN states; native Hopper DeepGEMM',
        protocol='One-GPU-sized full model, compared on two GPUs: EP2 Attention DP2/TP1 versus '
                 'PP2 Attention TP1/EP1 with two equal slots. Moderate cold Prefill4096 and '
                 'cached4096 Decode; capacity-relative cached Prefill512 and Decode1->128, '
                 '16/32 requests. Real GPU radix/GDN prefixes; all preparation outside timing. '
                 'Three formal rounds, independent phase-specific GPU profiles afterwards.')
    cfg['launch_env']['SGLANG_DEEPEP_BF16_DISPATCH'] = '1'
    cfg['launch_env']['SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK'] = '32'
    (run / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    inputs = [row for row in spec['frozen_inputs']
              if not Path(row['path']).is_relative_to(code)
              and not Path(row['path']).is_relative_to(old_model)
              and row['path'] not in (str(run / 'config.json'), str(node_path))]
    inputs += model_inputs
    inputs += [prep.base.old.descriptor(p) for p in code.glob('*.py')]
    inputs += [prep.base.old.descriptor(p) for p in (run / 'config.json', node_path)]
    inputs += [prep.base.old.descriptor(prep.base.old.ENGINE /
        'python/sglang/srt/layers/quantization/unquant.py')]
    spec.update(title='H100 Qwen35B BF16: parallel EP2 versus two-slot PP2', port=19930,
                frozen_inputs=list({r['path']: r for r in inputs}.values()))
    spec['tasks'][0].update(label='Single-GPU-sized model: EP2 GPU0/1 and PP2 GPU2/3', expected=12)
    prep.base.old.validate_contract(spec)
    (run / 'dashboard_spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    result['command'].insert(result['command'].index('--exit-after-run'), '--resource-choice')
    result['command'].insert(result['command'].index('--exit-after-run'), 'auto')
    (run / 'preparation.json').write_text(json.dumps(dict(command=result['command'],
        required_environment=prep.base.old.shared_environment(), node=node,
        job_id=cfg['job_id']), indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--node', default='spartan-gpgpu178')
    a = p.parse_args()
    print(json.dumps(build(a.run, a.node), indent=2))
