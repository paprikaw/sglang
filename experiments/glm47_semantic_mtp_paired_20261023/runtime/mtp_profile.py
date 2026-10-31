"""Narrow same-bank GLM MTP qualification layered over canonical GLM admission.

This does not change Guard, admit a new provider, suppress a loader proof, or
authorize an arbitrary speculative model. Both runners must use the SAME sealed
full-checkpoint bank. Draft receives its own exact-rank usage-log descriptors.
"""
import json
from common import MODEL, LAYOUTS, server_args, expected_ranks

PROFILE = 'glm47-canonical-same-bank-single-mtp-v1'

def validate(spec, cfg):
    config = json.loads((MODEL / 'config.json').read_text())
    if config.get('architectures') != ['Glm4MoeForCausalLM'] or config.get('num_nextn_predict_layers') != 1:
        raise ValueError('Only the qualified GLM single-layer MTP checkpoint is admitted')
    weights = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    if not any(k.startswith('model.layers.92.') for k in weights):
        raise ValueError('The canonical complete checkpoint must contain MTP layer 92')
    contract = spec['model_cache']
    if contract['policy'] != 'required' or contract['on_miss'] != 'fail':
        raise ValueError('Qualification is reuse-only; cannot build or skip the canonical bank')
    for node, descriptor in contract['node_specs'].items():
        if json.loads(open(descriptor['path']).read())['model_path'] != str(MODEL):
            raise ValueError('Target/draft bank checkpoint differs')
    for layout in LAYOUTS:
        args = server_args(dict(cfg, layout=layout), 0)
        if args[args.index('--model-path')+1] != args[args.index('--speculative-draft-model-path')+1]:
            raise ValueError('Separate draft checkpoint is not qualified')
        if args[args.index('--speculative-algorithm')+1] != 'EAGLE':
            raise ValueError('Only native single-layer MTP via EAGLE is qualified')
        for index, node in enumerate(cfg['nodes']):
            logs = {r['id']:r for r in contract['usage_logs'][node]}
            for role in ('target', 'draft') if layout == 'ep8' or index == 1 else ('target',):
                row = logs[layout+'-'+role+'-'+node]
                if row['expected_ranks'] != expected_ranks(layout,index):
                    raise ValueError('Every actual target/draft loader requires its own exact rank proof')
    if spec['workload'].get('speculative_profile') != PROFILE:
        raise ValueError('Speculative qualification profile must be declared explicitly')
    return dict(passed=True,profile=PROFILE,same_complete_bank=True,disk_fallback=False,
                guard_required_checks_unchanged=True)
