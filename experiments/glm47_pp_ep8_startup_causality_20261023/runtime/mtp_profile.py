"""Four-way same-checkpoint/full-bank qualification; no Guard exemptions."""
import json
from common import MODEL,ARMS,select_arm,server_args,expected_ranks,execution_arms,reuse_bank_descriptor
PROFILE='glm47-semantic-bank-fourway-same-checkpoint-v1'

def validate(spec,cfg):
    model=json.loads((MODEL/'config.json').read_text())
    if model.get('architectures')!=['Glm4MoeForCausalLM'] or model.get('num_nextn_predict_layers')!=1:
        raise ValueError('Only qualified single-layer GLM MTP checkpoint admitted')
    weights=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    if not any(key.startswith('model.layers.92.') for key in weights):raise ValueError('Complete checkpoint must include MTP weights')
    contract=spec['model_cache']
    if contract['policy']!='required' or contract['on_miss']!='fail':raise ValueError('Full CPU bank is mandatory reuse-only')
    for node,descriptor in contract['node_specs'].items():
        if json.loads(open(descriptor['path']).read())['model_path']!=str(MODEL):raise ValueError('Checkpoint differs')
    expected_usage={node:set() for node in cfg['nodes']}
    for arm,mode in execution_arms(cfg).items():
        args=server_args(select_arm(cfg,arm),0)
        actual='--speculative-algorithm' in args
        if actual!=mode['mtp']:raise ValueError('Speculative switch mismatch')
        if actual and args[args.index('--model-path')+1]!=args[args.index('--speculative-draft-model-path')+1]:raise ValueError('Draft must use same complete bank')
        for index,node in enumerate(cfg['nodes']):
            logs={row['id']:row for row in contract['usage_logs'][node]}
            roles=('target','draft') if mode['mtp'] and (mode['layout']=='ep8' or index==1) else ('target',)
            for role in roles:
                expected_usage[node].add(arm+'-'+role+'-'+node)
                row=logs[arm+'-'+role+'-'+node]
                if row['expected_ranks']!=expected_ranks(mode['layout'],index):raise ValueError('Exact initialized loader rank proof missing')
            if 'draft' not in roles and arm+'-draft-'+node in logs:raise ValueError('Nonexistent draft usage must not be required')
    if spec['workload'].get('speculative_profile')!=PROFILE:raise ValueError('Missing four-way qualification profile')
    for node,expected in expected_usage.items():
        if {row['id'] for row in contract['usage_logs'][node]}!=expected:raise ValueError('Unexpected or missing loader usage for actual execution arms')
    if cfg.get('reuse_semantic_bank'):
        if (spec['workload'].get('reuse_semantic_bank') is not True
            or spec['workload'].get('source_semantic_bank_ready')!=cfg.get('source_semantic_bank_ready')
            or reuse_bank_descriptor(cfg['semantic_bank_dir'],cfg)!=cfg.get('source_semantic_bank_ready')):
            raise ValueError('Reused semantic source differs from frozen READY identity')
    return dict(passed=True,profile=PROFILE,same_complete_bank=True,fresh_consumer_count=len(execution_arms(cfg)),
        one_actual_semantic_prefill_producer=True,producer_in_current_run=not cfg.get('reuse_semantic_bank',False),
        reused_semantic_bank=cfg.get('reuse_semantic_bank',False),disk_fallback=False,guard_required_checks_unchanged=True)
