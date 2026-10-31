"""Freeze the unstarted 3xEP2 control after the first layout fails.

Build both shapes with the audited preparation first, then select the existing
second shape. Do not change ARMS before shape construction: its first entry is
used to identify the EP4 layout. This is CPU-only preparation.
"""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
ARM = 'pd_2p6d_pp2_3ep2'
PREVIOUS = ROOT/'runs/two_p_six_d_ep4ep2_vs_3ep2_e8_v4_gdn256_20261031_pilot'


def main():
    spec = importlib.util.spec_from_file_location('mha_preparation', ROOT/'prepare_2p6d_mha_mapping.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    run = Path(sys.argv[sys.argv.index('--run')+1]).resolve()
    read = lambda p: json.loads(p.read_text())
    def save(p, obj):
        p.write_text(json.dumps(obj, indent=2)+'\n')
    cfg = read(run/'config.json')
    cfg['arms'] = [ARM]
    cfg['expected_points'] = 2
    cfg['policies'] = {ARM: cfg['policies'][ARM]}
    cfg['endpoints'] = {ARM: cfg['endpoints'][ARM]}
    cfg['role_gpu_counts'] = {ARM: cfg['role_gpu_counts'][ARM]}
    cfg['continuation_of'] = str(PREVIOUS)
    cfg['continuation_reason'] = 'First layout C128 native bootstrap/receive timeout; second layout was never started. Same engine, budgets, replay and cache protocol.'
    cfg['protocol'] += ' This run executes only the previously unstarted 3xEP2 D control; it imports no performance data.'
    cache = cfg['model_cache']
    for node in cfg['nodes']:
        local = cfg['node_configs'][node]
        chosen = next(x for x in local['layouts'] if x['id'] == ARM)
        # Ports were constructed at layout index 1. Keep the same actual
        # endpoint addresses after filtering the list to index 0.
        local['port'] += 100
        local['dist_port'] += 100
        local['layouts'] = [chosen]
        local['arm_order'] = [ARM]
        local['expected_points'] = 2
        cache['usage_logs'][node] = chosen['usage_logs']
        local['model_cache'] = cache
        if node == cfg['nodes'][1]:
            assert [i['ep'] for i in chosen['instances']] == [2, 2]
    save(run/'config.json', cfg)
    pre = read(run/'preflight.json')
    for key in ('commands', 'decode_slot_budget'):
        pre[key] = [x for x in pre[key] if x['arm'] == ARM]
    # Verify endpoints and command addresses after filtering.
    for command in pre['commands']:
        local = cfg['node_configs'][command['node']]
        args = command['args']
        assert int(args[args.index('--port')+1]) == local['port'] + command['replica']*10
    save(run/'preflight.json', pre)
    ds = read(run/'dashboard_spec.json')
    ds['title'] = 'AgentX 2P6D remaining control: PP2 P +3xEP2 D; GDN256'
    ds['model_cache'] = cache
    ds['tasks'][0].update(label='3xEP2 control, C8/C128, one pilot each', expected=2)
    import hashlib
    def descriptor(p):
        return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    paths = [Path(x['path']) for x in ds['frozen_inputs']] + [Path(__file__).resolve()]
    ds['frozen_inputs'] = list({str(p): descriptor(p) for p in paths}.values())
    save(run/'dashboard_spec.json', ds)
    manifest = read(run/'launch_manifest.json')
    manifest.update(arms=[ARM], expected_points=2, endpoints=cfg['endpoints'],
                    role_gpu_counts=cfg['role_gpu_counts'], model_cache=cache,
                    state_budgets=pre['decode_slot_budget'], protocol=cfg['protocol'],
                    continuation_of=str(PREVIOUS), frozen_inputs=ds['frozen_inputs'])
    save(run/'launch_manifest.json', manifest)
    progress = read(run/'task/progress.json')
    progress['expected'] = 2
    save(run/'task/progress.json', progress)
    print('Frozen remaining control:', run)


if __name__ == '__main__':
    main()
