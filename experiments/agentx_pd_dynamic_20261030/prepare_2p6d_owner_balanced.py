"""Freeze a corrected EP4+EP2 control with static per-owner placement."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
ARM = 'pd_2p6d_pp2_ep4ep2'
PREVIOUS = ROOT/'runs/two_p_six_d_ep4ep2_vs_3ep2_e8_v4_gdn256_20261031_pilot'


def main():
    log = ROOT/'unit_owner_routing_20261031.log'
    text = log.read_text()
    assert 'Ran 6 tests' in text and '\nOK\n' in text and 'FAILED' not in text
    spec = importlib.util.spec_from_file_location('mha_preparation', ROOT/'prepare_2p6d_mha_mapping.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    run = Path(sys.argv[sys.argv.index('--run')+1]).resolve()
    read = lambda p: json.loads(p.read_text())
    def save(p, obj):
        p.write_text(json.dumps(obj, indent=2)+'\n')
    def descriptor(p):
        return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    cfg = read(run/'config.json')
    cfg.update(arms=[ARM], expected_points=2,
        policies={ARM: cfg['policies'][ARM]}, endpoints={ARM: cfg['endpoints'][ARM]},
        role_gpu_counts={ARM: cfg['role_gpu_counts'][ARM]})
    cfg['placement_correction'] = dict(previous_run=str(PREVIOUS),
        previous='equal instances: EP2 owners32 roots each, EP4 owners16 roots each at C128',
        current='equal logical Attention-DP owners; root-copy block counterbalance; C12820-22 initial roots per owner',
        physical_control='Same physical GPU/cache owner and local/cross-node path as the original 3xEP2 control for identical first-placement arrival order',
        uniform_pools_unchanged=True, tests=descriptor(log))
    cfg['protocol'] += ' This run only executes EP4+EP2 D with corrected static per-owner initial placement. Uniform 3xEP2 and P pools use their original router unchanged. No dynamic policy or future-token oracle.'
    cache = cfg['model_cache']
    for node in cfg['nodes']:
        local = cfg['node_configs'][node]
        chosen = next(x for x in local['layouts'] if x['id'] == ARM)
        local.update(layouts=[chosen], arm_order=[ARM], expected_points=2)
        cache['usage_logs'][node] = chosen['usage_logs']
        local['model_cache'] = cache
        if node == cfg['nodes'][1]:
            assert [i['ep'] for i in chosen['instances']] == [4]
    patch = ROOT/'heterogeneous_owner_routing.py'
    shutil.copyfile(patch, run/'code'/patch.name)
    replay = run/'code/agentx_replay.py'
    with replay.open('a') as out:
        out.write('\n\n# Frozen static placement correction for unequal EP pools.\n'
                  'from heterogeneous_owner_routing import install as _install_owner_router\n'
                  '_install_owner_router(Router)\n')
    save(run/'config.json', cfg)
    pre = read(run/'preflight.json')
    for key in ('commands', 'decode_slot_budget'):
        pre[key] = [x for x in pre[key] if x['arm'] == ARM]
    pre['placement_correction'] = cfg['placement_correction']
    pre['replay']['sha256'] = descriptor(replay)['sha256']
    save(run/'preflight.json', pre)
    ds = read(run/'dashboard_spec.json')
    ds.update(title='AgentX2P6D PP2 P +EP4/EP2 D, corrected per-owner placement', model_cache=cache)
    ds['tasks'][0].update(label='Owner-balanced EP4+EP2 control C8/C128', expected=2)
    paths = [Path(x['path']) for x in ds['frozen_inputs']]
    paths += [Path(__file__).resolve(), patch, run/'code'/patch.name,
              ROOT/'test_heterogeneous_owner_routing.py', log]
    ds['frozen_inputs'] = list({str(p): descriptor(p) for p in paths}.values())
    save(run/'dashboard_spec.json', ds)
    manifest = read(run/'launch_manifest.json')
    manifest.update(arms=[ARM], expected_points=2, endpoints=cfg['endpoints'],
                    role_gpu_counts=cfg['role_gpu_counts'], model_cache=cache,
                    state_budgets=pre['decode_slot_budget'], protocol=cfg['protocol'],
                    placement_correction=cfg['placement_correction'], frozen_inputs=ds['frozen_inputs'])
    save(run/'launch_manifest.json', manifest)
    progress = read(run/'task/progress.json')
    progress['expected'] = 2
    save(run/'task/progress.json', progress)
    print('Frozen corrected EP4+EP2 control:', run)


if __name__ == '__main__':
    main()
