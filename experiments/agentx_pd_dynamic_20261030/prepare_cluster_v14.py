"""Bound the native Decode transfer queue within the declared GDN pool.

This is a launch configuration change, not an engine allocator patch. It keeps
E5, HiCache, native KV auto-sizing, and the active-request limit unchanged.
"""
import argparse
import ast
import importlib.util
from pathlib import Path
import runpy
import shutil
import sys

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(add_help=False)
ap.add_argument('--decode-extra-slots', type=int, default=8)
opts, rest = ap.parse_known_args(sys.argv[1:])
assert opts.decode_extra_slots > 0
sys.argv = [sys.argv[0]] + rest
run = Path(sys.argv[sys.argv.index('--run')+1]).resolve()
runpy.run_path(str(ROOT/'prepare_cluster_v13.py'), run_name='__main__')
spec = importlib.util.spec_from_file_location('slot_prep', ROOT/'prepare_cluster.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
shutil.copyfile(ROOT/'cluster_runner_v9.py', run/'code/cluster_runner.py')
ast.parse((run/'code/cluster_runner.py').read_text())
cfg = prep.read(run/'config.json')
cfg['decode_extra_slots'] = opts.decode_extra_slots
for c in cfg['node_configs'].values():
    c['decode_extra_slots'] = opts.decode_extra_slots
cfg['protocol'] += (
    f' Decode roles explicitly use {opts.decode_extra_slots} native transfer preallocation '
    'slots per Attention-DP worker. The upper bound (active request slots + transfer slots) '
    '* 3 fits the declared local GDN state pool, avoiding its unbudgeted expansion. '
    'Active request limits, HiCache, .95 static fraction and native KV auto-sizing remain. '
    'Actual Decode GDN slot counts and the configured queue size are checked before qualification. '
    'This is a capacity configuration workaround; the E5 allocator is unchanged.')
prep.save(run/'config.json', cfg)
sys.path.insert(0, str(run/'code'))
spec = importlib.util.spec_from_file_location('slot_preflight', run/'code/cluster_runner.py')
cluster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cluster)
verified = []
for node in cfg['nodes']:
    base, c = cluster.load_base(run, cfg, node)
    for arm in cfg['arms']:
        for index, inst in enumerate(base.layout(c, arm)['instances']):
            args = base.engine_args(c, arm, index)
            assert '--enable-hierarchical-cache' in args
            assert '--max-total-tokens' not in args
            if inst['role'] != 'decode':
                assert '--disaggregation-decode-extra-slots' not in args
                continue
            value = lambda name: int(args[args.index(name)+1])
            assert value('--disaggregation-decode-extra-slots') == opts.decode_extra_slots
            assert '--disable-overlap-schedule' not in args
            assert '--mamba-radix-cache-strategy' not in args  # Native extra_buffer default.
            dp = value('--dp-size')
            requested = value('--max-running-requests') // dp
            states = value('--max-mamba-cache-size') // dp
            # Native admission may further cap active requests; using the CLI
            # request limit is a conservative upper bound here.
            required = (requested + opts.decode_extra_slots) * 3
            assert required <= states, (node, arm, index, requested, required, states)
            verified.append(dict(node=node, arm=arm, replica=index,
                requested_active_per_dp=requested, decode_extra_slots=opts.decode_extra_slots,
                required_state_upper_bound=required, configured_states_per_rank=states))
assert verified, 'A Decode role is required for this capacity-qualified bundle'
preflight = prep.read(run/'preflight.json')
preflight['decode_slot_budget'] = verified
prep.save(run/'preflight.json', preflight)
ds = prep.read(run/'dashboard_spec.json')
paths = [Path(x['path']) for x in ds['frozen_inputs']]
paths += [ROOT/'cluster_runner_v9.py', Path(__file__).resolve(),
          ROOT/'test_decode_slot_budget.py', ROOT/'decode_slot_budget_tests.log']
ds['frozen_inputs'] = list({str(p): prep.desc(p) for p in paths}.values())
prep.save(run/'dashboard_spec.json', ds)
manifest = prep.read(run/'launch_manifest.json')
manifest.update(protocol=cfg['protocol'], decode_extra_slots=opts.decode_extra_slots,
                decode_slot_budget=verified, frozen_inputs=ds['frozen_inputs'])
prep.save(run/'launch_manifest.json', manifest)
print('Native Decode queue capacity preflight passed:', verified)
