"""Generalize clone counterbalancing to asymmetric P:D role pools."""
import ast
import importlib.util
from pathlib import Path
import runpy
import shutil
import sys

ROOT=Path(__file__).resolve().parent
run=Path(sys.argv[sys.argv.index('--run')+1]).resolve()
runpy.run_path(str(ROOT/'prepare_cluster_v11.py'),run_name='__main__')
spec=importlib.util.spec_from_file_location('pool_prep',ROOT/'prepare_cluster.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
shutil.copyfile(ROOT/'agentx_replay_v3.py',run/'code/agentx_replay.py')
ast.parse((run/'code/agentx_replay.py').read_text())
cfg=prep.read(run/'config.json')
cfg['clone_lane_policy']='pool_counterbalanced_copy_offset_v2'
cfg['protocol']+=(' For role pools whose cache-owner cycle does not divide the family count, '
    'cloned first-placement blocks use a co-prime stride in arrival order. This balances each initial family '
    'across six Decode owners in 2P6D, without token-length or future-work inputs. '
    'The existing two/four/eight-owner routes are exactly unchanged; unreplicated C8 is unchanged. '
    'Later conversations retain their first assigned owner, without claiming equal work distribution.')
prep.save(run/'config.json',cfg)
ds=prep.read(run/'dashboard_spec.json')
paths=[Path(d['path']) for d in ds['frozen_inputs']]+[ROOT/'agentx_replay_v3.py',Path(__file__).resolve(),ROOT/'test_clone_pool_routing.py',ROOT/'clone_pool_routing_tests.log']
ds['frozen_inputs']=list({str(p):prep.desc(p) for p in paths}.values())
prep.save(run/'dashboard_spec.json',ds)
manifest=prep.read(run/'launch_manifest.json')
manifest.update(protocol=cfg['protocol'],clone_lane_policy=cfg['clone_lane_policy'],frozen_inputs=ds['frozen_inputs'])
prep.save(run/'launch_manifest.json',manifest)
print('Role-pool clone counterbalance frozen:',cfg['clone_lane_policy'])
