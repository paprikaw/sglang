"""Counterbalance cloned trace families without changing cache affinity."""
import ast
import importlib.util
from pathlib import Path
import runpy
import shutil
import sys

ROOT = Path(__file__).resolve().parent
run = Path(sys.argv[sys.argv.index('--run') + 1]).resolve()
previous = 'prepare_cluster_v10.py' if '--pp-microbatch' in sys.argv else 'prepare_cluster_v9.py'
runpy.run_path(str(ROOT / previous), run_name='__main__')
spec = importlib.util.spec_from_file_location('balanced_prep', ROOT / 'prepare_cluster.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
shutil.copyfile(ROOT / 'agentx_replay_v2.py', run / 'code/agentx_replay.py')
ast.parse((run / 'code/agentx_replay.py').read_text())
cfg = prep.read(run / 'config.json')
cfg['clone_lane_policy'] = 'cyclic_copy_offset_v1'
cfg['protocol'] += (' Conversation first-placement lane includes the replay copy number as a cyclic offset; '
                    'this counterbalances repeated families across GPU cache owners without consulting '
                    'token lengths or future work. Each conversation keeps its assigned lane. '
                    'Unreplicated copy0 placement is unchanged.')
prep.save(run / 'config.json', cfg)
ds = prep.read(run / 'dashboard_spec.json')
paths = [Path(d['path']) for d in ds['frozen_inputs']] + [ROOT / 'agentx_replay_v2.py', Path(__file__).resolve()]
ds['frozen_inputs'] = list({str(p): prep.desc(p) for p in paths}.values())
prep.save(run / 'dashboard_spec.json', ds)
manifest = prep.read(run / 'launch_manifest.json')
manifest.update(protocol=cfg['protocol'], clone_lane_policy=cfg['clone_lane_policy'], frozen_inputs=ds['frozen_inputs'])
prep.save(run / 'launch_manifest.json', manifest)
print('Clone placement counterbalance frozen:', cfg['clone_lane_policy'])
