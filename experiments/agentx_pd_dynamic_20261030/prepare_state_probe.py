"""Freeze P/D native state fingerprints around the repeated restore cases."""
import ast
import importlib.util
from pathlib import Path
import runpy
import shutil
import sys

ROOT = Path(__file__).resolve().parent
run = Path(sys.argv[sys.argv.index('--run') + 1]).resolve()
runpy.run_path(str(ROOT / 'prepare_qualification_v5.py'), run_name='__main__')
spec = importlib.util.spec_from_file_location('state_prep', ROOT / 'prepare_cluster.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
shutil.copyfile(ROOT / 'state_probe.py', run / 'code/state_probe.py')
adapter = run / 'code/runtime_adapter.py'
adapter.write_text(adapter.read_text() + '\n    import state_probe\n    state_probe.install()\n')
ast.parse(adapter.read_text())
ast.parse((run / 'code/state_probe.py').read_text())
cfg = prep.read(run / 'config.json')
cfg['protocol'] += ' Intrusive state fingerprint diagnostic on P final-send and D committed-receive boundaries; GPU fences and CPU copies; never a performance sample.'
prep.save(run / 'config.json', cfg)
ds = prep.read(run / 'dashboard_spec.json')
paths = [Path(d['path']) for d in ds['frozen_inputs']] + [ROOT / 'state_probe.py', run / 'code/state_probe.py', Path(__file__).resolve()]
ds['frozen_inputs'] = list({str(p): prep.desc(p) for p in paths}.values())
prep.save(run / 'dashboard_spec.json', ds)
manifest = prep.read(run / 'launch_manifest.json')
manifest.update(protocol=cfg['protocol'], frozen_inputs=ds['frozen_inputs'])
prep.save(run / 'launch_manifest.json', manifest)
print('State fingerprints frozen; correctness diagnosis only')
