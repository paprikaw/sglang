"""Freeze runtime initialization in spawned rank entrypoints and proof gates."""
import importlib.util
from pathlib import Path
import runpy
import shutil
import sys
ROOT = Path(__file__).resolve().parent
run = Path(sys.argv[sys.argv.index('--run')+1]).resolve()
runpy.run_path(str(ROOT/'prepare_cluster_v4.py'), run_name='__main__')
spec = importlib.util.spec_from_file_location('prep_final', ROOT/'prepare_cluster.py')
prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)
shutil.copyfile(ROOT/'cluster_runner_v4.py', run/'code/cluster_runner.py')
ds = prep.read(run/'dashboard_spec.json')
paths = [Path(d['path']) for d in ds['frozen_inputs']]
paths += [ROOT/'cluster_runner_v4.py', Path(__file__).resolve()]
ds['frozen_inputs'] = list({str(p): prep.desc(p) for p in paths}.values())
prep.save(run/'dashboard_spec.json', ds)
manifest = prep.read(run/'launch_manifest.json')
manifest['frozen_inputs'] = ds['frozen_inputs']
prep.save(run/'launch_manifest.json', manifest)
