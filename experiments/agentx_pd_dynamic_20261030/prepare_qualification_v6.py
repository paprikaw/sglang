"""Freeze diagnostic Prefill H2D completion fence for the PP restore reproducer."""
import importlib.util
from pathlib import Path
import runpy
import shutil
import sys
ROOT = Path(__file__).resolve().parent
run = Path(sys.argv[sys.argv.index('--run')+1]).resolve()
runpy.run_path(str(ROOT/'prepare_qualification_v5.py'), run_name='__main__')
spec = importlib.util.spec_from_file_location('prep', ROOT/'prepare_cluster.py')
prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)
shutil.copyfile(ROOT/'cache_probe_v4.py', run/'code/cache_probe.py')
ds = prep.read(run/'dashboard_spec.json')
paths = [Path(x['path']) for x in ds['frozen_inputs']]
paths += [ROOT/'cache_probe_v4.py', Path(__file__).resolve()]
ds['frozen_inputs'] = list({str(p): prep.desc(p) for p in paths}.values())
prep.save(run/'dashboard_spec.json', ds)
manifest = prep.read(run/'launch_manifest.json'); manifest['frozen_inputs'] = ds['frozen_inputs']
prep.save(run/'launch_manifest.json', manifest)
