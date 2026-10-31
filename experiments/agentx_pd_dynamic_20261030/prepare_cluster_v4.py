"""Preserve private policy file modes and finish cleanup after final progress."""
import importlib.util
import os
from pathlib import Path
import runpy
import shutil
import stat
import sys
ROOT = Path(__file__).resolve().parent
run = Path(sys.argv[sys.argv.index('--run')+1]).resolve()
runpy.run_path(str(ROOT/'prepare_cluster_v3.py'), run_name='__main__')
spec = importlib.util.spec_from_file_location('prep_final', ROOT/'prepare_cluster.py')
prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)
shutil.copyfile(ROOT/'cluster_runner_v3.py', run/'code/cluster_runner.py')
controls = []
for path in run.glob('*.bin'):
    os.chmod(path, 0o600)
    info = path.lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_size == 1
    assert not info.st_mode & 0o077
    controls.append(path.name)
preflight = prep.read(run/'preflight.json')
preflight['private_controls'] = controls
prep.save(run/'preflight.json', preflight)
ds = prep.read(run/'dashboard_spec.json')
paths = [Path(d['path']) for d in ds['frozen_inputs']]
paths += [ROOT/'cluster_runner_v3.py', Path(__file__).resolve()]
ds['frozen_inputs'] = list({str(p): prep.desc(p) for p in paths}.values())
prep.save(run/'dashboard_spec.json', ds)
manifest = prep.read(run/'launch_manifest.json')
manifest['frozen_inputs'] = ds['frozen_inputs']
prep.save(run/'launch_manifest.json', manifest)
print('Private one-byte controls checked:', len(controls))
