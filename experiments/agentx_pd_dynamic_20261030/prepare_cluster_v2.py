"""Create node-specific loader log files before canonical Guard admission."""
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cluster_prepare', ROOT/'prepare_cluster.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
run = Path(sys.argv[sys.argv.index('--run')+1]).resolve()
module.main()
cfg = module.read(run/'config.json')
for logs in cfg['model_cache']['usage_logs'].values():
    for descriptor in logs:
        path = Path(descriptor['path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x'):
            pass
        os.chmod(path, 0o600)
ds = module.read(run/'dashboard_spec.json')
ds['frozen_inputs'].append(module.desc(Path(__file__).resolve()))
module.save(run/'dashboard_spec.json', ds)
manifest = module.read(run/'launch_manifest.json')
manifest['frozen_inputs'] = ds['frozen_inputs']
module.save(run/'launch_manifest.json', manifest)
