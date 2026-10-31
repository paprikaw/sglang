"""Diagnostic rerun only, preserving the failed candidate and its source tag."""
import importlib.util
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('qualification_v3', ROOT/'prepare_qualification_v3.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
run = Path(sys.argv[sys.argv.index('--run')+1]).resolve()
module.main()
spec = importlib.util.spec_from_file_location('v1', ROOT/'prepare_qualification.py')
util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util)
shutil.copyfile(ROOT/'pp_stall_probe.py', run/'code/pp_stall_probe.py')
adapter = run/'code/runtime_adapter.py'
adapter.write_text(adapter.read_text()+'\n    import pp_stall_probe\n    pp_stall_probe.install()\n')
cfg = util.read(run/'config.json')
cfg['request_timeout_s'] = 130
util.save(run/'config.json', cfg)
ds = util.read(run/'dashboard_spec.json')
files = [Path(item['path']) for item in ds['frozen_inputs']]
files += [Path(__file__).resolve(), ROOT/'pp_stall_probe.py', run/'code/pp_stall_probe.py']
ds['frozen_inputs'] = list({str(p): util.desc(p) for p in files}.values())
util.save(run/'dashboard_spec.json', ds)
manifest = util.read(run/'launch_manifest.json')
manifest.update(frozen_inputs=ds['frozen_inputs'], diagnostic_only=True)
util.save(run/'launch_manifest.json', manifest)
