"""Freeze untimed per-rank forward snapshots around each full replay."""
import ast
import importlib.util
from pathlib import Path
import runpy
import shutil
import sys

ROOT=Path(__file__).resolve().parent
run=Path(sys.argv[sys.argv.index('--run')+1]).resolve()
runpy.run_path(str(ROOT/'prepare_cluster_v7.py'),run_name='__main__')
spec=importlib.util.spec_from_file_location('counter_prep',ROOT/'prepare_cluster.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
shutil.copyfile(ROOT/'cluster_runner_v6.py',run/'code/cluster_runner.py')
ast.parse((run/'code/cluster_runner.py').read_text())
cfg=prep.read(run/'config.json')
cfg['protocol']+=' Existing Python forward counters are snapshotted on all eight ranks at idle episode boundaries outside formal timing; no GPU timing fences or new hot-path instrumentation.'
cfg['forward_snapshot_protocol']='per-rank before/after counters, outside formal timing'
prep.save(run/'config.json',cfg)
ds=prep.read(run/'dashboard_spec.json')
paths=[Path(d['path']) for d in ds['frozen_inputs']]+[ROOT/'cluster_runner_v6.py',Path(__file__).resolve()]
ds['frozen_inputs']=list({str(p):prep.desc(p) for p in paths}.values())
prep.save(run/'dashboard_spec.json',ds)
manifest=prep.read(run/'launch_manifest.json')
manifest.update(protocol=cfg['protocol'],forward_snapshot_protocol=cfg['forward_snapshot_protocol'],frozen_inputs=ds['frozen_inputs'])
prep.save(run/'launch_manifest.json',manifest)
print('Per-rank forward snapshots frozen outside formal replay timing')
