"""Warm short cached batches as well as long single-context shapes."""
import ast
import importlib.util
from pathlib import Path
import runpy
import shutil
import sys

ROOT=Path(__file__).resolve().parent
run=Path(sys.argv[sys.argv.index('--run')+1]).resolve()
runpy.run_path(str(ROOT/'prepare_cluster_v8.py'),run_name='__main__')
spec=importlib.util.spec_from_file_location('batch_warm_prep',ROOT/'prepare_cluster.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
shutil.copyfile(ROOT/'cluster_runner_v7.py',run/'code/cluster_runner.py')
ast.parse((run/'code/cluster_runner.py').read_text())
cfg=prep.read(run/'config.json')
width=min(c['profile']['max_requests']//4 for c in cfg['node_configs'].values())
cases={(length,1) for length in cfg['warmup_contexts']}
cases.update((8192,w) for w in (4,width) if w<=width)
cfg['warmup_cases']=[dict(input_tokens=length,requests_per_attention_rank=w)
                     for length,w in sorted(cases)]
cfg['protocol']+=' Short-prefix warmup also offers widths 4 and '+str(width)+' per Attention-DP rank; cold and cached phases are untimed, and actual batches remain scheduler-dependent.'
prep.save(run/'config.json',cfg)
ds=prep.read(run/'dashboard_spec.json')
paths=[Path(d['path']) for d in ds['frozen_inputs']]+[ROOT/'cluster_runner_v7.py',Path(__file__).resolve()]
ds['frozen_inputs']=list({str(p):prep.desc(p) for p in paths}.values())
prep.save(run/'dashboard_spec.json',ds)
manifest=prep.read(run/'launch_manifest.json')
manifest.update(protocol=cfg['protocol'],warmup_cases=cfg['warmup_cases'],frozen_inputs=ds['frozen_inputs'])
prep.save(run/'launch_manifest.json',manifest)
print('Untimed batch warmup frozen:',cfg['warmup_cases'])
