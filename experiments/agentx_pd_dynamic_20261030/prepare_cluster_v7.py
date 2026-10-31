"""Freeze explicit all-rank warmup, retaining every later formal repeat."""
import importlib.util
import math
from pathlib import Path
import runpy
import shutil
import sys
ROOT=Path(__file__).resolve().parent
run=Path(sys.argv[sys.argv.index('--run')+1]).resolve()
runpy.run_path(str(ROOT/'prepare_cluster_v6.py'),run_name='__main__')
spec=importlib.util.spec_from_file_location('warm_prep',ROOT/'prepare_cluster.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
shutil.copyfile(ROOT/'cluster_runner_v5.py',run/'code/cluster_runner.py')
cfg=prep.read(run/'config.json')
selected={i for c in cfg['cases'] for i in c['graph_ids']}
graphs=prep.read(Path(cfg['graphs_file']))['graphs']
max_input=max(t['input_tokens'] for g in graphs if g['id'] in selected for c in g['conversations'] for t in c['turns'])
cfg['warmup_contexts']=sorted({8192,65536,64*math.ceil(max_input/64)})
cfg['source_max_input_tokens']=max_input
assert max(cfg['warmup_contexts'])+96 <= min(c['profile']['max_context'] for c in cfg['node_configs'].values())
cfg['protocol']+=' Untimed cold-prefix and GPU prefix-reuse warmup covers every instance/Attention-DP rank at '+str(cfg['warmup_contexts'])+' tokens; formal episodes begin after native cache flush; every formal repeat retained.'
prep.save(run/'config.json',cfg)
ds=prep.read(run/'dashboard_spec.json')
paths=[Path(d['path']) for d in ds['frozen_inputs']]+[ROOT/'cluster_runner_v5.py',Path(__file__).resolve()]
ds['frozen_inputs']=list({str(p):prep.desc(p) for p in paths}.values())
prep.save(run/'dashboard_spec.json',ds)
manifest=prep.read(run/'launch_manifest.json')
manifest.update(protocol=cfg['protocol'],warmup_contexts=cfg['warmup_contexts'],frozen_inputs=ds['frozen_inputs'])
prep.save(run/'launch_manifest.json',manifest)
print('All-rank cold and prefix-reuse warmup frozen:',cfg['warmup_contexts'])
