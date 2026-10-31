"""Set a declared PP microbatch cap after the shared warmup protocol."""
import argparse
import importlib.util
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(add_help=False)
ap.add_argument('--pp-microbatch',type=int,choices=[8,16],default=16)
opts,rest=ap.parse_known_args(sys.argv[1:])
sys.argv=[sys.argv[0]]+rest
run=Path(sys.argv[sys.argv.index('--run')+1]).resolve()
runpy.run_path(str(ROOT/'prepare_cluster_v9.py'),run_name='__main__')
spec=importlib.util.spec_from_file_location('pp_prep',ROOT/'prepare_cluster.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
cfg=prep.read(run/'config.json')
for node_cfg in cfg['node_configs'].values():
    node_cfg['pp_stream_microbatch']=opts.pp_microbatch
cfg['pp_microbatch_cap']=opts.pp_microbatch
cfg['protocol']+=' PP per-slot microbatch cap '+str(opts.pp_microbatch)+'; native policy0 retained, fixed-batch policy9 is not reused for AgentX.'
prep.save(run/'config.json',cfg)
sys.path.insert(0,str(run/'code'))
spec=importlib.util.spec_from_file_location('pp_cap_preflight',run/'code/cluster_runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
verified=[]
for node in cfg['nodes']:
    base,conf=runner.load_base(run,cfg,node)
    for arm in cfg['arms']:
        for index,inst in enumerate(base.layout(conf,arm)['instances']):
            args=base.engine_args(conf,arm,index)
            if inst['pp']>1:
                assert int(args[args.index('--pp-max-micro-batch-size')+1])==opts.pp_microbatch
                verified.append(dict(node=node,arm=arm,replica=index,pp_microbatch_cap=opts.pp_microbatch))
assert verified
preflight=prep.read(run/'preflight.json');preflight['pp_cap_commands']=verified
prep.save(run/'preflight.json',preflight)
ds=prep.read(run/'dashboard_spec.json')
paths=[Path(d['path']) for d in ds['frozen_inputs']]+[Path(__file__).resolve()]
ds['frozen_inputs']=list({str(p):prep.desc(p) for p in paths}.values())
prep.save(run/'dashboard_spec.json',ds)
manifest=prep.read(run/'launch_manifest.json')
manifest.update(protocol=cfg['protocol'],pp_microbatch_cap=opts.pp_microbatch,frozen_inputs=ds['frozen_inputs'])
prep.save(run/'launch_manifest.json',manifest)
print('PP microbatch command preflight passed:',verified)
