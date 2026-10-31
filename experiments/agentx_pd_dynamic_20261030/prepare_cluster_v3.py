"""Freeze private auth and validate both node launch bundles before Guard."""
import importlib.util
import os
from pathlib import Path
import runpy
import secrets
import shutil
import sys

ROOT = Path(__file__).resolve().parent
run = Path(sys.argv[sys.argv.index('--run')+1]).resolve()
runpy.run_path(str(ROOT/'prepare_cluster_v2.py'), run_name='__main__')
spec = importlib.util.spec_from_file_location('prepare_cluster', ROOT/'prepare_cluster.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
shutil.copyfile(ROOT/'cluster_runner_v2.py', run/'code/cluster_runner.py')
cfg = prep.read(run/'config.json')
for node_cfg in cfg['node_configs'].values():
    node_cfg.pop('screening_windows', None)
key_path = run/'admin.key'
fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as out:
    out.write(secrets.token_hex(32)+'\n')
prep.save(run/'config.json', cfg)
sys.path.insert(0, str(run/'code'))
spec = importlib.util.spec_from_file_location('cluster_preflight', run/'code/cluster_runner.py')
cluster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cluster)
validated = []
for node in cfg['nodes']:
    base, c = cluster.load_base(run, cfg, node)
    assert base.h.read_key(c)
    for name, value in c.items():
        if isinstance(value, str) and value.startswith('/'):
            assert Path(value).exists(), (node, name, value)
    for arm in cfg['arms']:
        for i, inst in enumerate(base.layout(c, arm)['instances']):
            args = base.engine_args(c, arm, i)
            assert '--enable-hierarchical-cache' in args
            assert '--max-total-tokens' not in args
            assert args.count('--disaggregation-mode') == int(inst['role'] != 'colocated')
            assert args[args.index('--host')+1] == '0.0.0.0'
            assert int(args[args.index('--pp-size')+1]) == inst['pp']
            validated.append(dict(node=node, arm=arm, replica=i, role=inst['role'], ep=inst['ep'], pp=inst['pp']))
prep.save(run/'preflight.json', dict(passed=True, instances=validated, authentication='private generated key validated without disclosure'))
ds = prep.read(run/'dashboard_spec.json')
paths = [Path(d['path']) for d in ds['frozen_inputs']]
paths += [ROOT/'cluster_runner_v2.py', Path(__file__).resolve(), key_path, run/'preflight.json']
ds['frozen_inputs'] = list({str(p): prep.desc(p) for p in paths}.values())
prep.save(run/'dashboard_spec.json', ds)
manifest = prep.read(run/'launch_manifest.json')
manifest['frozen_inputs'] = ds['frozen_inputs']
prep.save(run/'launch_manifest.json', manifest)
print('Private auth and two-node command preflight passed for', len(validated), 'instances')
