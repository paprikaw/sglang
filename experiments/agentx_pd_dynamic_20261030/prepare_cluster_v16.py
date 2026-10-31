"""Require a real colocated output reference before preparing cross-node P/D."""
import importlib.util
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parent
assert '--reference' in sys.argv, 'Cross-node P/D needs a frozen serial reference'
source=Path(sys.argv[sys.argv.index('--reference')+1]).resolve()
reference=json.loads(source.read_text())
assert str(reference.get('source_arm','')).startswith('col_')
assert len(reference['output_ids'])==6
assert all(isinstance(ids,list) and len(ids)==32 for ids in reference['output_ids'])
run=Path(sys.argv[sys.argv.index('--run')+1]).resolve()
runpy.run_path(str(ROOT/'prepare_cluster_v15.py'),run_name='__main__')
spec=importlib.util.spec_from_file_location('reference_prep',ROOT/'prepare_cluster.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
assert prep.read(run/'evidence/reference.json')==reference
assert prep.read(run/'reference.input.json')==reference
preflight=prep.read(run/'preflight.json')
preflight['serial_reference']=dict(source=str(source),sha256=prep.desc(source)['sha256'],
    source_arm=reference['source_arm'],sequences=6,tokens_per_sequence=32)
prep.save(run/'preflight.json',preflight)
ds=prep.read(run/'dashboard_spec.json')
paths=[Path(x['path']) for x in ds['frozen_inputs']]+[Path(__file__).resolve(),source]
ds['frozen_inputs']=list({str(p):prep.desc(p) for p in paths}.values())
prep.save(run/'dashboard_spec.json',ds)
manifest=prep.read(run/'launch_manifest.json')
manifest.update(serial_reference=preflight['serial_reference'],frozen_inputs=ds['frozen_inputs'])
prep.save(run/'launch_manifest.json',manifest)
print('Serial reference preflight passed: six complete colocated output sequences')
