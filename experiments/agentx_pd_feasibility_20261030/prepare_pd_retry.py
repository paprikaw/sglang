"""Reuse completed colocated baselines; add the historical pinned Mooncake dependency."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def desc(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--port',type=int,required=True);a=ap.parse_args();run=a.run.resolve();ref=a.reference.resolve()
    source=read(ref/'config.json');q=read(ref/'evidence/colocated/output_qualification.json');assert q['passed']
    for c in ('c1','c8'):assert read(ref/f'results/colocated/{c}/completed.json')['passed']
    spec=importlib.util.spec_from_file_location('orig_prepare',ROOT/'prepare.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    sys.argv=[str(ROOT/'prepare.py'),'--run',str(run),'--node',source['node'],'--job',source['job_id'],'--family',source['family'],'--port',str(a.port)]
    old.main();cfg=read(run/'config.json');cfg['layouts']=[x for x in cfg['layouts'] if x['id']=='pd'];cfg['expected_points']=2
    dependency=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/deps/sglang_pd_mooncake0.3.13.post1_router0.3.2')
    assert list((dependency/'mooncake').glob('*.so'))
    cfg['mooncake_dependency_path']=str(dependency);cfg['baseline_reference_run']=str(ref)
    shutil.copyfile(ref/'evidence/colocated/output_qualification.json',run/'colocated_reference.json')
    cfg['colocated_reference']=str(run/'colocated_reference.json')
    cfg['model_cache']['usage_logs']={cfg['node']:cfg['layouts'][0]['usage_logs']}
    cfg['protocol']+=' Retry only missing PD arms with frozen Mooncake0.3.13.post1; unchanged colocated baseline is reused from '+str(ref)+'.'
    save(run/'config.json',cfg)
    ds=read(run/'dashboard_spec.json');ds['model_cache']=cfg['model_cache'];ds['tasks'][0]['expected']=2
    files=[Path(x['path']) for x in ds['frozen_inputs']]+[run/'colocated_reference.json',ROOT/'prepare_pd_retry.py',ref/'config.json']
    files+=list((dependency/'mooncake').rglob('*.so'))+list((dependency/'mooncake').rglob('*.py'))
    files+=list(dependency.glob('mooncake*dist-info/METADATA'))
    ds['frozen_inputs']=list({str(p):desc(p) for p in files}.values());save(run/'dashboard_spec.json',ds)
    m=read(run/'launch_manifest.json');m.update(layouts=cfg['layouts'],protocol=cfg['protocol'],frozen_inputs=ds['frozen_inputs'],
        baseline_reference=str(ref),dependency=dict(path=str(dependency),version='0.3.13.post1',native=[desc(p) for p in (dependency/'mooncake').rglob('*.so')]))
    save(run/'launch_manifest.json',m);print(json.dumps(dict(run=str(run),baseline=str(ref))))

if __name__=='__main__':main()
