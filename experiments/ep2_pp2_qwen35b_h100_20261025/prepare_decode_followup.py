"""Fresh frozen Guard contract for PP-only diagnostic windows."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def prepare(run, reference, node):
    loader=importlib.util.spec_from_file_location('small_h100_preparer',Path(__file__).with_name('prepare.py'))
    preparer=importlib.util.module_from_spec(loader);loader.loader.exec_module(preparer)
    result=preparer.build(run,node)
    run=run.resolve();reference=reference.resolve()
    cfg_path=run/'config.json';spec_path=run/'dashboard_spec.json'
    cfg=json.loads(cfg_path.read_text());spec=json.loads(spec_path.read_text())
    prior=json.loads((reference/'completed.json').read_text())
    assert prior['passed'] and cfg['node']==node
    cfg.update(decode_profile_only=True,decode_profile_steps=16,reference_run=str(reference),
        reference_capacities=prior['logical_capacities'],
        reference_cases=[r for r in prior['cases'] if r['id'].startswith('near_')])
    assert len(cfg['reference_cases'])==2
    cfg['layouts']=[r for r in cfg['layouts'] if r['id']=='pp2']
    for contract in (cfg['model_cache'],spec['model_cache']):
        contract['usage_logs'][node]=[u for u in contract['usage_logs'][node]
                                     if u['id'].endswith('.pp2')]
    cfg_path.write_text(json.dumps(cfg,indent=2)+'\n')
    def descriptor(path):
        return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    spec['tasks'][0].update(expected=2,label='PP2 Decode profile windows only; no formal rerun')
    spec['title']='Qwen35B PP2: two missing long-cache Decode Graph windows'
    spec['frozen_inputs']=[x for x in spec['frozen_inputs'] if x['path']!=str(cfg_path)]
    spec['frozen_inputs'] += [descriptor(cfg_path),descriptor(reference/'completed.json')]
    spec_path.write_text(json.dumps(spec,indent=2)+'\n')
    print(json.dumps(dict(command=result['command'],run=str(run)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--node',default='spartan-gpgpu178');a=p.parse_args()
    prepare(a.run,a.reference,a.node)
