"""Long-cache qualification of the unchanged A100 producer-ready adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import prepare


def build(run):
    prepare.build(run)
    run=run.resolve();cfg=json.loads((run/'config.json').read_text())
    spec=json.loads((run/'dashboard_spec.json').read_text())
    cfg.update(port=20931,dist_port=20941,expected_points=4,
        optimization_kind='ready_send',fixed_trial_branch=1400,
        optimization_cases=[r for r in cfg['optimization_cases'] if r['id']=='long_cached32'],
        protocol='H100 long-cache PP2 producer-ready metadata publication. '
                 'Same-process A/B/B/A, all four cohorts use identical inputs after '
                 'independently rebuilding genuine GPU cached prefixes. '
                 'Original 4096-token native chunks and two PP slots unchanged. '
                 'No coalescing, precision, kernel, routing or transport changes.')
    cfg['layouts']=[r for r in cfg['layouts'] if r['id']=='pp2']
    uses=[r['usage_logs'][0] for r in cfg['layouts']]
    for contract in (cfg['model_cache'],spec['model_cache']):contract['usage_logs'][cfg['node']]=uses
    shutil.copyfile(Path(__file__).with_name('optimization.py'),run/'code/optimization.py')
    (run/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    def descriptor(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    inputs=[x for x in spec['frozen_inputs'] if not Path(x['path']).is_relative_to(run/'code')
            and x['path']!=str(run/'config.json')]
    inputs += [descriptor(p) for p in (run/'code').glob('*.py')]+[descriptor(run/'config.json')]
    spec.update(port=20930,title='H100 long cached PP2 Prefill: producer-ready ABBA',frozen_inputs=inputs)
    spec['tasks'][0].update(expected=4,label='PP2 long cache, original vs producer-ready send, identical inputs')
    (run/'dashboard_spec.json').write_text(json.dumps(spec,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();build(a.run)
