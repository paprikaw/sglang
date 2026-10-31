"""Fresh PP-only chunk sweep; old admission experiment remains immutable."""
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
    cfg.update(chunk_policy_path=f'/tmp/{run.name}-chunk-budget',expected_points=24,
               port=20531,dist_port=20541,
               protocol='Same-process PP2 native Prefill budgets 4096/2048/1024/8192 in mirrored order '
               '0/1/2/3/3/2/1/0, genuine GPU cached prefixes, same inputs and output count. '
               'Original coalescing and producer-ready switches remain OFF.')
    cfg['layouts']=[r for r in cfg['layouts'] if r['id']=='pp2']
    cfg['launch_env']['PP_CHUNK_POLICY_PATH']=cfg['chunk_policy_path']
    uses=[r['usage_logs'][0] for r in cfg['layouts']]
    for contract in (cfg['model_cache'],spec['model_cache']):contract['usage_logs'][cfg['node']]=uses
    for name in ('chunk_budget.py','chunk_optimization.py'):
        shutil.copyfile(Path(__file__).with_name(name),run/'code'/name)
    runner=run/'code/runner.py'
    runner.write_text(runner.read_text().replace('from optimization import worker','from chunk_optimization import worker'))
    adapter=run/'code/runtime_adapter.py'
    adapter.write_text(adapter.read_text().replace('    coalesce.install()',
        '    coalesce.install()\n    import chunk_budget\n    chunk_budget.install()'))
    (run/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    def descriptor(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    inputs=[x for x in spec['frozen_inputs'] if not Path(x['path']).is_relative_to(run/'code')
            and x['path']!=str(run/'config.json')]
    inputs += [descriptor(p) for p in (run/'code').glob('*.py')]+[descriptor(run/'config.json')]
    spec.update(frozen_inputs=inputs,title='H100 PP2: native Prefill chunk-budget sweep',port=20530)
    spec['tasks'][0].update(expected=24,label='Same PP2 process: 4096 / 2048 / 1024 / 8192-token Prefill chunks')
    (run/'dashboard_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    p=json.loads((run/'preparation.json').read_text())
    p['command'][1]=str(Path(cfg['guard_scripts'])/'dashboard_guard.py')
    (run/'preparation.json').write_text(json.dumps(p,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();build(a.run)
