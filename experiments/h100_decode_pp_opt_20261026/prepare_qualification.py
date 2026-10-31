"""Fresh, frozen PP-only diagnostic using the qualified complete model bank."""
import argparse
import json
import os
from pathlib import Path
import shutil

import prepare


def build(run, node, model):
    prepare.build(run,node,model)
    code=run/'code'
    cfg=json.loads((run/'config.json').read_text())
    spec=json.loads((run/'dashboard_spec.json').read_text())
    cfg['layouts']=[r for r in cfg['layouts'] if r['id']=='pp2']
    cfg.update(arm_order=['pp2'],expected_points=1,port=22421 if model=='35b' else 22521,
        dist_port=22422 if model=='35b' else 22522,
        transfer_audit_control=str(run/'transfer_audit_control.bin'),
        protocol='Untimed real-prefix PP transfer-integrity and microbatch audit: '
                 'baseline/early/early/baseline/baseline/early; cached4096/new1/output128/B8')
    control=Path(cfg['transfer_audit_control']);control.write_bytes(b'0');control.chmod(0o600)
    cfg['model_cache']['usage_logs'][node]=cfg['layouts'][0]['usage_logs']
    spec['model_cache']=cfg['model_cache']
    runner=(code/'runner.py').read_text()
    assert runner.count('optimization.install(sys.modules[__name__])')==1
    runner=runner.replace('import optimization\noptimization.install(sys.modules[__name__])',
        'import qualification\nqualification.install(sys.modules[__name__])')
    (code/'runner.py').write_text(runner)
    adapter=(code/'runtime_adapter.py').read_text()
    assert adapter.count('pp_schedule_opt.install()')==1
    adapter += '\n# Private diagnostic; no graph/tensor/math replacement.\n'
    adapter=adapter.replace('    pp_schedule_opt.install()',
        '    pp_schedule_opt.install()\n    import pp_transfer_audit\n    pp_transfer_audit.install()')
    (code/'runtime_adapter.py').write_text(adapter)
    for name in ('pp_transfer_audit.py','qualification.py'):
        shutil.copyfile(prepare.HERE/name,code/name)
    for p in code.glob('*.py'):p.chmod(0o600)
    prepare.save(run/'config.json',cfg)
    inputs=[r for r in spec['frozen_inputs'] if not Path(r['path']).is_relative_to(code)
            and r['path']!=str(run/'config.json')]
    inputs += [prepare.descriptor(p) for p in code.glob('*.py')]+[prepare.descriptor(run/'config.json')]
    spec.update(title=f'H100 {model} PP2 bounded transfer-integrity diagnostic',
        port=22420 if model=='35b' else 22520,
        frozen_inputs=list({r['path']:r for r in inputs}.values()))
    spec['tasks'][0].update(expected=1,label='PP2 baseline and early activation; integrity only')
    prepare.save(run/'dashboard_spec.json',spec)
    progress=json.loads((run/'task/progress.json').read_text());progress['expected']=1
    prepare.save(run/'task/progress.json',progress)
    prepare.save(run/'optimization_manifest.json',dict(kind='qualification_only',
        model=model,performance_sample=False,exact_gpu_pair=[0,1],cohorts=6,
        fixed_capture_limit_bytes_per_rank=64*1024**2,
        controls='Private one-byte mapped flags; mutate only between completed cohorts',
        no_shared_engine_changes=True,no_math_changes=True,no_new_allocations=True))
    return dict(run=str(run),model=model,prepared=True,expected_points=1)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--node',default='spartan-gpgpu178')
    parser.add_argument('--model',choices=('35b','122b'),required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.run.absolute(),args.node,args.model)))
