"""Prepare a new fixed-request-group EP/PP comparison, never edit live runs."""
import argparse
import json
from pathlib import Path
import shutil
import prepare


def build(run,node,model):
    prepare.build(run,node,model)
    cfg=json.loads((run/'config.json').read_text())
    spec=json.loads((run/'dashboard_spec.json').read_text());code=run/'code'
    cfg.update(expected_points=3,port=24321 if model=='35b' else 24421,
        dist_port=24322 if model=='35b' else 24422,
        fixed_cohort_control=str(run/'fixed_cohort_control.bin'),
        transfer_audit_control=str(run/'transfer_audit_control.bin'),
        protocol='EP2 and PP2 share even/odd request groups; same GPUs0/1, '
                 'native CUDA Graph, real GPUcached4096/new1/output128/B8; '
                 'three EP baselines, PP baseline/early/early/baseline/baseline/early; '
                 'separate payload audit and profiles, no producer-ready change')
    for field,byte in (('fixed_cohort_control',b'\x00'),('transfer_audit_control',b'0')):
        path=Path(cfg[field]);path.write_bytes(byte);path.chmod(0o600)
    runner=(code/'runner.py').read_text()
    assert runner.count('optimization.install(sys.modules[__name__])')==1
    (code/'runner.py').write_text(runner.replace(
        'import optimization\noptimization.install(sys.modules[__name__])',
        'import controlled\ncontrolled.install(sys.modules[__name__])'))
    adapter=(code/'runtime_adapter.py').read_text()
    assert adapter.count('pp_schedule_opt.install()')==1
    adapter=adapter.replace('    pp_schedule_opt.install()',
        '    pp_schedule_opt.install()\n    import pp_transfer_audit, pp_fixed_cohort\n'
        '    pp_transfer_audit.install()\n    pp_fixed_cohort.install()')
    (code/'runtime_adapter.py').write_text(adapter)
    for name in ('controlled.py','pp_transfer_audit.py','pp_fixed_cohort.py'):
        shutil.copyfile(prepare.HERE/name,code/name)
    for path in code.glob('*.py'):path.chmod(0o600)
    prepare.save(run/'config.json',cfg)
    inputs=[r for r in spec['frozen_inputs'] if not Path(r['path']).is_relative_to(code)
            and r['path']!=str(run/'config.json')]
    inputs += [prepare.descriptor(p) for p in code.glob('*.py')]+[prepare.descriptor(run/'config.json')]
    spec.update(title=f'H100 {model}: fixed request groups EP2 and PP2 scheduling',
        port=24320 if model=='35b' else 24420,
        frozen_inputs=list({r['path']:r for r in inputs}.values()))
    spec['tasks'][0].update(expected=3,label='EP2 reference then PP2 A/B with identical request groups')
    prepare.save(run/'dashboard_spec.json',spec)
    progress=json.loads((run/'task/progress.json').read_text());progress['expected']=3
    prepare.save(run/'task/progress.json',progress)
    prepare.save(run/'optimization_manifest.json',dict(kind='fixed_groups_controlled_optimization',
        model=model,model_math_unchanged=True,gpu_pair=[0,1],pp_slots=2,
        document_groups=[[0,2,4,6],[1,3,5,7]],admission_wait_included_in_client_time=True,
        no_shared_source_changes=True,no_new_allocations=True,
        tests='Unprofiled formal latency; independent payload correctness audit and GPU profiles'))
    return dict(prepared=True,run=str(run),expected_points=3,model=model)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--node',default='spartan-gpgpu178')
    parser.add_argument('--model',choices=('35b','122b'),required=True);args=parser.parse_args()
    print(json.dumps(build(args.run.absolute(),args.node,args.model)))
