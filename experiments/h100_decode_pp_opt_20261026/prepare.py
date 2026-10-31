"""Prepare a fresh private scheduling comparison; CPU only, no node probes."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import secrets

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
STEADY = HERE.parent / 'pp_ep_steady_decode_20261026'
A100 = HERE.parent / 'ep2_pp2_qwen35b_a100_20261025'


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path,data):
    path.write_text(json.dumps(data,indent=2)+'\n')


def descriptor(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def build(run,node,model='35b'):
    if model=='122b':
        return build_large(run,node)
    prep=load('qualified_preparation',STEADY/'prepare.py')
    prep.build(run,node,'H100')
    cfg=json.loads((run/'config.json').read_text())
    spec=json.loads((run/'dashboard_spec.json').read_text())
    code=run/'code'
    runner=(A100/'runner.py').read_text()
    marker="if __name__ == '__main__':"
    assert runner.count(marker)==1
    runner=runner.replace(marker,'import optimization\noptimization.install(sys.modules[__name__])\n\n'+marker)
    runner=runner.replace("expected=2 * len(cfg['layouts'])","expected=cfg['expected_points']")
    (code/'runner.py').write_text(runner)
    (code/'runtime_adapter.py').write_text('''"""Native model mathematics plus private PP scheduling."""
def install():
    import h100_runtime_adapter
    import pp_schedule_opt
    h100_runtime_adapter.install()
    pp_schedule_opt.install()
''')
    for name in ('optimization.py','pp_schedule_opt.py'):
        shutil.copyfile(HERE/name,code/name)
    cfg.update(expected_points=5,port=21921,dist_port=21922,
        decode_opt_policy_path=str(run/'decode_opt_policy.bin'),
        protocol='Same GPU pair, real GPU-cached4096 / new1 / output128 / B8. '
          'Same-process balanced four-policy pilot, then independent three-pair confirmation. '
          'Native CUDA Graph, unchanged model math, separate profiles and token audit.')
    Path(cfg['decode_opt_policy_path']).write_bytes(b'0')
    os.chmod(cfg['decode_opt_policy_path'],0o600)
    cfg.pop('diagnostic_eager_policy_path',None)
    for path in code.glob('*.py'): os.chmod(path,0o600)
    save(run/'config.json',cfg)
    inputs=[r for r in spec['frozen_inputs'] if not Path(r['path']).is_relative_to(code)
            and r['path']!=str(run/'config.json')]
    inputs += [descriptor(p) for p in code.glob('*.py')]+[descriptor(run/'config.json')]
    spec.update(title='H100 PP2 Decode scheduling: same-process controlled optimization',
        port=21920,frozen_inputs=list({r['path']:r for r in inputs}.values()))
    spec['tasks'][0].update(label='EP2 reference, PP2 four policies; same GPUs0/1',expected=5)
    save(run/'dashboard_spec.json',spec)
    progress=json.loads((run/'task/progress.json').read_text())
    progress['expected']=5;save(run/'task/progress.json',progress)
    save(run/'optimization_manifest.json',dict(modes=['baseline','producer_ready','early_activation','early_and_ready'],
        pilot_order=['baseline','producer_ready','early_activation','early_and_ready',
                     'early_and_ready','early_activation','producer_ready','baseline'],
        confirmation='A/B/B/A/A/B, independent of pilot; all results retained',
        no_math_changes=True,no_ring_protocol_changes=True,pp_slots=2,requests_per_slot=4,
        correctness='All per-request output tokens retained; native A/A and A/B compared',
        modifications=[descriptor(HERE/'pp_schedule_opt.py'),descriptor(HERE/'optimization.py')]))
    return dict(prepared=True,run=str(run),node=node,expected_points=5)


def build_large(run,node):
    source=HERE.parent/'ep2_pp2_qwen_h100_20261025/runs/h100_ep2_pp2_r7'
    reference=HERE/'runs/35b_r1'
    # Reuse the qualified large model's exact model, native and cache protocol;
    # only reuse the new sequential controller/experiment adapter from 35B.
    original=json.loads((source/'config.json').read_text())
    jobs=load('current_booking_provider',A100/'prepare.py').old.booked_jobs()
    if jobs.get(node)!=original['job_id'] or node!=original['node']:
        raise RuntimeError('Qualified large-model bank must match the current booking')
    run.mkdir(mode=0o700,parents=True)
    for name in ('code','cache','task','ep2','pp2'): (run/name).mkdir(mode=0o700)
    def rewrite(value):
        if isinstance(value,str): return value.replace(str(source),str(run)).replace(original['id'],run.name)
        if isinstance(value,list): return [rewrite(x) for x in value]
        if isinstance(value,dict): return {k:rewrite(v) for k,v in value.items()}
        return value
    cfg=rewrite(original);spec=rewrite(json.loads((source/'dashboard_spec.json').read_text()))
    code=run/'code'
    for file in (source/'code').glob('*.py'): shutil.copyfile(file,code/file.name)
    shutil.copyfile(source/'code/runner.py',code/'hopper_reference.py')
    shutil.copyfile(source/'code/runtime_adapter.py',code/'h100_runtime_adapter.py')
    for name in ('runner.py','runtime_adapter.py','export_ready.py','optimization.py','pp_schedule_opt.py'):
        shutil.copyfile(reference/'code'/name,code/name)
    cfg.update(id=run.name,code_root=str(code),admin_key_file=str(run/'admin.key'),
        policy_path=str(run/'pp_ready_policy.bin'),decode_opt_policy_path=str(run/'decode_opt_policy.bin'),
        expected_points=5,port=22121,dist_port=22122,arm_order=['ep2','pp2'],
        requests=8,prefix_tokens=4096,output_tokens=128,repeats=3,
        protocol=json.loads((reference/'config.json').read_text())['protocol'])
    for field in ('policy_path','decode_opt_policy_path'):
        Path(cfg[field]).write_bytes(b'0');os.chmod(cfg[field],0o600)
    Path(cfg['admin_key_file']).write_text(secrets.token_hex(32)+'\n')
    os.chmod(cfg['admin_key_file'],0o600)
    node_spec=json.loads((source/'cache/node.json').read_text());save(run/'cache/node.json',node_spec)
    for row in cfg['layouts']:
        row['gpu_groups']=[[0,1]]
        for log in row['usage_logs']:
            path=Path(log['path']);path.touch(mode=0o600,exist_ok=False)
    contract=cfg['model_cache']
    contract['node_specs'][node]=descriptor(run/'cache/node.json')
    spec['model_cache']=contract
    for path in code.glob('*.py'): os.chmod(path,0o600)
    save(run/'config.json',cfg)
    # Historical external hashes are verified by Guard, not silently replaced.
    inputs=[x for x in json.loads((source/'dashboard_spec.json').read_text())['frozen_inputs']
            if not Path(x['path']).is_relative_to(source)]
    inputs += [descriptor(p) for p in code.glob('*.py')]
    inputs += [descriptor(run/'config.json'),descriptor(run/'cache/node.json')]
    spec.update(id=run.name,title='H100 122B PP2 Decode scheduling: controlled optimization',
        port=22120,runtime_dir=str(run/'dashboard_guard'),
        frozen_inputs=list({r['path']:r for r in inputs}.values()))
    spec['tasks'][0].update(label='122B EP2 then PP2; same GPUs0/1, four policies',expected=5)
    save(run/'dashboard_spec.json',spec)
    save(run/'task/progress.json',dict(experiment_id=run.name,task_id='comparison',stage='queued',
        valid_points=0,expected=5))
    oldprep=json.loads((source/'preparation.json').read_text())
    oldprep['command']=[cfg['python'],str(Path(cfg['guard_scripts'])/'dashboard_guard.py'),'run',
        '--spec',str(run/'dashboard_spec.json'),'--resource-choice','auto','--exit-after-run','--',
        cfg['python'],str(code/'runner.py'),'--run',str(run)]
    save(run/'preparation.json',oldprep)
    save(run/'optimization_manifest.json',dict(model='122B FP8',same_policy_as=str(reference),
        gpu_pair=[0,1],new_allocations=False,no_math_changes=True,pp_slots=2,requests_per_slot=4))
    return dict(prepared=True,run=str(run),node=node,model='122b',expected_points=5)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--node',default='spartan-gpgpu178')
    parser.add_argument('--model',choices=['35b','122b'],default='35b')
    args=parser.parse_args()
    print(json.dumps(build(args.run.absolute(),args.node,args.model)))
