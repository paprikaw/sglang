"""Clone qualified inputs into a fresh immutable optimization run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'ep2_pp2_qwen35b_h100_20261025/runs/h100_qwen35b_ep2_pp2_r3'
GUARD=Path('/home/bxb1/.codex/skills/llm-experiment-guard/scripts')


def build(run):
    run=run.resolve(); run.mkdir(parents=True,exist_ok=False); os.chmod(run,0o700)
    cfg=json.loads((SOURCE/'config.json').read_text())
    spec=json.loads((SOURCE/'dashboard_spec.json').read_text())
    def fresh(x):
        if isinstance(x,str):
            return x.replace(str(SOURCE),str(run)).replace(SOURCE.name,run.name)
        if isinstance(x,list):return [fresh(v) for v in x]
        if isinstance(x,dict):return {k:fresh(v) for k,v in x.items()}
        return x
    cfg=fresh(cfg);spec=fresh(spec)
    shutil.copytree(SOURCE/'code',run/'code',ignore=shutil.ignore_patterns('__pycache__'))
    for folder in ('cache','task','pp2','ep2'):(run/folder).mkdir(mode=0o700)
    shutil.copyfile(SOURCE/'cache/node.json',run/'cache/node.json')
    # Reuse the private credential file; never emit or copy its contents.
    cfg['admin_key_file']=json.loads((SOURCE/'config.json').read_text())['admin_key_file']
    cfg.update(port=20131,dist_port=20141,expected_points=18,
        coalesce_policy_path=f'/tmp/{run.name}-coalesce',
        optimization_cases=[
            dict(id='short_cached32',requests=32,cached_tokens=4096,new_tokens=512),
            dict(id='long_cached32',requests=32,cached_tokens=89984,new_tokens=512),
            dict(id='cold8',requests=8,cached_tokens=0,new_tokens=4096)],
        protocol='PP2 admission baseline/candidate ABBA, bounded 5ms coalescing included in TTFT; '
                 'same-process, exact-shape warmup and independently reset genuine GPU prefixes. '
                 'EP2 reference uses same cohort and cache state. No Decode or CPU restore timed.')
    cfg['layouts']=[r for r in cfg['layouts'] if r['id']!='single']
    uses=[r['usage_logs'][0] for r in cfg['layouts']]
    for use in uses:
        fd=os.open(use['path'],os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    def descriptor(path):
        path=Path(path); return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for contract in (cfg['model_cache'],spec['model_cache']):
        contract['usage_logs'][cfg['node']]=uses
        contract['node_specs'][cfg['node']]=descriptor(run/'cache/node.json')
    cfg['launch_env']['PP_COALESCE_POLICY_PATH']=cfg['coalesce_policy_path']
    for name in ('coalesce.py','optimization.py'):
        shutil.copyfile(ROOT/name,run/'code'/name)
    adapter=run/'code/runtime_adapter.py'
    code=adapter.read_text().replace('    pp_ready_send.install(torch)',
        '    pp_ready_send.install(torch)\n    import coalesce\n    coalesce.install()')
    adapter.write_text(code)
    runner=run/'code/runner.py';code=runner.read_text()
    code=code.replace("expected=2 if profile_only else 12", "expected=cfg.get('expected_points', 12)")
    code=code.replace("expected=12, **extra", "expected=cfg.get('expected_points', 12), **extra")
    code=code.replace("if __name__ == '__main__':", "from optimization import worker\n\nif __name__ == '__main__':")
    runner.write_text(code)
    (run/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');os.chmod(run/'config.json',0o600)
    # Keep unchanged engine/model/native descriptors, but rehash every private file.
    original_inputs=json.loads((SOURCE/'dashboard_spec.json').read_text())['frozen_inputs']
    inputs=[x for x in original_inputs if not Path(x['path']).is_relative_to(SOURCE)]
    inputs += [descriptor(p) for p in (run/'code').glob('*.py')]
    inputs += [descriptor(run/'config.json'),descriptor(run/'cache/node.json')]
    spec.update(port=20130,title='H100 Qwen35B: bounded PP Prefill admission ABBA',
                frozen_inputs=list({x['path']:x for x in inputs}.values()))
    spec['tasks'][0].update(expected=18,label='PP2 admission ABBA + EP2 reference; cached and cold Prefill')
    (run/'dashboard_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    os.chmod(run/'dashboard_spec.json',0o600)
    preparation=json.loads((SOURCE/'preparation.json').read_text())
    command=[cfg['python'],str(Path(cfg['guard_scripts'])/'dashboard_guard.py'),'run','--spec',str(run/'dashboard_spec.json'),
             '--resource-choice','auto','--exit-after-run','--',cfg['python'],str(runner),'--run',str(run)]
    preparation.update(command=command)
    (run/'preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
    print(json.dumps(dict(run=str(run),command=command),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    build(a.run)
