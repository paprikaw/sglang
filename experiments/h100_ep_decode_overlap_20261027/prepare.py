"""Freeze one static graph policy per independent Guard-owned deployment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT.parents[1] / ".engine-worktrees/qwen35b-decode-ep-overlap-20261027"
TAG = "experiment/sglang/qwen35b-decode-ep-overlap-20261027-r1-cpu"

def read(p): return json.loads(p.read_text())
def save(p,v): p.write_text(json.dumps(v,indent=2)+"\n")
def descriptor(p): return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--variant',choices=['native','serial','overlap','pp'],required=True)
    ap.add_argument('--focus',action='store_true')
    a=ap.parse_args();run=a.run.resolve()
    subprocess.run([sys.executable,'/home/bxb1/.codex/skills/llm-experiment-guard/scripts/backfill_trigger.py'],
                   check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(ROOT.parent/'h100_pp_lookahead_workloads_20261027/prepare.py'),
                    '--run',str(run)],check=True,stdout=subprocess.DEVNULL)
    cfg=read(run/'config.json');old=Path(cfg['engine_root']);code=run/'code'
    commit=subprocess.check_output(['git','-C',str(ENGINE),'rev-parse','HEAD'],text=True).strip()
    assert subprocess.check_output(['git','-C',str(ENGINE),'rev-parse',TAG+'^{commit}'],text=True).strip()==commit
    arm='pp2' if a.variant=='pp' else 'ep2'
    mode={'native':0,'serial':1,'overlap':2,'pp':0}[a.variant]
    cfg.update(engine_root=str(ENGINE),engine_commit=commit,engine_tag=TAG,
        variant=a.variant,decode_overlap_policy=mode,port=27421,dist_port=27422,
        arm_order=[arm],layouts=[r for r in cfg['layouts'] if r['id']==arm],
        protocol='H100178 NVLink GPU0/1, complete Qwen3.6-35B-A3B BF16. Same B8/B16/B32 repeated and B32 diverse prompts; '
                 'GPUcached4096+new1 -> output128. Same generic Expert Triton family on both layouts. '
                 'Static per-process Decode graph policy: EP0 native, EP1 serial split, EP2 overlapping split; '
                 'PP two fixed B/2 slots with policy9 lookahead. Whole Attention/GDN; shared-expert overlap ON. '
                 'Three full warmup cohorts and five profiler-off formal cohorts; independent profile. '
                 'Real-weight forty-layer/rank numerical qualification outside timing; no CPU KV restore.')
    if a.focus: cfg['workloads']=[r for r in cfg['workloads'] if r['id']=='b32_repeated']
    cfg['expected_points']=len(cfg['workloads'])
    cfg['model_cache']['usage_logs'][cfg['node']]=cfg['layouts'][0]['usage_logs']
    cfg['launch_env']['EP_DECODE_OVERLAP_POLICY']=str(mode)
    (run/'decode_overlap_verify.bin').write_bytes(b'0')
    (run/'decode_overlap_verify.bin').chmod(0o600)
    shutil.copyfile(ROOT/'decode_benchmark.py',code/'decode_benchmark.py')
    adapter=code/'runtime_adapter.py'
    adapter.write_text(adapter.read_text()+'''
_before_decode_overlap = install
def install():
    _before_decode_overlap()
    from sglang.srt.layers.moe import decode_overlap
    decode_overlap.install()
''')
    runner=code/'runner.py';text=runner.read_text()
    marker="if __name__ == '__main__':"
    assert text.count(marker)==1
    runner.write_text(text.replace(marker,'import decode_benchmark\ndecode_benchmark.install(sys.modules[__name__])\n\n'+marker))
    save(run/'config.json',cfg)
    spec=read(run/'dashboard_spec.json');inputs=[]
    for d in spec['frozen_inputs']:
        p=Path(d['path'])
        if p.is_relative_to(code) or p==run/'config.json':continue
        if p.is_relative_to(old):p=ENGINE/p.relative_to(old)
        inputs.append(descriptor(p))
    inputs.extend(descriptor(p) for p in sorted(code.glob('*.py')))
    inputs.extend([descriptor(run/'config.json'),descriptor(ENGINE/'python/sglang/srt/layers/moe/decode_overlap.py'),
                   descriptor(ENGINE/'python/sglang/srt/layers/moe/prefill_overlap.py')])
    spec.update(title='H100 Decode MoE overlap: '+a.variant,port=27420,
        sources=spec['sources']+[dict(worktree=str(ENGINE),commit=commit,tag=TAG)],
        model_cache=cfg['model_cache'],frozen_inputs=list({d['path']:d for d in inputs}.values()))
    spec['tasks'][0].update(expected=cfg['expected_points'],label='Cached Decode, static '+a.variant+' graph')
    save(run/'dashboard_spec.json',spec)
    manifest=read(run/'launch_manifest.json')
    manifest.update(engine_root=str(ENGINE),engine_commit=commit,engine_source_tag=TAG,
        private_adapters=[descriptor(p) for p in sorted(code.glob('*.py'))],
        workloads=cfg['workloads'],protocol=cfg['protocol'],decode_overlap_policy=mode,
        decode_overlap_source=descriptor(ENGINE/'python/sglang/srt/layers/moe/decode_overlap.py'),
        graph_policy_fixed_before_capture=True,variant=a.variant)
    save(run/'launch_manifest.json',manifest)
    save(run/'task/progress.json',dict(experiment_id=run.name,task_id='comparison',stage='queued',valid_points=0,expected=cfg['expected_points']))
    print(json.dumps(dict(run=str(run),variant=a.variant,commit=commit,points=cfg['expected_points'])))

if __name__=='__main__':main()
