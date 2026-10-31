"""Freeze a fresh, Guard-admitted P/D arrival experiment; never launch a GPU here."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[1]
OLD = WS / 'experiments/h100_decode_pool_necessity_20261029'

def read(p): return json.loads(p.read_text())
def save(p,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,indent=2)+'\n')
def desc(p): return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--node',required=True)
    ap.add_argument('--job',required=True)
    ap.add_argument('--port',type=int,required=True)
    ap.add_argument('--layouts',default='ep4,two_ep2,four_ep1,two_pp2,mixed')
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--cases',default='all')
    ap.add_argument('--controlled-qualification',action='store_true')
    ap.add_argument('--phase-isolation',action='store_true')
    ap.add_argument('--pp-microbatch',type=int,choices=(4,8,16),default=8)
    a=ap.parse_args(); run=a.run.resolve()
    assert a.repeats>=3 and not run.exists()
    # Reuse the qualified source/cache ownership contract, including fresh
    # canonical backfill reconciliation, commit/tag and native hash checks.
    spec=importlib.util.spec_from_file_location('prior_prepare',OLD/'prepare.py')
    prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    old_argv=sys.argv
    sys.argv=[str(OLD/'prepare.py'),'--run',str(run),'--node',a.node,'--job',a.job,
              '--dashboard-port',str(a.port),'--profiles','short','--layouts','ep4',
              '--cases','p4096_n8_repeated','--repeats',str(a.repeats)]
    try:prior.main()
    finally:sys.argv=old_argv
    cfg=read(run/'config.json')
    configs={'ep4':[(4,1,[0,1,2,3])], 'two_ep2':[(2,1,[0,1]),(2,1,[2,3])],
             'four_ep1':[(1,1,[i]) for i in range(4)],
             'two_pp2':[(1,2,[0,1]),(1,2,[2,3])],
             'mixed':[(2,1,[0,1]),(1,2,[2,3])]}
    layouts=[]
    for name in a.layouts.split(','):
        instances=[];logs=[]
        for i,(ep,pp,gpus) in enumerate(configs[name]):
            instances.append(dict(ep=ep,tp=ep,dp=ep,pp=pp,gpus=gpus))
            logs.append(dict(id=f'{run.name}.{name}.r{i}',path=str(run/'cache'/f'{name}.r{i}.log'),
                expected_ranks=[f'PP{s} TP0' for s in range(pp)] if pp>1 else [f'TP{r}' for r in range(ep)]))
        assert sorted(g for s in instances for g in s['gpus'])==[0,1,2,3]
        layouts.append(dict(id=name,kind=name,replicas=len(instances),instances=instances,usage_logs=logs))
    cases=[dict(id=f'cold_b{n}_q{q}',kind='cold_prefill',requests=n,Q=q,O=1,rps=0)
           for n,q in [(32,512),(16,8192),(8,16384)]]
    for fraction,label in [(0.75,'pheavy'),(.5,'balanced'),(.25,'dheavy')]:
        for rps in (16,32):
            cases.append(dict(id=f'{label}_rps{rps}',kind='mixed_arrivals',requests=64,
                              p_fraction=fraction,rps=rps,P_Q=8192,P_O=16,D_Q=256,D_O=256))
    if a.cases!='all':
        wanted=a.cases.split(',');assert set(wanted)<=set(c['id'] for c in cases)
        cases=[c for c in cases if c['id'] in wanted]
    cfg.update(layouts=layouts,workloads=cases,repeats=a.repeats,
        expected_points=sum((3 if row['id']=='mixed' else
                              2 if a.phase_isolation and any(s['pp']>1 for s in row['instances']) and row['replicas']>1 else 1)
                            if c['kind']=='mixed_arrivals' else 1
                            for row in layouts for c in cases),
        port=28121,dist_port=31121,max_context=32768,requests=64,
        profile=dict(max_context=32768,max_requests=64,mamba_equiv=80,kv_token_cap_per_full_gpu=147456),
        pp_stream_microbatch=a.pp_microbatch,pp_candidate_policy=1,
        controlled_qualification=a.controlled_qualification,phase_isolation=a.phase_isolation,
        protocol='Complete Qwen3.6-35B-A3B BF16, four H100 GPUs per deployment on one node. '
        'Cold P=0 synthetic unique prompts; isolated Prefill plus real open-loop mixed P/D request arrivals. '
        'Each request completes P and D on its selected instance. No state migration, HiCache, CPU KV restore, '
        'weight repartitioning, or online resizing. CUDA Graph Decode and DeepEP AUTO. '
        'PP2 variable-message native transport with qualified early activation send (policy1); '
        'fixed-schema policy9 is inapplicable to this arrival protocol. Three profiler-off full-trace repeats. '
        'Report arithmetic mean and retain all raw requests, TTFT, TPOT, completion latency, client arrival lag.')
    cfg['launch_env']['PP_GAP_SEPARATE_CHANNELS']='0'
    if a.controlled_qualification:
        cfg['protocol']+=' Follow-up qualification uses controlled initial PP cohorts with unequal prompt/output lengths; '
        cfg['protocol']+='exact native/early-send token agreement is required with the same admission order. '
        cfg['protocol']+='Free-arrival output differences are recorded separately because the batching can differ.'
    cfg['protocol']+=f' This run freezes the streaming PP microbatch upper bound at {a.pp_microbatch} per slot.'
    cfg['model_cache']['usage_logs']={a.node:[u for row in layouts for u in row['usage_logs']]}
    cfg['primary_hypothesis']='A PP-containing equal-GPU deployment enters the best-layout envelope for some P/D mixtures, while a non-PP deployment wins another; a resident mixed pool may improve actual trace latency/goodput.'
    cfg['slo_sensitivity']=dict(ttft_ms=[200,500,1000,2000],tpot_ms=[10,15,20,30])
    cfg['interest_thresholds']=dict(two_sided_layout_advantage_pct=20,mixed_pool_goodput_gain_pct=10,
        interpretation='Engineering screening thresholds, not statistical significance tests.')
    shutil.copyfile(ROOT/'mixed_workload.py',run/'code/mixed_workload.py')
    runner=run/'code/runner.py';s=runner.read_text()
    assert 'import verify_workload\nverify_workload.install(sys.modules[__name__])' in s
    s=s.replace('import verify_workload\nverify_workload.install(sys.modules[__name__])',
                'import mixed_workload\nmixed_workload.install(sys.modules[__name__])')
    runner.write_text(s)
    # Additional read-only phase/backend counters; no device reads or sync.
    observer=run/'code/forward_observer.py';s=observer.read_text()
    s=s.replace('ModelRunner.forward=forward', '''ModelRunner.forward=forward
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher
    dispatch=DeepEPDispatcher.dispatch
    @wraps(dispatch)
    def count_dispatch(self,*args,**kwargs):
        result=dispatch(self,*args,**kwargs)
        _COUNTS['dispatch_output:'+type(result).__name__]+=1
        return result
    DeepEPDispatcher.dispatch=count_dispatch''')
    observer.write_text(s)
    if a.controlled_qualification:
        gap=run/'code/gap_schedule.py';s=gap.read_text()
        marker='self._gap_proxy_sent.append((batch,result))'
        assert s.count(marker)==1
        s=s.replace(marker,marker+'\n            self._gap_early_sent=getattr(self,"_gap_early_sent",0)+1')
        s=s.replace('policy=mode(), produced=', 'policy=mode(), early_sent=getattr(self,"_gap_early_sent",0), produced=')
        gap.write_text(s)
    save(run/'config.json',cfg)
    spec=read(run/'dashboard_spec.json')
    spec.update(title='35B four-H100 Prefill/Decode mixture and resident EP/PP pools',model_cache=cfg['model_cache'])
    spec['tasks'][0].update(expected=cfg['expected_points'],label='Complete measured arrival traces / isolated Prefill')
    files=[Path(d['path']) for d in spec['frozen_inputs']]+[run/'code/mixed_workload.py',ROOT/'PROTOCOL.md']
    if a.controlled_qualification or a.phase_isolation:files.append(ROOT/'CONTROLLED_QUALIFICATION.md')
    if a.pp_microbatch!=8:files.append(ROOT/'MICROBATCH_CAPACITY.md')
    spec['frozen_inputs']=list({str(p):desc(p) for p in files}.values())
    save(run/'dashboard_spec.json',spec)
    manifest=read(run/'launch_manifest.json')
    manifest.update(protocol=cfg['protocol'],layouts=layouts,workloads=cases,
        frozen_inputs=spec['frozen_inputs'],primary_hypothesis=cfg['primary_hypothesis'],
        decision_rule='Compare all full-budget baselines on the same node and arrival trace; '
        'evaluate phase affinity and least-outstanding routing in a resident PP2+EP2 pool. '
        'No claim of dynamic resizing without measured transition cost and online control.')
    save(run/'launch_manifest.json',manifest)
    save(run/'task/progress.json',dict(experiment_id=cfg['id'],task_id='comparison',timestamp=time.time(),stage='queued',valid_points=0,expected=cfg['expected_points']))
    for p in (run/'code').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    print(json.dumps(dict(run=str(run),layouts=[r['id'] for r in layouts],points=cfg['expected_points'],frozen_files=len(spec['frozen_inputs']))))

if __name__=='__main__':main()
