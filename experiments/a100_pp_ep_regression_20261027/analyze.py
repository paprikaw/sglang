"""Recompute A100 timing and attribute eager GPU work via host-launch ancestry."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics as st

ROOT=Path(__file__).resolve().parent


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_csv(p,rows):
    if not rows:return
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def profile(path,arm,case):
    spec=importlib.util.spec_from_file_location('eager_tools',ROOT.parent/'pp_ep_steady_decode_20261026/analyze.py')
    tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)
    with gzip.open(path,'rt') as f:
        ev=[e for e in json.load(f)['traceEvents'] if e.get('ph')=='X' and e.get('dur',0)>0]
    assert not any(e['name']=='cudaGraphLaunch' for e in ev),'Unexpected Graph execution'
    ann=[e for e in ev if e.get('cat')=='user_annotation']
    steps=[e for e in ann if e['name'].startswith('step[')]
    si=tools.IntervalIndex(steps)
    layers=[e for e in ann if re.fullmatch(r'(?:component\.)?layer\.\d+',e['name'])]
    li=tools.IntervalIndex(layers)
    ci=tools.IntervalIndex([e for e in ann if e['name'].startswith('component.') and not e['name'].startswith('component.layer.')])
    grouped=defaultdict(list);step_by_id={}
    for layer in layers:
        step=si.find(layer['ts'],layer.get('tid'))
        wanted=(step is not None and (f"step[DECODE bs={case['requests']//2} " in step['name']
                if case['phase']=='decode' else ('EXTEND' in step['name'] or 'PREFILL' in step['name'])))
        if wanted:grouped[id(step)].append(layer);step_by_id[id(step)]=step
    chosen=sorted(grouped,key=lambda k:step_by_id[k]['ts'])
    if case['phase']=='decode':
        chosen=chosen[20:36];assert len(chosen)==16,(path,'insufficient late Decode forwards')
    assert chosen,(path,'no phase-qualified layer invocations')
    expected=40 if arm=='ep2' else 20
    assert all(len(grouped[k])==expected for k in chosen),(path,'incomplete layer coverage')
    selected={id(l) for k in chosen for l in grouped[k]}
    runtime={e.get('args',{}).get('correlation'):e for e in ev
        if e.get('cat') in ('cuda_runtime','cuda_driver') and e.get('args',{}).get('correlation') is not None}
    costs=defaultdict(lambda:defaultdict(list));kernel_count=defaultdict(int)
    for e in ev:
        if e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset'):continue
        launch=runtime.get(e.get('args',{}).get('correlation'))
        if launch is None:continue
        layer=li.find(launch['ts'],launch.get('tid'))
        if layer is None or id(layer) not in selected:continue
        scope=ci.find(launch['ts'],launch.get('tid'))
        label=tools.label(scope) if scope else 'other'
        name=e['name']
        if label=='expert.routed':
            bucket='expert_gemm' if 'fused_moe_kernel' in name else 'expert_other'
        elif label in ('communication.dispatch','communication.combine'):bucket=label.split('.')[-1]
        elif label.startswith('attention.'):bucket='attention'
        elif label=='expert.shared':bucket='shared_expert'
        else:bucket='other'
        costs[id(layer)][bucket].append((e['ts'],e['ts']+e['dur']));kernel_count[bucket]+=1
    rows=[]
    for k in chosen:
        row=defaultdict(float)
        for l in grouped[k]:
            for bucket,spans in costs[id(l)].items():row[bucket]+=tools.union_us(spans)/1000
        rows.append(row)
    result={k:st.mean(r.get(k,0) for r in rows)/expected for k in ('expert_gemm','expert_other','dispatch','combine','attention','shared_expert','other')}
    result.update(expert_total=result['expert_gemm']+result['expert_other'],dispatch_combine=result['dispatch']+result['combine'],
        selected_forwards=len(chosen),actual_layer_invocations=len(selected),kernel_count=sum(kernel_count.values()),
        unit='GPU union ms per card per actual layer invocation; separate profiled cohort; not additive E2E slices',
        trace=str(path),trace_sha256=sha(path))
    assert result['expert_gemm']>0 and result['attention']>0,(path,'missing compute attribution')
    if arm=='ep2':assert result['dispatch']>0 and result['combine']>0,(path,'missing DeepEP attribution')
    else:assert result['dispatch_combine']==0,(path,'EP1 unexpectedly exchanges Expert tokens')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--partial',action='store_true');ap.add_argument('--skip-profiles',action='store_true');a=ap.parse_args()
    run=a.run.absolute();cfg=read(run/'config.json');out=ROOT/'analysis'/run.name;out.mkdir(parents=True,exist_ok=True)
    audit=dict(partial=a.partial,stage=read(run/'task/progress.json')['stage'])
    if not a.partial:
        assert audit['stage']=='complete';assert read(run/'task/cleanup.json')['passed']
        checks=[read(p) for p in (run/'dashboard_guard/checks').glob('*-post.json')]
        assert len(checks)==1 and checks[0]['passed'] and checks[0]['cleanup_verified']
        assert read(run/'topology_qualification.json')['passed']
        assert read(run/'input_compatibility.json')['passed']
        for d in read(run/'dashboard_spec.json')['frozen_inputs']:assert sha(d['path'])==d['sha256'],d['path']
        audit.update(guard_postcheck=True,cleanup_verified=True,frozen_inputs_verified=True,topology=read(run/'topology_qualification.json'))
    metrics=[];components=[];sources=[];agreement=[]
    for arm in cfg['arm_order']:
        for case in cfg['workloads']:
            root=run/arm/'workloads'/case['id'];policies=[0,9] if arm=='pp2' and case['phase']=='decode' else [0]
            for p in policies:
                path=root/f"metrics.{case['phase']}.p{p}.json"
                if not path.exists():continue
                data=read(path);groups=data['rounds']
                if len(groups)!=5:
                    if a.partial:continue
                    raise AssertionError('Incomplete formal cohorts')
                rates=[];rows=[]
                for g in groups:
                    assert len(g['requests'])==case['requests']
                    count=128 if case['phase']=='decode' else 1
                    assert all(r['output_tokens']==count and len(r['output_ids'])==count for r in g['requests'])
                    assert all((r.get('cached_tokens') or 0)==(4096 if case['phase']=='decode' else 0) for r in g['requests'])
                    tokens=case['requests']*(128 if case['phase']=='decode' else case['new_input'])
                    rates.append(tokens/g['makespan_s']);rows+=g['requests']
                    if arm=='pp2' and p==9:assert all(x>0 for x in g['scheduling_receipt']['ahead_deltas'])
                metrics.append(dict(workload=case['id'],phase=case['phase'],family=case['family'],arm=arm,policy=p,
                    batch=case['requests'],context=4096 if case['phase']=='decode' else 0,
                    new_input=1 if case['phase']=='decode' else case['new_input'],output_tokens=count,
                    throughput_tok_s=st.mean(rates),throughput_stdev=st.stdev(rates),
                    cohort_ms=st.mean(g['makespan_s'] for g in groups)*1000,
                    e2e_ms=st.mean(r['e2e_s'] for r in rows)*1000,ttft_ms=st.mean(r['ttft_s'] for r in rows)*1000,
                    tpot_ms=st.mean(r['tpot_ms'] for r in rows) if case['phase']=='decode' else None,
                    formal_cohorts=5,source=str(path),sha256=sha(path)))
                sources.append(dict(path=str(path),sha256=sha(path)))
                if not a.skip_profiles:
                    manifest=root/'traces'/f'p{p}'/'manifest.json'
                    if not manifest.exists():
                        assert a.partial;continue
                    exports=read(manifest)['files']
                    identities=[int(re.search(r'PP-(\d+)' if arm=='pp2' else r'TP-(\d+)',Path(t['path']).name)[1]) for t in exports]
                    assert len(exports)==2 and set(identities)=={0,1},manifest
                    for t,rank in zip(exports,identities):
                        trace=Path(t['path']);assert sha(trace)==t['sha256']
                        components.append(dict(workload=case['id'],phase=case['phase'],arm=arm,policy=p,rank=rank,**profile(trace,arm,case)))
                    assert len(read(manifest)['files'])==2
            p=root/'output_agreement.json'
            if p.exists():agreement.append(dict(workload=case['id'],arm=arm,**read(p)))
    comparisons=[]
    for case in cfg['workloads']:
        rows=[r for r in metrics if r['workload']==case['id']]
        ep=next((r for r in rows if r['arm']=='ep2'),None)
        for pp in [r for r in rows if r['arm']=='pp2']:
            if ep:comparisons.append(dict(workload=case['id'],phase=case['phase'],policy=pp['policy'],
                ep_throughput=ep['throughput_tok_s'],pp_throughput=pp['throughput_tok_s'],
                pp_vs_ep_pct=(pp['throughput_tok_s']/ep['throughput_tok_s']-1)*100))
    # EP/PP equality is reported separately from the stricter policy0/policy9
    # equivalence: layout changes may alter BF16 reductions and later routing.
    cross=[]
    for case in cfg['workloads']:
        outputs={}
        for arm in cfg['arm_order']:
            p=run/arm/'workloads'/case['id']/f"metrics.{case['phase']}.p0.json"
            if p.exists():outputs[arm]=[r['output_ids'] for r in read(p)['rounds'][0]['requests']]
        if len(outputs)==2:
            cross.append(dict(workload=case['id'],equal_requests=sum(x==y for x,y in zip(outputs['ep2'],outputs['pp2'])),total_requests=case['requests']))
    result=dict(run=str(run),audit=audit,metrics=metrics,components=components,comparisons=comparisons,
        output_agreement=agreement,cross_layout_output_agreement=cross,sources=sources)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in [('metrics',metrics),('components',components),('comparisons',comparisons)]:write_csv(out/(name+'.csv'),rows)
    print(json.dumps(dict(out=str(out),points=len(metrics),profiles=len(components),comparisons=comparisons),indent=2))


if __name__=='__main__':main()
