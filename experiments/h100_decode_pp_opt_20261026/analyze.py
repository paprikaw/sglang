"""Offline native Graph timing attribution; never imports Torch or uses GPUs."""
from bisect import bisect_right
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def save(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def union(intervals):
    total=0.;edge=None
    for lo,hi in sorted(intervals):
        total+=hi-lo if edge is None or lo>edge else max(0.,hi-edge)
        edge=hi if edge is None else max(edge,hi)
    return total


def intersection_union(left, right):
    """GPU interval overlap, not a claim about SM occupancy or causation."""
    return union([(max(a,c),min(b,d)) for a,b in left for c,d in right
                  if max(a,c)<min(b,d)])


def bucket(name):
    if 'flash::FlashAttnFwd' in name or 'flash::prepare_varlen_num_blocks_kernel' in name or 'fused_recurrent_gated_delta_rule' in name:
        return 'attention_core'
    if 'deep_gemm::' in name and '(deep_gemm::GemmType)2' in name: return 'routed_gemm'
    if 'deep_ep::' in name:
        return 'dispatch' if 'dispatch' in name else 'combine' if 'combine' in name else 'deepep_other'
    if 'nccl' in name.lower():return 'nccl'
    return 'other'


def profile(path,arm,model):
    with gzip.open(path,'rt') as handle: data=json.load(handle)
    events=[x for x in data['traceEvents'] if x.get('ph')=='X' and x.get('dur',0)>0]
    steps=sorted([x for x in events if x.get('cat')=='user_annotation' and x['name'].startswith('step[')],key=lambda x:x['ts'])
    starts=[x['ts'] for x in steps]
    groups=defaultdict(list)
    for event in events:
        if event.get('cat')=='kernel': groups[event.get('args',{}).get('correlation')].append(event)
    candidates=[]
    for launch in events:
        if launch['name']!='cudaGraphLaunch': continue
        index=bisect_right(starts,launch['ts'])-1
        if index<0:continue
        step=steps[index]
        kernels=groups.get(launch.get('args',{}).get('correlation'),[])
        if 'step[DECODE bs=4 ' not in step['name'] or not kernels:continue
        if not step['ts']<=launch['ts']<step['ts']+step['dur']: continue
        first=min(x['ts'] for x in kernels);last=max(x['ts']+x['dur'] for x in kernels)
        candidates.append((first,last,kernels))
    candidates.sort(key=lambda x:x[0])
    chosen=candidates[4:28]
    if len(chosen)!=24:raise AssertionError((path,len(candidates),'Need 24 warmed native Graphs'))
    nccl_kernels=[x for x in events if x.get('cat')=='kernel' and bucket(x['name'])=='nccl']
    rows=[]
    for i,(first,last,kernels) in enumerate(chosen):
        costs=defaultdict(list)
        for k in kernels: costs[bucket(k['name'])].append((k['ts'],k['ts']+k['dur']))
        row={key+'_union_ms':union(costs[key])/1000 for key in
             ('attention_core','routed_gemm','dispatch','combine','deepep_other','nccl','other')}
        row['graph_span_ms']=(last-first)/1000
        row['graph_active_union_ms']=union([(k['ts'],k['ts']+k['dur']) for k in kernels])/1000
        devices={k.get('args',{}).get('device') for k in kernels}
        nccl=[(k['ts'],k['ts']+k['dur']) for k in nccl_kernels
              if k.get('args',{}).get('device') in devices
              and k['ts']<last and k['ts']+k['dur']>first]
        row['nccl_during_graph_span_ms']=intersection_union([(first,last)],nccl)/1000
        row['routed_gemm_nccl_overlap_ms']=intersection_union(costs['routed_gemm'],nccl)/1000
        if i+1<len(chosen):
            period=(chosen[i+1][0]-first)/1000
            row['gpu_graph_start_period_ms']=period
            row['between_graph_spans_ms']=period-row['graph_span_ms']
        rows.append(row)
    result=dict(model=model,arm=arm,rank_trace=path.name,trace=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),steady_graphs=len(rows),
        unit='ms/card/actual native Graph, local batch4; GPU-clock periods within one rank')
    for key in dict.fromkeys(k for r in rows for k in r): result[key]=statistics.mean(r[key] for r in rows if key in r)
    result['category_scope']='Attention CORE and routed GEMM only; other includes projections/shared/routing. Inter-Graph span includes communication and dependencies, not pure idle.'
    return result


def main():
    out=HERE/'analysis';out.mkdir(exist_ok=True)
    metrics=[];profiles=[];sources=[];audits=[]
    for model in ('35b','122b'):
        run=HERE/'runs'/(model+'_r1')
        if not (run/'task/cleanup.json').exists(): continue
        cleanup=json.loads((run/'task/cleanup.json').read_text())
        if not cleanup.get('passed'):raise RuntimeError('Cleanup was not verified')
        selection=json.loads((run/'pilot_selection.json').read_text())
        for arm in ('ep2','pp2'):
            for path in (run/arm).glob('metrics.*.json'):
                data=json.loads(path.read_text())
                metrics.append(dict(model=model,arm=arm,mode=path.stem.split('.')[-1],
                    source=str(path),rounds=len(data['rounds']),confirmation=data.get('confirmation',True),
                    output_tokens_per_s=data['output_tokens_per_s'],
                    ttft_ms=data['ttft_s']['mean']*1000,tpot_ms=data['tpot_ms']['mean'],
                    e2e_ms=data['e2e_s']['mean']*1000))
            for directory in (run/arm/'traces').iterdir():
                # The producer's existing export manifest was awaited under Guard.
                manifest=json.loads((directory/'manifest.json').read_text())
                if len(manifest['files'])!=2:raise AssertionError('Both rank exports required')
                for path in directory.glob('*.trace.json.gz'):
                    profiles.append(dict(mode=directory.name,**profile(path,arm,model)))
        agreement=json.loads((run/'pp2/output_agreement.json').read_text())
        agreement.pop('outputs_by_mode',None)
        audits.append(dict(model=model,cleanup=cleanup,output_agreement=agreement,
            selected_candidate=selection['candidate'],candidate_promoted=False,
            reason='Raw speed result only until native A/A output variability is resolved or bounded.'))
        for relative in ('config.json','dashboard_spec.json','pilot_selection.json','pp2/output_agreement.json','task/cleanup.json'):
            path=run/relative;sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for name,rows in [('metrics',metrics),('profiles',profiles)]:
        with (out/(name+'.csv')).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(out/'audit.json',dict(runs=audits,profile_rank_rows=len(profiles),sources=sources))
    print(json.dumps(dict(metrics=len(metrics),profiles=len(profiles),models=[r['model'] for r in audits])))


if __name__=='__main__': main()
