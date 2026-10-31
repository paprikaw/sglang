"""Read-only exact-name MTP CUDA activity attribution; no kernel-name guesses."""
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'glm47_a100_cached_decode_stage_profile_20261022/analysis'))
from kineto_summary import ScopeIndex,arg,interval,RUNTIME_CATEGORIES

def read(path):return json.loads(path.read_text())

def calibrate(folder):
    labels=defaultdict(set);exposures=defaultdict(set);sources=[]
    for path in sorted(folder.glob('calibration_rank*_graph*_warm*.json')):
        trace=read(path)['traceEvents'];runtime=defaultdict(list)
        scopes=[]
        for i,e in enumerate(trace):
            if e.get('ph')!='X' or 'ts' not in e or 'dur' not in e:continue
            if e.get('cat') in RUNTIME_CATEGORIES:
                runtime[arg(e,'correlation','correlation id')].append(e)
            if e.get('name','').startswith('mtp-cal:'):
                e=dict(e,name='attr::'+e['name'].split(':',1)[1]+'::')
                scopes.append((i,e))
        index=ScopeIndex(scopes)
        for e in trace:
            if e.get('cat')!='kernel':continue
            calls=runtime.get(arg(e,'correlation','correlation id'),[])
            if len(calls)!=1:continue
            containers=index.containing(calls[0])
            if not containers:continue
            label=containers[0]['row']['name'].split('::')[1]
            labels[e['name']].add(label);exposures[e['name']].add(path.name)
        sources.append(str(path))
    known={name:next(iter(values)) for name,values in labels.items()
           if len(values)==1 and any('warm0' in p for p in exposures[name]) and any('warm1' in p for p in exposures[name])}
    # PP SendRecv has independent repeated CPU-API-scoped calibration from the
    # same torch/NCCL binary and exact engine PP transport (unchanged here).
    old=ROOT.parent/'glm47_a100_cached_decode_gpu_activity_20261022'/('calibration_'+folder.name+'.json')
    if old.exists():
        for name,value in read(old)['exact_names'].items():
            if value=='pp_communication' and name not in labels:known[name]=value
        sources.append(str(old))
    return known,dict(sources=sources,accepted_exact_names=known,
        ambiguous_or_unvalidated={name:sorted(values) for name,values in labels.items() if name not in known})

def partition(events,low,high,known,marker):
    changes=defaultdict(Counter)
    for event in events:
        if event.get('cat') not in ('kernel','gpu_memcpy','gpu_memset'):continue
        start,end=interval(event);start=max(start,low);end=min(end,high)
        if start>=end:continue
        name=event.get('name','')
        label=('marker' if name==marker else known.get(name,'unknown_gpu')) if event.get('cat')=='kernel' else event['cat']
        changes[start][label]+=1;changes[end][label]-=1
    active=Counter();last=low;total=Counter()
    for point,delta in sorted(changes.items()):
        labels={k for k,v in active.items() if v>0}
        label=next(iter(labels)) if len(labels)==1 else 'overlap_multiple_categories' if labels else 'no_traced_gpu_activity'
        total[label]+=point-last;active.update(delta);last=point
    total['no_traced_gpu_activity']+=high-last
    if abs(sum(total.values())-(high-low))>1e-3:raise ValueError('Timeline partition is not exhaustive')
    return {key:value/1000 for key,value in total.items()}

def rank_result(folder,rank,known):
    receipt=read(folder/f'mtp_activity_receipt_rank{rank}.json')
    signature=read(folder/f'mtp_marker_signature_rank{rank}.json')['name']
    events=read(Path(receipt['trace']))['traceEvents']
    markers=sorted((e for e in events if e.get('cat')=='kernel' and e.get('name')==signature),key=lambda e:e['ts'])
    if not receipt['passed'] or len(markers)!=2*len(receipt['segments']):
        raise ValueError(f'Unmatched actual GPU markers rank {rank}: {len(markers)} / {len(receipt["segments"])}')
    rows=[]
    for i,metadata in enumerate(receipt['segments']):
        low=markers[2*i]['ts']+markers[2*i]['dur'];high=markers[2*i+1]['ts']
        if low>=high:raise ValueError('Reversed marker order')
        rows.append(dict(**metadata,envelope_ms=(high-low)/1000,
                         partition_ms=partition(events,low,high,known,signature)))
    summary={}
    for phase in sorted({r['phase'] for r in rows}):
        selected=[r for r in rows if r['phase']==phase]
        # Drop the first and last occurrence to avoid profile startup and edge
        # effects; report the actual local B, never call PP's two B4 slots B8.
        selected=selected[1:-1]
        if not selected:continue
        keys={k for r in selected for k in r['partition_ms']}
        summary[phase]=dict(calls=len(selected),batch_sizes=sorted({r['batch_size'] for r in selected}),
            mean_envelope_ms=statistics.mean(r['envelope_ms'] for r in selected),
            median_envelope_ms=statistics.median(r['envelope_ms'] for r in selected),
            mean_partition_ms={k:statistics.mean(r['partition_ms'].get(k,0) for r in selected) for k in keys})
    return dict(rank=rank,phases=summary,segments=rows,
                full_profile_window_partition_ms=partition(events,markers[0]['ts'],markers[-1]['ts']+markers[-1]['dur'],known,signature))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);args=parser.parse_args()
    output={}
    for arm in ('ep8','pp2ep4'):
        folder=args.run/arm;known,audit=calibrate(folder)
        output[arm]=dict(calibration=audit,ranks=[rank_result(folder,r,known) for r in range(8)])
    output['semantics']='Each ms value is one local API GPU timeline partition, not all-rank sums or TPOT. Dispatch/combine includes GPU waiting, not pure wire time. Unknown and idle remain explicit. Draft phases include their own A/E/communication. Production performance was measured in a preceding unprofiled cohort in the same process.'
    # Every TP group duplicates the local layer execution. Only one
    # representative rank per group may enter a serial PP-stage sum.
    pp=output['pp2ep4']['ranks']
    by_rank={r['rank']:r for r in pp}
    def keyed(rank):
        rows={}
        for row in by_rank[rank]['segments']:
            if row['phase']=='target_verify' and row.get('request_ids'):
                key=(tuple(row['request_ids']),row['cohort_iteration'])
                rows[key]=row
        return rows
    first,last=keyed(0),keyed(4)
    pairs=[]
    for key in sorted(first.keys()&last.keys()):
        if key[1]==0:continue
        a,b=first[key],last[key]
        keys=set(a['partition_ms'])|set(b['partition_ms'])
        pairs.append(dict(request_ids=list(key[0]),cohort_iteration=key[1],
             local_batch_size=a['batch_size'],
             serial_target_stage_envelope_sum_ms=a['envelope_ms']+b['envelope_ms'],
             serial_target_stage_partition_sum_ms={k:a['partition_ms'].get(k,0)+b['partition_ms'].get(k,0) for k in keys}))
    output['pp2ep4']['matched_serial_target_stage_sums']=dict(
         meaning='Same request IDs and local verify ordinal, rank0 plus rank4. Excludes PP boundary and draft APIs; not TPOT or an all-rank GPU sum.',
         pairs=pairs)
    path=args.run/'mtp_component_profile.json';path.write_text(json.dumps(output,indent=2)+'\n')
    print(path)

if __name__=='__main__':main()
