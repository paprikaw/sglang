"""Exact-name GPU phase calibration from real CPU-scoped Kineto traces.

Train on first two detailed forwards, validate on remaining held-out forwards.
Cross-node ranks 0/4 are both audited. Conflicting names are never classified.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'glm47_a100_cached_decode_stage_profile_20261022'
sys.path.insert(0,str(SOURCE/'analysis'))
from kineto_summary import ScopeIndex,arg,interval,parse_scope,scope_category,RUNTIME_CATEGORIES


def broad(category):
    if category=='attention':return 'attention'
    if category in ('expert.routed','expert.shared','dense_mlp'):return 'expert_ffn'
    if category.startswith('deep_ep.'):return category
    if category=='tp_collective':return 'tp_communication'
    if category=='pp_transport_api':return 'pp_communication'
    return 'other_gpu'


def read(path):return json.loads(path.read_text())


def rank_evidence(arm,rank):
    path=SOURCE/'runs/profile_r1'/arm/f'attribution_rank{rank}.json';sample=read(path)
    if not sample['passed']:raise ValueError('Invalid calibration receipt')
    trace_path=path.parent/sample['trace_filename'];trace=read(trace_path)
    events=[(i,e) for i,e in enumerate(trace['traceEvents']) if e.get('ph')=='X' and 'ts'in e and 'dur'in e]
    scopes=ScopeIndex(events);runtime=defaultdict(list)
    for i,e in events:
        if e.get('cat') in RUNTIME_CATEGORIES:
            key=arg(e,'correlation','correlation id')
            if key is not None:runtime[key].append(e)
    model_scopes=sorted([(i,e) for i,e in events if (parse_scope(e)or(None,))[0]=='model_forward'],key=lambda x:interval(x[1])[0])
    receipts=sorted([e for e in sample['events'] if e['category']=='model_forward'],key=lambda e:e['cpu_start_ns'])
    if len(model_scopes)!=len(receipts):raise ValueError('Unmatched CPU forward scopes')
    ids={i:e['forward_index'] for (i,_),e in zip(model_scopes,receipts)}
    selected=sorted([e for e in sample['forwards'] if e['detailed']],key=lambda e:e['extend_index'])
    training={e['forward_index'] for e in selected[:2]};heldout={e['forward_index'] for e in selected[2:]}
    if len(training)!=2 or not heldout:raise ValueError('No independent held-out scope')
    observations=defaultdict(lambda:dict(phases=set(),training=set(),heldout=set(),pp_scopes=set(),count=0))
    for _,event in events:
        if event.get('cat')!='kernel':continue
        calls=runtime.get(arg(event,'correlation','correlation id'),[])
        if len(calls)!=1:continue
        containers=scopes.containing(calls[0]);lineage=[parse_scope(n['row']) for n in containers]
        category=scope_category(lineage)
        model_node=next((n['index'] for n in containers if parse_scope(n['row'])[0]=='model_forward'),None)
        forward=ids.get(model_node)
        if forward not in training|heldout and category!='pp_transport_api':continue
        item=observations[event['name']];item['phases'].add(broad(category));item['count']+=1
        if forward in training:item['training'].add(forward)
        if forward in heldout:item['heldout'].add(forward)
        if category=='pp_transport_api':
            pp_node=next(n['index'] for n in containers if parse_scope(n['row'])[0].startswith('pp_'))
            item['pp_scopes'].add(pp_node)
    output={}
    for name,value in observations.items():
        output[name]=dict(phases=sorted(value['phases']),count=value['count'],
            training_forwards=sorted(value['training']),heldout_forwards=sorted(value['heldout']),
            pp_independent_scope_count=len(value['pp_scopes']),
            has_training_and_holdout=value['training']==training and bool(value['heldout']))
    return dict(arm=arm,rank=rank,names=output,
        all_kernel_names=sorted({e['name'] for _,e in events if e.get('cat')=='kernel'}),
        source_receipt=str(path),source_trace=str(trace_path),
        receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        trace_bytes=trace_path.stat().st_size,trace_mtime_ns=trace_path.stat().st_mtime_ns)


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('ep8','pp2ep4'),required=True);a=p.parse_args()
    ranks=[rank_evidence(a.arm,rank) for rank in (0,4)]
    names=sorted({name for rank in ranks for name in rank['names']});accepted={};audit={}
    for name in names:
        evidence=[dict(rank=r['rank'],**r['names'][name]) for r in ranks if name in r['names']]
        phases={phase for e in evidence for phase in e['phases']}
        validated=any(e['has_training_and_holdout'] for e in evidence)
        if phases=={'pp_communication'}:validated=any(e['pp_independent_scope_count']>=3 for e in evidence)
        eligible=len(phases)==1 and validated
        if eligible:accepted[name]=next(iter(phases))
        audit[name]=dict(eligible=eligible,phases=sorted(phases),evidence=evidence)
    result=dict(arm=a.arm,exact_names=accepted,audit=audit,ranks=ranks,
        all_observed_kernel_names=sorted({name for r in ranks for name in r['all_kernel_names']}),
        method='Actual GPU kernel launch -> unique CUDA-runtime correlation -> innermost CPU attr scopes. Two training forwards plus held-out forwards; conflicting names rejected. Independent PP API scopes for PP transport. No substring heuristics.')
    out=ROOT/f'calibration_{a.arm}.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(a.arm,'eligible',len(accepted),'conflict_or_unvalidated',len(names)-len(accepted),flush=True)


if __name__=='__main__':main()
