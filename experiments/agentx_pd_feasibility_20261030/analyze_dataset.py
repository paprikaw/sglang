"""Raw-file accounting and explicitly optimistic temporal prefix reuse bounds."""
import collections
import csv
import heapq
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / 'datasets/agentx_20261030'

def describe(v):
    v = sorted(v)
    return dict(n=len(v), mean=statistics.mean(v), median=statistics.median(v),
                p90=v[int((len(v)-1)*.9)], p99=v[int((len(v)-1)*.99)], max=v[-1])

def analyze(name):
    inputs=[];outputs=[];uncached=[];rows=[];groups=0;with_sub=0;selected=[]
    types=collections.Counter(); caps=collections.Counter();windows=[]
    with (DATA / name / 'traces.jsonl').open() as source:
        for line in source:
            trace=json.loads(line);normal=[];sub=[]
            for i,r in enumerate(trace['requests']):
                if r['type']=='subagent':
                    groups+=1
                    sub.extend(dict(x,source_path=f'requests/{i}/requests/{j}',agent=r['agent_id']) for j,x in enumerate(r['requests']))
                else:normal.append(dict(r,source_path=f'requests/{i}',agent='top_level'))
            with_sub+=bool(sub); allrows=sorted(normal+sub,key=lambda r:(r['t'],r['source_path']))
            seen=set();pending=[];events=[];session_q=[]
            for idx,r in enumerate(allrows):
                assert r['in']==64*len(r['hash_ids']) and r['out']>=0
                while pending and pending[0][0]<=r['t']:
                    _,j=heapq.heappop(pending);seen.update(allrows[j]['hash_ids'])
                hit=0
                for h in r['hash_ids']:
                    if h not in seen:break
                    hit+=64
                q=r['in']-hit;session_q.append(q)
                heapq.heappush(pending,(r['t']+max(r.get('api_time') or 0,0),idx))
                duration=max(r.get('api_time') or 0,0)
                events.extend([(r['t'],1),(r['t']+duration,-1)])
                types['nested' if r['agent']!='top_level' else 'top_level']+=1
                inputs.append(r['in']);outputs.append(r['out']);uncached.append(q)
                for cap in (32768,65536,131072,256000,262144):
                    caps[str(cap)]+=r['in']+r['out']<=cap
            busy=peak=0
            for _,delta in sorted(events):busy+=delta;peak=max(peak,busy)
            rows.append(dict(id=trace['id'],requests=len(allrows),top_level=len(normal),nested=len(sub),
                mean_input=statistics.mean(r['in'] for r in allrows),
                mean_output=statistics.mean(r['out'] for r in allrows),
                mean_new_tokens_optimistic=statistics.mean(session_q),
                input_tokens=sum(r['in'] for r in allrows),new_tokens_optimistic=sum(session_q),
                recorded_api_peak_overlap=peak))
            # Eligibility before GPU measurements: five contiguous top-level
            # requests, same model, no nested entry/overlap, bounded practical
            # runtime, and real extending prefixes. This is a named subset,
            # not the official full AgentX graph benchmark.
            if name.endswith('256k'):
                stream=trace['requests']
                for start in range(len(stream)-4):
                    seq=stream[start:start+5]
                    if any(r['type']=='subagent' or not (1024<=r['in']<=65536 and 16<=r['out']<=512) for r in seq):continue
                    if len(set(r['model'] for r in seq))!=1:continue
                    delays=[seq[j]['t']-seq[j-1]['t']-max(seq[j-1].get('api_time') or 0,0) for j in range(1,5)]
                    if any(not 0<=x<=2 for x in delays):continue
                    hits=[]
                    for a,b in zip(seq,seq[1:]):
                        k=0
                        for x,y in zip(a['hash_ids'],b['hash_ids']):
                            if x!=y:break
                            k+=64
                        hits.append(k)
                    if any(k<1024 for k in hits):continue
                    q=[r['in']-h for r,h in zip(seq[1:],hits)]
                    windows.append(dict(session_id=trace['id'],start_index=start,requests=seq,
                        waits_s=delays,pairwise_prefix_tokens=hits,mean_new_tokens=statistics.mean(q),
                        mean_output=statistics.mean(r['out'] for r in seq[1:])))
                    break
    result=dict(dataset=name,sessions=len(rows),requests=len(inputs),request_types=dict(types),
        subagent_groups=groups,sessions_with_subagents=with_sub,total_input_tokens=sum(inputs),
        total_output_tokens=sum(outputs),input=describe(inputs),output=describe(outputs),
        optimistic_new_input=describe(uncached),optimistic_token_weighted_reuse=1-sum(uncached)/sum(inputs),
        context_cap_eligible_requests=dict(caps),
        prefix_bound='Infinite session-local cache of input blocks from requests completed before each recorded start; no eviction, no migration, no GDN endpoint limit. Not a measured cache hit rate.',
        eligible_five_turn_windows=len(windows))
    with (ROOT/(name+'-sessions.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if windows:
        # Sample evenly over Q/O, rather than choosing based on engine results.
        windows.sort(key=lambda w:w['mean_new_tokens']/w['mean_output'])
        chosen=[windows[round(i*(len(windows)-1)/7)] for i in range(8)]
        (ROOT/'screening_windows.json').write_text(json.dumps(dict(protocol='AgentX-derived contiguous linear five-turn screen; no graph/fanout claim; unchanged lengths and inter-turn waits',eligible=len(windows),windows=chosen),indent=2)+'\n')
        result['selected_window_summary']=[{k:w[k] for k in ('session_id','start_index','mean_new_tokens','mean_output')} for w in chosen]
    (ROOT/(name+'-statistics.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    for name in ('cc-traces-weka-062126-256k','cc-traces-weka-062126'):analyze(name)
