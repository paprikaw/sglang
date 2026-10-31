"""Separate host communication scopes from GPU kernel time in three main-run points."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics as st

ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    run=args.run.absolute();out=ROOT/'analysis'/run.name
    spec=importlib.util.spec_from_file_location('prefill_host_tools',ROOT.parent/'pp_ep_steady_decode_20261026/analyze.py')
    tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)
    rows=[]
    for case in ('b32_q128_repeated','b32_q512_repeated','b32_q4096_repeated'):
        for arm,policy in (('ep2',0),('pp2',1)):
            folder=run/arm/'workloads'/case/'traces'/f'p{policy}'
            manifest=json.loads((folder/'manifest.json').read_text())
            for d in manifest['files']:
                path=Path(d['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==d['sha256']
                with gzip.open(path,'rt') as f:
                    events=[e for e in json.load(f)['traceEvents'] if e.get('ph')=='X' and e.get('dur',0)>0]
                ann=[e for e in events if e.get('cat')=='user_annotation']
                si=tools.IntervalIndex([e for e in ann if e['name'].startswith('step[')])
                layers=[]
                for e in ann:
                    if re.fullmatch(r'component\.layer\.\d+',e['name']):
                        parent=si.find(e['ts'],e.get('tid'))
                        if parent and 'EXTEND' in parent['name']:layers.append(e)
                li=tools.IntervalIndex(layers);scopes=Counter();sync=Counter();sync_counts=Counter()
                for e in ann:
                    if e['name'].startswith('component.') and not e['name'].startswith('component.layer.'):
                        layer=li.find(e['ts'],e.get('tid'))
                        if layer is not None:scopes[tools.label(e)]+=e['dur']
                for e in events:
                    if e.get('cat')=='cuda_runtime' and e['name'] in ('cudaStreamSynchronize','cudaEventSynchronize','cudaDeviceSynchronize'):
                        if li.find(e['ts'],e.get('tid')) is not None:
                            sync[e['name']]+=e['dur'];sync_counts[e['name']]+=1
                rows.append(dict(workload=case,arm=arm,policy=policy,rank_trace=path.name,
                    actual_layer_calls=len(layers),
                    host_layer_us_per_layer=sum(e['dur'] for e in layers)/len(layers),
                    host_dispatch_us_per_layer=scopes['communication.dispatch']/len(layers),
                    host_combine_us_per_layer=scopes['communication.combine']/len(layers),
                    host_dispatch_combine_us_per_layer=(scopes['communication.dispatch']+scopes['communication.combine'])/len(layers),
                    host_routed_expert_us_per_layer=scopes['expert.routed']/len(layers),
                    cuda_sync_host_us_per_layer=sum(sync.values())/len(layers),
                    cuda_sync_calls_per_layer=sum(sync_counts.values())/len(layers),
                    scope='Host wall durations in independent instrumented Prefill profile; includes launch/sync/wait and profiler overhead; not additive with GPU kernel times'))
    fields=list(rows[0]);
    with (out/'host_components_per_rank.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    means=[]
    for case in ('b32_q128_repeated','b32_q512_repeated','b32_q4096_repeated'):
        for arm in ('ep2','pp2'):
            chosen=[r for r in rows if r['workload']==case and r['arm']==arm];assert len(chosen)==2
            m=dict(workload=case,arm=arm,rank_profiles=2)
            for k in fields:
                if k.endswith(('_us_per_layer','_calls_per_layer')):m[k]=st.mean(r[k] for r in chosen)
            means.append(m)
    with (out/'host_components.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(means[0]));w.writeheader();w.writerows(means)
    (out/'host_analysis.json').write_text(json.dumps(dict(run=str(run),rows=rows,means=means),indent=2)+'\n')
    print(json.dumps(means,indent=2))


if __name__=='__main__':main()
