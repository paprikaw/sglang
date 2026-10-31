"""Audit independent fixed-policy deployments and real Decode graph overlap."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def union(spans):
    out=[]
    for a,b in sorted(spans):
        if not out or a>out[-1][1]:out.append([a,b])
        else:out[-1][1]=max(out[-1][1],b)
    return out
def intersect(a,b):
    i=j=0;total=0.
    while i<len(a) and j<len(b):
        total+=max(0.,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<=b[j][1]:i+=1
        else:j+=1
    return total

def profile(entry,arm,cfg,helper,tools):
    trace=helper.read_trace(Path(entry['path']),entry['sha256'],cfg['local'])
    selected=trace['graphs'][cfg['drop_initial_graphs']:cfg['drop_initial_graphs']+cfg['selected_forwards']]
    assert len(selected)==cfg['selected_forwards']
    launch_by_ts={e['ts']:e for e in trace['events'] if e['name']=='cudaGraphLaunch'}
    by_corr=defaultdict(list)
    for event in trace['events']:
        if event.get('cat')=='kernel':by_corr[event['args'].get('correlation')].append(event)
    durations,counts,all_spans=defaultdict(float),defaultdict(int),defaultdict(list)
    for graph in selected:
        kernels=by_corr[launch_by_ts[graph['launch']]['args']['correlation']]
        assert kernels and all(k['args'].get('graph id') for k in kernels)
        for event in kernels:
            bucket=tools.graph_bucket(event['name'])
            durations[bucket]+=event['dur'];counts[bucket]+=1
            if bucket in ('communication.dispatch','communication.combine'):kind='comm'
            elif bucket=='expert.routed_gemm':kind='gemm'
            else:continue
            all_spans[kind].append((event['ts'],event['ts']+event['dur']))
    layers=40 if arm=='ep2' else 20
    divisor=len(selected)*layers
    comm=union(all_spans['comm']);gemm=union(all_spans['gemm'])
    hit=intersect(comm,gemm);comm_union=sum(b-a for a,b in comm)
    return dict(trace=entry,selected_graphs=len(selected),layers_per_rank=layers,
        kernel_us_per_layer={k:v/divisor for k,v in durations.items()},
        kernel_count_per_layer={k:v/divisor for k,v in counts.items()},
        comm_gemm_overlap_us_per_layer=hit/divisor,
        communication_overlapped_pct=100*hit/comm_union if comm_union else 0.,
        model_graph_ms=st.mean((g['end']-g['start'])/1000 for g in selected))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--skip-profiles',action='store_true')
    a=ap.parse_args();run=a.run.resolve();cfg=read(run/'config.json')
    assert read(run/'task/progress.json')['stage']=='complete'
    assert read(run/'task/cleanup.json')['passed']
    checks=[read(p) for p in (run/'dashboard_guard/checks').glob('*-post.json')]
    assert len(checks)==1 and checks[0]['passed'] and checks[0]['cleanup_verified']
    assert '"gpu_registry_state": "released"' in (run/'supervisor.log').read_text()
    assert read(run/'cpu_qualification.json')['passed']
    for item in read(run/'dashboard_spec.json')['frozen_inputs']:assert sha(item['path'])==item['sha256'],item
    helper=load('decode_overlap_graphs',ROOT.parent/'h100_pp_lookahead_workloads_20261027/analyze.py')
    tools=load('decode_overlap_components',ROOT.parent/'h100_pp_expert_fragmentation_20261026/analyze.py').load_analysis()
    old=tools.graph_bucket
    tools.graph_bucket=lambda n:'expert.layout_adapter' if n=='_metadata' or n.startswith('_metadata<') else old(n)
    rows,components,sources=[],[],[]
    arm=cfg['arm_order'][0]
    assert read(run/arm/'cleanup.json')['cleanup_verified'] and read(run/arm/'completed.json')['passed']
    for case in cfg['workloads']:
        folder=run/arm/'workloads'/case['id'];p=folder/'metrics.decode.json'
        d=read(p);groups=d['rounds'];proto=read(folder/'protocol.json')
        previous=ROOT.parent/'h100_pp_lookahead_workloads_20261027/runs/lookahead_r2'/arm/'workloads'/case['id']/'protocol.json'
        assert proto['prompt_sha256']==read(previous)['prompt_sha256']
        assert d['valid'] and not d['profiler_enabled'] and len(groups)==cfg['repeats']
        assert read(folder/'completed.json')['passed']
        if arm=='pp2':assert read(folder/'output_agreement.json')['all_formal_sequences_equal']
        flat=[];throughput=[]
        for group in groups:
            requests=group['requests'];assert len(requests)==case['requests']
            for r in requests:
                assert r['cached_tokens']==4096 and r['output_tokens']==128 and len(r['output_ids'])==128
                assert sorted(set(e['completion_tokens'] for e in r['events']))==list(range(1,129))
                observed=(r['events'][-1]['time_s']-r['events'][0]['time_s'])*1000/127
                assert abs(observed-r['tpot_ms'])<1e-6
            rate=sum(r['output_tokens'] for r in requests)/group['makespan_s']
            assert abs(rate-group['output_tokens_per_s'])<1e-6
            throughput.append(rate);flat.extend(requests)
            if arm=='ep2':
                proof=group['overlap_receipts'];assert all(n>0 for n in proof['graph_replay_deltas'])
                assert all(s['drained'] and s['policy']==cfg['decode_overlap_policy'] for s in proof['states'])
            else:
                assert all(s['ahead_produced']>0 and s['ahead_produced']==s['ahead_consumed'] for s in group['scheduling_receipt']['states'])
        if arm=='ep2':
            quals=[read(q) for q in (folder/'overlap_qualification').glob('rank*.json')]
            assert len(quals)==2 and all(len(q)==40 and all(v['passed'] and v['serial_vs_overlap']['exact'] for v in q.values()) for q in quals)
        common=dict(workload=case['id'],arm=arm,variant=cfg['variant'],requests=case['requests'],local_tokens=case['requests']//2)
        rows.append(dict(**common,output_tok_s=st.mean(throughput),stdev_tok_s=st.stdev(throughput),
            repeats=len(groups),tpot_ms=st.mean(r['tpot_ms'] for r in flat),
            ttft_ms=1000*st.mean(r['ttft_s'] for r in flat),e2e_ms=1000*st.mean(r['e2e_s'] for r in flat),
            samples_tok_s=throughput,source=str(p),source_sha256=sha(p)))
        sources.extend(dict(path=str(q),sha256=sha(q)) for q in [p,folder/'protocol.json'])
        if not a.skip_profiles:
            manifest=read(folder/'traces/decode/manifest.json');assert len(manifest['files'])==2
            for rank,entry in enumerate(sorted(manifest['files'],key=lambda e:e['path'])):
                method=sha(Path(__file__))+sha(ROOT.parent/'h100_pp_lookahead_workloads_20261027/analyze.py')+sha(ROOT.parent/'h100_pp_expert_fragmentation_20261026/analyze.py')
                cache=ROOT/'analysis/profile_cache'/hashlib.sha256(method.encode()).hexdigest()[:16]/(entry['sha256']+'.json')
                assert sha(entry['path'])==entry['sha256']
                v=read(cache) if cache.exists() else profile(entry,arm,dict(cfg,local=common['local_tokens']),helper,tools)
                save(cache,v);components.append(dict(**common,rank=rank,**v));sources.append(entry)
    result=dict(run=str(run),config=cfg,audit=dict(guard_and_cleanup_verified=True,frozen_hashes_verified=True),
        metrics=rows,components=components,profiles_skipped=a.skip_profiles,sources=sources)
    save(ROOT/'analysis'/run.name/'results.json',result)
    print(json.dumps(dict(run=run.name,metrics=rows),indent=2))
if __name__=='__main__':main()
