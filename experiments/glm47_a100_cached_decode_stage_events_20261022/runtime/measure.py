"""Warm prefixes, then compare hot-prefix control with bounded Decode profile."""
from concurrent.futures import ThreadPoolExecutor
import json
import statistics
import threading
import time
import urllib.request
from common import save
from workload import prefixes,query,near_capacity

def request(url,cfg,prefix,index,phase,warm,barrier):
    q,n=1,cfg['profile_output_tokens']
    payload=dict(input_ids=prefix+query(index,q,warm),
        rid=f"{cfg.get('cohort','warm')}-{phase}-{len(prefix)}-{index}",
        sampling_params=dict(temperature=0,ignore_eos=True,max_new_tokens=n),stream=True)
    if cfg['layout']=='ep8':payload['data_parallel_rank']=index%2
    req=urllib.request.Request(url+'/generate',data=json.dumps(payload).encode(),
                               headers={'Content-Type':'application/json'})
    barrier.wait();start=time.perf_counter();events=[];last=None
    with urllib.request.urlopen(req,timeout=1200) as f:
        for line in f:
            if not line.startswith(b'data:'):continue
            data=line[5:].strip()
            if data==b'[DONE]':break
            obj=json.loads(data);now=time.perf_counter();last=obj
            count=obj.get('meta_info',{}).get('completion_tokens',0)
            if count and (not events or count>events[-1][0]):events.append((count,now))
    end=time.perf_counter()
    meta=last.get('meta_info',{}) if last else {}
    details=meta.get('cached_tokens_details',{})
    if (meta.get('prompt_tokens')!=len(prefix)+q or meta.get('cached_tokens')!=len(prefix)
        or details.get('host',0)+details.get('device',0)!=len(prefix)):
        raise RuntimeError('Request did not use the complete cached prefix')
    if not warm and details!={'device':len(prefix),'host':0}:
        raise RuntimeError('Stable Decode profile requires the complete GPU-resident prefix')
    if not events or events[-1][0]!=n:raise RuntimeError('Incomplete streamed output')
    if n>1 and events[0][0]!=1:raise RuntimeError('Buffered first chunk cannot establish true TTFT/TPOT')
    return dict(request_index=index,prefix_tokens=len(prefix),query_tokens=q,completion_tokens=n,
        ttft_s=events[0][1]-start,
        tpot_s=(events[-1][1]-events[0][1])/(n-1) if n>1 else None,
        end_to_end_s=end-start,token_events=[(c,t-start) for c,t in events],
        meta=last.get('meta_info',{}),first_stream_chunk_tokens=events[0][0])

def command(run,cfg,prefix,action,id):
    out=run/cfg['layout'];save(out/'command.json',dict(run=cfg['id'],id=id,prefix_length=prefix,action=action))
    deadline=time.monotonic()+300
    while True:
        paths=[out/'receipts'/f'{id}_rank{r}.json' for r in range(8)]
        if all(p.exists() for p in paths):
            rows=[json.loads(p.read_text()) for p in paths]
            if not all(r['passed'] for r in rows):raise RuntimeError('Native cache proof failed')
            return rows
        if time.monotonic()>deadline:raise TimeoutError('Missing native cache receipts: '+id)
        time.sleep(.25)

def stats(values):
    v=sorted(x for x in values if x is not None)
    return dict(mean=statistics.mean(v),median=statistics.median(v),p99=v[-1],p99_estimator='nearest_rank_N8') if v else None

def measure_all(url,run,cfg):
    out=run/cfg['layout'];p=cfg['prefix_lengths'][0];phase='decode';seqs=prefixes(p)
    command(run,cfg,p,'fill','native_fill')
    cohorts=[]
    for cohort,warm in [('warm',True),('control',False),('timed-profile',False)]:
        if not warm:
            proof=command(run,cfg,p,'observe',cohort+'_gpu_ready')
            if not all(all(m['gpu_tokens']==p for m in r['matches']) for r in proof):
                raise RuntimeError('A target GPU prefix was evicted before stable Decode')
        if cohort=='timed-profile':
            marker=dict(schema_version=1,run_id=run.name,arm=cfg['layout'],
                token=cfg['attribution']['run_token'],enabled=True,
                detailed_forward_ordinals=cfg['attribution']['by_arm'][cfg['layout']])
            save(out/'ATTRIBUTION_CONTROL.json',marker)
            deadline=time.monotonic()+60
            while True:
                paths=[out/f'attribution_armed_rank{r}.json' for r in range(8)]
                if all(x.exists() for x in paths):
                    acks=[json.loads(x.read_text()) for x in paths]
                    if not all(x['passed'] for x in acks):raise RuntimeError('Profiler arming failed')
                    break
                if time.monotonic()>deadline:raise TimeoutError('Missing profiler arming receipts')
                time.sleep(.1)
            save(out/'TIMED',dict(case=cohort,started_at=time.time()))
        cohort_start=[];barrier=threading.Barrier(8,action=lambda:cohort_start.append(time.perf_counter()))
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                rows=list(pool.map(lambda i:request(url,dict(cfg,cohort=cohort),seqs[i],i,phase,warm,barrier),range(8)))
        finally:
            if cohort=='timed-profile':(out/'TIMED').unlink(missing_ok=True)
        wall=time.perf_counter()-cohort_start[0]
        row=dict(layout=cfg['layout'],workload=phase,cohort=cohort,request_count=8,
            prefix_tokens_per_request=p,total_context_tokens=8*p,query_tokens_per_request=1,
            output_tokens_per_request=cfg['profile_output_tokens'],wall_s=wall,
            throughput_tok_s=8*cfg['profile_output_tokens']/wall,throughput_kind='output_tokens',
            ttft_s=stats([r['ttft_s'] for r in rows]),tpot_s=stats([r['tpot_s'] for r in rows]),
            end_to_end_s=stats([r['end_to_end_s'] for r in rows]),requests=rows,
            gpu_hot_prefix_profile=True,weights_loading_timed=False,old_context_constructed_by_model=False)
        save(out/(cohort+'_results.json'),row)
        if not warm:cohorts.append(row)
    command(run,cfg,p,'profile_dump','profile_dump')
    profiles=[json.loads((out/f'attribution_rank{r}.json').read_text()) for r in range(8)]
    for r in profiles:
        if not r['passed'] or r['detail_ordinals_seen']!=cfg['attribution']['by_arm'][cfg['layout']]:
            raise RuntimeError('Invalid or incomplete all-rank Decode profile')
        selected=[f for f in r['forwards'] if f['detailed']]
        if not all(f['metadata']['batch_size']==4 and f['metadata']['forward_mode']=='2' for f in selected):
            raise RuntimeError('Profiled forward was not steady B4 Decode')
    row=dict(cohorts[-1],profile_validated=True,profile_ranks=8,control=cohorts[0],
        detail_ordinals=cfg['attribution']['by_arm'][cfg['layout']],
        profile_to_control_tpot_ratio=cohorts[-1]['tpot_s']['mean']/cohorts[0]['tpot_s']['mean'])
    save(out/'profile_result.json',row)
    return [row]
