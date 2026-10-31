"""Real semantic Prefill outside timing; cold native HiCache H2D plus MTP Decode."""
from concurrent.futures import ThreadPoolExecutor
import json
import statistics
import threading
import time
import urllib.request
from common import save
from workload import prefixes,query,near_capacity,cached_prefix_length,manifest
from spec_proof import validate_receipts

def request(url,cfg,prefix,index,phase,warm,barrier):
    q,n=(256,1) if phase=='prefill' else (1,128)
    payload=dict(input_ids=prefix+query(index,q,warm),
        rid=f"{'warm' if warm else 'timed'}-{phase}-{len(prefix)}-{index}",
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
    if (meta.get('prompt_tokens')!=len(prefix)+q or meta.get('cached_tokens')!=cached_prefix_length(len(prefix))
        or meta.get('cached_tokens_details',{}).get('host')!=cached_prefix_length(len(prefix))
        or meta.get('cached_tokens_details',{}).get('device')!=0):
        raise RuntimeError('Request did not use exact native bigram CPU cache: '+str(meta))
    if not events or events[-1][0]!=n:raise RuntimeError('Incomplete streamed output')
    if n>1 and events[0][0]!=1:raise RuntimeError('Buffered first chunk cannot establish true TTFT/TPOT')
    return dict(request_index=index,prefix_tokens=len(prefix),query_tokens=q,completion_tokens=n,
        ttft_s=events[0][1]-start,
        tpot_s=(events[-1][1]-events[0][1])/(n-1) if n>1 else None,
        end_to_end_s=end-start,token_events=[(c,t-start) for c,t in events],
        meta=last.get('meta_info',{}),first_stream_chunk_tokens=events[0][0])

def json_request(url,payload,timeout=1200):
    request=urllib.request.Request(url+'/generate',data=json.dumps(payload).encode(),
                                   headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=timeout) as response:
        return json.load(response)

def semantic_sanity(url,run,cfg):
    rows=[]
    for prompt in manifest()['sanity_prompts']:
        payload=dict(input_ids=prompt['input_ids'],rid='natural-sanity-'+str(prompt['index']),
                     sampling_params=dict(temperature=0,ignore_eos=True,max_new_tokens=32))
        if cfg['layout']=='ep8':payload['data_parallel_rank']=0
        result=json_request(url,payload,timeout=180)
        rows.append(dict(prompt=prompt['text'],response=result))
        save(run/cfg['layout']/'semantic_sanity.json',dict(prompts=rows,complete=False))
    metas=[row['response'].get('meta_info',{}) for row in rows]
    verified=all(m.get('spec_verify_ct',0)>0 for m in metas)
    accepted=any(m.get('spec_accept_length',0)>1 for m in metas)
    save(run/cfg['layout']/'semantic_sanity.json',dict(prompts=rows,complete=True,
         actual_mtp_verify=verified,at_least_one_draft_token_accepted=accepted,
         numerical_parity_with_non_speculative_not_asserted=True))
    if not verified or not accepted:
        raise RuntimeError('Short semantic MTP qualification failed; inspect actual outputs and acceptance before long-context Prefill')

def prepare_semantic_cohort(url,run,cfg,seqs):
    start=time.perf_counter()
    def one(index):
        payload=dict(input_ids=seqs[index]+query(index,1,False),
                     rid=f'prepare-semantic-{index}',
                     sampling_params=dict(temperature=0,ignore_eos=True,max_new_tokens=1))
        if cfg['layout']=='ep8':payload['data_parallel_rank']=index%2
        result=json_request(url,payload,timeout=1800)
        meta=result.get('meta_info',{})
        if meta.get('prompt_tokens')!=81921 or meta.get('completion_tokens')!=1:
            raise RuntimeError('Incomplete actual semantic Prefill: '+str(meta))
        # No page-long cross-request prefix is shared. All 81,921 prompt tokens
        # must be produced by actual model Prefill, not injected into host RAM.
        if meta.get('cached_tokens',0)!=0:
            raise RuntimeError('Unexpected shared prefix during semantic construction: '+str(meta))
        return dict(request_index=index,prompt_tokens=81921,meta=meta)
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(one,range(8)))
    value=dict(actual_model_prefill=True,synthetic_kv=False,requests=rows,
               wall_s=time.perf_counter()-start,excluded_from_decode_timing=True,
               checkpoint_replay=False,cache_lifetime='This serving process, native HiCache')
    save(run/cfg['layout']/'semantic_prefill_preparation.json',value)
    return value

def command(run,cfg,prefix,action,id):
    out=run/cfg['layout'];save(out/'command.json',dict(run=cfg['id'],id=id,prefix_length=prefix,action=action))
    deadline=time.monotonic()+300
    while True:
        paths=[out/'receipts'/f'{id}_rank{r}.json' for r in range(8)]
        if all(p.exists() for p in paths):
            rows=[json.loads(p.read_text()) for p in paths]
            if not all(r['passed'] for r in rows):raise RuntimeError('Native cache proof failed')
            return rows
        for node in cfg['nodes']:
            if (out/node/'exit.json').exists():raise RuntimeError('Model worker exited before cache command '+id)
        if time.monotonic()>deadline:raise TimeoutError('Missing native cache receipts: '+id)
        time.sleep(.25)

def stats(values):
    v=sorted(x for x in values if x is not None)
    return dict(mean=statistics.mean(v),median=statistics.median(v),p99=v[-1],p99_estimator='nearest_rank_N8') if v else None

def measure_all(url,run,cfg):
    output=[];out=run/cfg['layout']
    semantic_sanity(url,run,cfg)
    lengths=cfg['prefix_lengths']
    if lengths==[0]:
        caps=[json.loads((out/'capacity'/f'rank{r}.json').read_text())['token_capacity'] for r in range(8)]
        lengths=[near_capacity(min(caps),4 if cfg['layout']=='ep8' else 8)]
        if lengths[0]<=0:raise RuntimeError('Insufficient KV for the frozen batch and tail reserve')
        save(out/'selected_workload.json',dict(prefix_lengths=lengths,rule='90pct_min_rank_capacity_minus_384_tail',capacities=caps))
    for p in lengths:
        seqs=prefixes(p)
        for phase in ('decode',):
            case=f'p{p}_{phase}';command(run,cfg,p,'reset',case+'_reset')
            save(run/'task/progress.json',dict(stage='building_semantic_hicache',layout=cfg['layout'],
                 timestamp=time.time(),valid_points=0,actual_model_prefill=True))
            preparation=prepare_semantic_cohort(url,run,cfg,seqs)
            prepared=command(run,cfg,p,'ready',case+'_semantic_ready')
            save(run/'task/progress.json',dict(stage='measuring_semantic_decode',layout=cfg['layout'],
                 timestamp=time.time(),valid_points=0))
            for warm in (True,False):
                if not warm:
                    proof=command(run,cfg,p,'cold',case+'_cold')
                    save(out/'TIMED',dict(case=case,started_at=time.time()))
                cohort_start=[]
                barrier=threading.Barrier(8,action=lambda:cohort_start.append(time.perf_counter()))
                try:
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        rows=list(pool.map(lambda i:request(url,cfg,seqs[i],i,phase,warm,barrier),range(8)))
                finally:
                    if not warm:(out/'TIMED').unlink(missing_ok=True)
                wall=time.perf_counter()-cohort_start[0]
                if not warm:
                    row=dict(layout=cfg['layout'],workload=phase,request_count=8,
                        prefix_tokens_per_request=p,total_context_tokens=8*p,
                        actual_cached_tokens_per_request=cached_prefix_length(p),
                        prefix_tail_recomputed_tokens=p-cached_prefix_length(p),
                        first_forward_new_tokens_per_request=p+1-cached_prefix_length(p),
                        query_tokens_per_request=256 if phase=='prefill' else 1,
                        output_tokens_per_request=1 if phase=='prefill' else 128,wall_s=wall,
                        throughput_tok_s=8*(256 if phase=='prefill' else 128)/wall,
                        throughput_kind='new_input_tokens' if phase=='prefill' else 'output_tokens',
                        ttft_s=stats([r['ttft_s'] for r in rows]),tpot_s=stats([r['tpot_s'] for r in rows]),
                        end_to_end_s=stats([r['end_to_end_s'] for r in rows]),requests=rows,
                        native_cpu_ready_gpu_empty=proof,old_context_constructed_by_model=True,
                        weights_loading_timed=False,hicache_h2d_timed=True,
                        persistent_source_kv_bank=False,p99_is_eight_request_sample=True,
                        synthetic_kv=False,semantic_input_manifest=manifest(),
                        preparation=preparation,semantic_ready_receipts=prepared)
                    after=command(run,cfg,p,'observe',case+'_after')
                    row['actual_graph_proof']=validate_receipts(after,cfg['layout'])
                    if not all(r['meta'].get('spec_verify_ct',0)>0 for r in rows):
                        raise RuntimeError('Missing per-request actual MTP verify counts')
                    row['spec_accept_length']=stats([r['meta'].get('spec_accept_length') for r in rows])
                    row['spec_accept_rate']=stats([r['meta'].get('spec_accept_rate') for r in rows])
                    row['spec_verify_ct']=stats([r['meta'].get('spec_verify_ct') for r in rows])
                    row['acceptance_scope']='Frozen real source-code/documentation long contexts; not a general corpus quality claim'
                    row['all_rank_after']=after
                    output.append(row);save(out/(case+'_results.json'),row)
                    save(run/'task/partial_results.json',output)
    return output
