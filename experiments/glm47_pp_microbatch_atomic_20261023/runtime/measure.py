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

def request(url,cfg,prefix,index,phase,warm,barrier,*,max_output_tokens=None,cohort_prefix=None):
    if max_output_tokens is None:max_output_tokens=cfg['production_output_tokens']
    q,n=(256,1) if phase=='prefill' else (1,max_output_tokens)
    cohort_prefix=cohort_prefix or ('warm' if warm else 'timed')
    payload=dict(input_ids=prefix+query(index,q,warm),
        rid=f"{cohort_prefix}-{phase}-{len(prefix)}-{index}",
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
    output_ids=last.get('output_ids')
    if not isinstance(output_ids,list) or len(output_ids)!=n:
        raise RuntimeError('Complete generated token IDs required for paired correctness')
    return dict(request_index=index,prefix_tokens=len(prefix),query_tokens=q,completion_tokens=n,
        ttft_s=events[0][1]-start,
        tpot_s=(events[-1][1]-events[0][1])/(n-1) if n>1 else None,
        end_to_end_s=end-start,token_events=[(c,t-start) for c,t in events],
        meta=last.get('meta_info',{}),first_stream_chunk_tokens=events[0][0],
        output_text=last.get('text'),output_ids=output_ids)

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

def batch_request(url,cfg,seqs,cohort_base,*,max_output_tokens=None,prepared_cache=True):
    from atomic_stream import make_payload,generate_batch
    n=cfg['production_output_tokens'] if max_output_tokens is None else max_output_tokens
    payload=make_payload(seqs,query,cohort_base,n)
    rows,wall=generate_batch(url,payload,timeout=1800)
    for row in rows:
        p=len(seqs[row['request_index']]);meta=row['meta']
        expected=cached_prefix_length(p) if prepared_cache else 0
        if meta.get('prompt_tokens')!=p+1 or meta.get('cached_tokens',0)!=expected:
            raise RuntimeError('Exact real semantic/cache length mismatch: '+str(meta))
        if prepared_cache and (meta.get('cached_tokens_details',{}).get('host')!=expected or
                               meta.get('cached_tokens_details',{}).get('device')!=0):
            raise RuntimeError('Exact native cold CPU cache was not used: '+str(meta))
        row.update(prefix_tokens=p,query_tokens=1)
    return rows,wall


def prepare_semantic_cohort(url,run,cfg,seqs):
    rows,wall=batch_request(url,cfg,seqs,'prepare-semantic',max_output_tokens=1,prepared_cache=False)
    value=dict(actual_model_prefill=True,synthetic_kv=False,requests=rows,
        wall_s=wall,excluded_from_decode_timing=True,checkpoint_replay=False,
        atomic_http_batch=True,cache_lifetime='This serving process, native HiCache')
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

def profile_once(url,run,cfg,p,seqs,case):
    """Separate bounded profile after saving the unprofiled production cohort."""
    out=run/cfg['layout']
    if (out/'TIMED').exists() or not (out/(case+'_results.json')).exists():
        raise RuntimeError('Profiling cannot precede or overlap production timing')
    proof=command(run,cfg,p,'profile_cold',case+'_profile_cold')
    if not all(r['profiler']['armed'] and r['profiler']['active'] for r in proof):
        raise RuntimeError('All ranks must arm the separate profile')
    rows,wall=batch_request(url,cfg,seqs,f'profile-decode-{p}',
        max_output_tokens=cfg['profile_output_tokens'])
    after=command(run,cfg,p,'observe',case+'_profile_after')
    graph=validate_receipts(after,cfg['layout'],cohort_prefix='profile-')
    if not all(r['profiler']['done'] and not r['profiler']['active'] for r in after):
        raise RuntimeError('The bounded GPU profile did not finish on every rank')
    from pathlib import Path
    from artifact_visibility import wait_for_stable_json
    receipts=[]
    for rank in range(8):
        receipt,_=wait_for_stable_json(out/f'mtp_activity_receipt_rank{rank}.json',timeout_s=30)
        if (not receipt['passed'] or receipt['rank']!=rank or receipt['verify_count']!=12
            or not Path(receipt['trace']).is_file()):
            raise RuntimeError('Incomplete bounded profile receipt')
        receipts.append(receipt)
    value=dict(profile_only=True,performance_metrics_not_for_production_comparison=True,
        output_tokens_per_request=cfg['profile_output_tokens'],requests=rows,
        production_output_tokens=cfg['production_output_tokens'],native_cpu_ready_gpu_empty=proof,
        graph_proof=graph,all_rank_after=after,activity_receipts=receipts,
        real_semantic_kv_reused_in_same_process=True,additional_long_prefill=False,
        atomic_http_batch=True,wall_s=wall,
        actual_microbatch_scheduler_proof=validate_microbatch_receipts(after,cfg['mode'],'profile-'))
    save(out/(case+'_profile_results.json'),value)

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
            for receipt in prepared:
                cohorts=receipt['graph'].get('cohorts',{})
                item=cohorts.get('prepare-semantic',{})
                target=sum(v for k,v in item.get('forwards',{}).items() if k.startswith('target:') and 'EXTEND' in k)
                draft=item.get('phases',{}).get('draft_prefill',0)
                if not target or bool(draft)!=(cfg['layout']=='ep8' or receipt['rank']>=4):
                    raise RuntimeError('Real target/MTP semantic Prefill coverage missing: '+str(receipt['rank']))
            save(run/'task/progress.json',dict(stage='measuring_semantic_decode',layout=cfg['layout'],
                 timestamp=time.time(),valid_points=0))
            warm_rows,warm_wall=batch_request(url,cfg,seqs,f'warm-decode-{p}')
            save(out/(case+'_warm_results.json'),dict(requests=warm_rows,wall_s=warm_wall,untimed=True))
            production=[]
            for cohort in ('timed','repeat'):
                proof=command(run,cfg,p,'cold',case+'_'+cohort+'_cold')
                save(out/'TIMED',dict(case=case,cohort=cohort,started_at=time.time()))
                try:
                    rows,wall=batch_request(url,cfg,seqs,f'{cohort}-decode-{p}')
                finally:(out/'TIMED').unlink(missing_ok=True)
                after=command(run,cfg,p,'observe',case+'_'+cohort+'_after')
                if any(r['profiler']['armed'] or r['profiler']['active'] for r in proof+after):
                    raise RuntimeError('CUDA profiling contaminated production timing')
                graph=validate_receipts(after,cfg['layout'],cohort_prefix=cohort+'-')
                slots=validate_microbatch_receipts(after,cfg['mode'],cohort+'-')
                if not all(r['meta'].get('spec_verify_ct',0)>0 for r in rows):
                    raise RuntimeError('Missing per-request actual MTP verify counts')
                row=dict(layout=cfg['layout'],workload=phase,request_count=8,cohort=cohort,
                    microbatch_mode=cfg['mode'],engine_commit=cfg['engine_commit'],
                    prefix_tokens_per_request=p,total_context_tokens=8*p,
                    actual_cached_tokens_per_request=cached_prefix_length(p),
                    prefix_tail_recomputed_tokens=p-cached_prefix_length(p),
                    first_forward_new_tokens_per_request=p+1-cached_prefix_length(p),
                    query_tokens_per_request=1,output_tokens_per_request=cfg['production_output_tokens'],wall_s=wall,
                    throughput_tok_s=8*cfg['production_output_tokens']/wall,throughput_kind='output_tokens',
                    ttft_s=stats([r['ttft_s'] for r in rows]),tpot_s=stats([r['tpot_s'] for r in rows]),
                    end_to_end_s=stats([r['end_to_end_s'] for r in rows]),requests=rows,
                    native_cpu_ready_gpu_empty=proof,old_context_constructed_by_model=True,
                    weights_loading_timed=False,hicache_h2d_timed=True,persistent_source_kv_bank=False,
                    p99_is_eight_request_sample=True,synthetic_kv=False,semantic_input_manifest=manifest(),
                    preparation=preparation,semantic_ready_receipts=prepared,actual_graph_proof=graph,
                    spec_accept_length=stats([r['meta'].get('spec_accept_length') for r in rows]),
                    spec_accept_rate=stats([r['meta'].get('spec_accept_rate') for r in rows]),
                    spec_verify_ct=stats([r['meta'].get('spec_verify_ct') for r in rows]),
                    profile_only_performance_not_production_measurement=False,
                    cuda_profiler_off_during_production=True,startup_calibration_and_dormant_cpu_hooks_present=True,
                    acceptance_scope='Frozen real source-code/documentation contexts; not a general corpus quality claim',
                    all_rank_after=after,actual_microbatch_scheduler_proof=slots,atomic_http_batch=True,
                    requested_fcfs_groups=[[0,1,2,3],[4,5,6,7]])
                production.append(row)
                save(out/(case+('_results.json' if cohort=='timed' else '_repeat_results.json')),row)
            from atomic_stream import output_parity
            repeatability=output_parity(production[0]['requests'],production[1]['requests'])
            repeatability.update(first_slot_composition=production[0]['actual_microbatch_scheduler_proof'],
                second_slot_composition=production[1]['actual_microbatch_scheduler_proof'],
                both_cohorts_unprofiled=True,both_start_native_cpu_ready_gpu_empty=True,
                same_serving_process=True)
            save(out/(case+'_repeatability.json'),repeatability)
            production[0]['repeatability']=repeatability
            production[0]['repeat_performance']={k:production[1][k] for k in
                ('wall_s','throughput_tok_s','ttft_s','tpot_s','end_to_end_s','spec_accept_rate')}
            output.append(production[0]);save(out/(case+'_results.json'),production[0])
            save(run/'task/partial_results.json',output)
            save(run/'task/progress.json',dict(stage='separate_cuda_profile',layout=cfg['layout'],
                 timestamp=time.time(),valid_points=1))
            profile_once(url,run,cfg,p,seqs,case)
    return output

def validate_microbatch_receipts(receipts,mode,cohort_prefix='timed-'):
    if mode not in ('baseline','candidate') or len(receipts)!=8 or sorted(r['rank'] for r in receipts)!=list(range(8)):
        raise RuntimeError('Missing exact eight-rank microbatch proof')
    composition=[]
    for receipt in receipts:
        proof=receipt.get('microbatch',{})
        if proof.get('enabled')!=(mode=='candidate') or proof.get('loop_size')!=2:
            raise RuntimeError('Microbatch mode/slot proof mismatch')
        if mode=='candidate' and (not proof.get('feedback_group_name') or not proof.get('feedback_group_ranks')):
            raise RuntimeError('Candidate feedback group identity missing')
        batches=[b for b in receipt.get('batches',[]) if b.get('request_ids') and all(rid.startswith(cohort_prefix) for rid in b['request_ids'])]
        if not batches or not any(b.get('batch_size')==4 for b in batches):
            raise RuntimeError('No actual B4 microbatch was observed')
        if any(b.get('batch_size',0)>4 for b in batches):
            raise RuntimeError('B4 slot bound exceeded')
        observed={rid for b in batches for rid in b.get('request_ids',[])}
        if len(observed)!=8:
            raise RuntimeError('Two-slot cohort did not cover eight requests')
        groups=sorted({(b.get('pp_slot_id'),tuple(b.get('request_ids',[]))) for b in batches if b.get('batch_size')==4},key=repr)
        expected_groups={tuple(range(4)),tuple(range(4,8))}
        actual_groups={tuple(int(rid.rsplit('-',1)[1]) for rid in ids) for _,ids in groups}
        if len(groups)!=2 or {slot for slot,_ in groups}!={0,1}:
            raise RuntimeError('Expected exactly one stable B4 group per PP slot')
        for rid in observed:
            if len({b['pp_slot_id'] for b in batches if rid in b['request_ids']})!=1:
                raise RuntimeError('Request migrated between PP slots: '+rid)
        if actual_groups!=expected_groups:
            raise RuntimeError('FCFS ordered 2xB4 membership changed: '+str(actual_groups))
        admissions=[a for a in receipt.get('atomic_admissions',[]) if a['cohort'].startswith(cohort_prefix)]
        if len(admissions)!=1 or admissions[0].get('request_indices')!=list(range(8)):
            raise RuntimeError('Actual scheduler atomic B8 admission proof missing')
        composition.append(dict(rank=receipt['rank'],atomic_admission=admissions[0],
            slot_request_groups=[dict(slot_id=slot,request_ids=list(rids)) for slot,rids in groups]))
    return dict(passed=True,mode=mode,actual_slots=2,max_requests_per_slot=4,
                feedback_group='isolated' if mode=='candidate' else 'legacy',ranks=8,
                actual_slot_request_composition=composition)
