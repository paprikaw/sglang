"""Four fresh consumers; immutable semantic bank imported outside timing."""
import json,statistics,time
from pathlib import Path
from common import save,GUARD
from workload import prefixes,query,prompts,cached_prefix_length,manifest
from spec_proof import validate_receipts

def stable_json(path,timeout=60):
    import sys
    sys.path.insert(0,str(GUARD))
    from artifact_visibility import wait_for_stable_json
    return wait_for_stable_json(path,timeout_s=timeout)[0]

def command(run,cfg,prefix,action,ident):
    out=run/cfg['arm'];save(out/'command.json',dict(run=cfg['id'],id=ident,prefix_length=prefix,action=action))
    deadline=time.monotonic()+600
    while True:
        paths=[out/'receipts'/f'{ident}_rank{rank}.json' for rank in range(8)]
        if all(path.exists() for path in paths):
            rows=[stable_json(path) for path in paths]
            if not all(row['passed'] for row in rows):raise RuntimeError('Idle native cache proof failed')
            return rows
        if any((out/node/'exit.json').exists() for node in cfg['nodes']):raise RuntimeError('Model worker exited during idle command '+ident)
        if time.monotonic()>deadline:raise TimeoutError('Missing idle receipts: '+ident)
        time.sleep(.25)

def stats(values):
    values=sorted(value for value in values if value is not None)
    return dict(mean=statistics.mean(values),median=statistics.median(values),p99=values[-1],p99_estimator='nearest_rank_N8') if values else None

def batch_request(url,cfg,cohort,*,prepared=True,output_tokens=64):
    from atomic_stream import make_payload,generate_batch
    payload=make_payload(prefixes(4096),query,cohort,output_tokens)
    rows,wall=generate_batch(url,payload,timeout=1800)
    for row in rows:
        meta=row['meta'];expected=4032 if prepared else 0
        if meta.get('prompt_tokens')!=4097 or meta.get('cached_tokens',0)!=expected:
            raise RuntimeError('Exact semantic input/cache length mismatch: '+str(meta))
        if prepared and (meta.get('cached_tokens_details',{}).get('host')!=4032 or meta.get('cached_tokens_details',{}).get('device')!=0):
            raise RuntimeError('Native cold CPU4032/GPU0 prefix was not used: '+str(meta))
        if prepared:
            verify=meta.get('spec_verify_ct',0)
            if (verify>0)!=cfg['mtp']:raise RuntimeError('Actual per-request speculative mode disagrees with arm')
        row.update(prefix_tokens=4096,query_tokens=1)
    return rows,wall

def scheduler_proof(receipts,cfg,prefix):
    if cfg['layout']=='ep8':
        from dp_proof import validate
        return validate(receipts,prefix)
    from pp_proof import validate
    return validate(receipts,cfg['mtp'],prefix)

def import_cold(run,cfg,ident):
    # Uniform reimport for all arms prevents plain radix warmup promoting4032
    # to4096 while MTP bigram remains4032. All source I/O is before timing.
    command(run,cfg,4096,'reset',ident+'_reset')
    imported=command(run,cfg,4096,'bank_import',ident+'_import')
    if not all(row.get('bank_import',{}).get('passed') for row in imported):raise RuntimeError('Semantic bank import missing')
    return imported

def produce_bank(url,run,cfg):
    out=run/cfg['arm'];command(run,cfg,4096,'reset','producer_reset')
    rows,wall=batch_request(url,cfg,'prepare-semantic',prepared=False,output_tokens=1)
    ready=command(run,cfg,4096,'ready','producer_ready')
    for receipt in ready:
        item=receipt['graph'].get('cohorts',{}).get('prepare-semantic',{})
        target=sum(value for key,value in item.get('forwards',{}).items() if key.startswith('target:') and 'EXTEND' in key)
        draft=item.get('phases',{}).get('draft_prefill',0)
        if not target or bool(draft)!=(receipt['rank']>=4):raise RuntimeError('Actual target/draft producer Prefill coverage missing')
    exported=command(run,cfg,4096,'bank_export','producer_export')
    if not all(row.get('bank_export',{}).get('passed') for row in exported):raise RuntimeError('Semantic bank export missing')
    result=dict(role='producer',passed=True,semantic_kv=True,synthetic_kv=False,actual_model_prefill=True,
        requests=rows,wall_s=wall,excluded_from_all_consumer_performance=True,ready_receipts=ready,export_receipts=exported,
        bank_dir=cfg['semantic_bank_dir'],publication_pending_producer_exit=True)
    save(out/'bank_build_results.json',result)
    return [result]

def profile_once(url,run,cfg):
    out=run/cfg['arm'];imported=import_cold(run,cfg,'profile_preparation')
    proof=command(run,cfg,4096,'profile_cold','profile_cold')
    if not all(row['profiler']['armed'] and row['profiler']['active'] for row in proof):raise RuntimeError('All ranks must arm profile before client release')
    rows,wall=batch_request(url,cfg,'profile-decode-4096')
    after=command(run,cfg,4096,'profile_finish','profile_after')
    graph=validate_receipts(after,cfg['layout'],'profile-',mtp=cfg['mtp'])
    scheduling=scheduler_proof(after,cfg,'profile-')
    activity=[]
    for rank in range(8):
        value=stable_json(out/f'activity_receipt_rank{rank}.json')
        if not value.get('passed') or value['rank']!=rank or value['target_forward_count']!=12 or not Path(value['trace']).is_file():
            raise RuntimeError('Bounded all-rank profile incomplete')
        activity.append(value)
    result=dict(arm=cfg['arm'],layout=cfg['layout'],mtp=cfg['mtp'],profile_only=True,
        performance_metrics_not_for_production_comparison=True,requests=rows,wall_s=wall,
        import_receipts=imported,native_cpu_ready_gpu_empty=proof,all_rank_after=after,
        actual_graph_proof=graph,actual_scheduler_proof=scheduling,activity_receipts=activity,
        profile_started_after_two_unprofiled_cohorts=True,source_bank_import_timed=False,hicache_h2d_timed=True)
    save(out/'p4096_decode_profile_results.json',result)
    return result

def measure_all(url,run,cfg):
    if cfg['role']=='producer':return produce_bank(url,run,cfg)
    out=run/cfg['arm']
    from semantic_bank import load_manifest
    bank=load_manifest(cfg['semantic_bank_dir'],prompts(),cfg['bank_model_identity'])
    imported=import_cold(run,cfg,'warm_preparation')
    warm,warm_wall=batch_request(url,cfg,'warm-decode-4096')
    save(out/'p4096_decode_warm_results.json',dict(requests=warm,wall_s=warm_wall,untimed=True,import_receipts=imported))
    production=[]
    for cohort in ('timed','repeat'):
        imported=import_cold(run,cfg,cohort+'_preparation')
        proof=command(run,cfg,4096,'cold',cohort+'_cold')
        if any(row['profiler']['armed'] or row['profiler']['active'] for row in proof):raise RuntimeError('Profile contaminated production')
        save(out/'TIMED',dict(case='p4096_decode',cohort=cohort,started_at=time.time()))
        try:rows,wall=batch_request(url,cfg,cohort+'-decode-4096')
        finally:(out/'TIMED').unlink(missing_ok=True)
        after=command(run,cfg,4096,'observe',cohort+'_after')
        graph=validate_receipts(after,cfg['layout'],cohort+'-',mtp=cfg['mtp'])
        scheduling=scheduler_proof(after,cfg,cohort+'-')
        row=dict(arm=cfg['arm'],layout=cfg['layout'],mtp=cfg['mtp'],workload='decode',request_count=8,cohort=cohort,
            engine_commit=cfg['engine_commit'],prefix_tokens_per_request=4096,total_context_tokens=32768,
            actual_cached_tokens_per_request=4032,prefix_tail_recomputed_tokens=64,first_forward_new_tokens_per_request=65,
            query_tokens_per_request=1,output_tokens_per_request=64,wall_s=wall,throughput_tok_s=512/wall,throughput_kind='output_tokens',
            ttft_s=stats([x['ttft_s'] for x in rows]),tpot_s=stats([x['tpot_s'] for x in rows]),end_to_end_s=stats([x['end_to_end_s'] for x in rows]),requests=rows,
            native_cpu_ready_gpu_empty=proof,all_rank_after=after,actual_graph_proof=graph,actual_scheduler_proof=scheduling,
            old_context_constructed_by_model=True,synthetic_kv=False,semantic_input_manifest=manifest(),
            persistent_source_kv_bank=True,bank_manifest_sha256=bank['manifest_sha256'],bank_identity_sha256=bank['bank_identity_sha256'],
            bank_import_receipts=imported,weights_loading_timed=False,source_bank_import_timed=False,hicache_h2d_timed=True,
            per_cohort_reimport_reason='Keep plain and bigram native CPU prefix exactly4032 after warm/repeat',
            spec_accept_length=stats([x['meta'].get('spec_accept_length') for x in rows]) if cfg['mtp'] else None,
            spec_accept_rate=stats([x['meta'].get('spec_accept_rate') for x in rows]) if cfg['mtp'] else None,
            spec_verify_ct=stats([x['meta'].get('spec_verify_ct') for x in rows]) if cfg['mtp'] else None,
            cuda_profiler_off_during_production=True,profile_only_performance_not_production_measurement=False,
            p99_is_eight_request_sample=True,atomic_http_batch=True,requested_fcfs_groups=[[0,1,2,3],[4,5,6,7]],
            strict_cross_topology_token_equivalence_not_assumed=True,acceptance_scope='Frozen real semantic corpus; not corpus-wide quality claim')
        production.append(row)
        save(out/('p4096_decode_results.json' if cohort=='timed' else 'p4096_decode_repeat_results.json'),row)
    from atomic_stream import output_parity
    repeatability=output_parity(production[0]['requests'],production[1]['requests'])
    repeatability.update(both_cohorts_unprofiled=True,both_start_native_cpu_ready_gpu_empty=True,
        same_serving_process=True,same_immutable_bank=True)
    save(out/'p4096_decode_repeatability.json',repeatability)
    production[0]['repeatability']=repeatability
    production[0]['repeat_performance']={key:production[1][key] for key in ('wall_s','throughput_tok_s','ttft_s','tpot_s','end_to_end_s','spec_accept_rate')}
    save(out/'p4096_decode_results.json',production[0])
    if not repeatability['passed']:raise RuntimeError('Within-arm exact generated ID/text repeatability failed; measurements preserved')
    profile_once(url,run,cfg)
    return [production[0]]
