"""Four fresh consumers; immutable semantic bank imported outside timing."""
import json,statistics,time
from pathlib import Path
from common import save,GUARD,ROOT
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
            if cfg.get('prefill_first_proposal'):
                from pp_prefill_seed import validate_receipts as validate_seed
                validate_seed(rows,require_work=action=='observe')
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
    timeout=cfg.get('request_timeout_s', (90 if cohort.startswith('warm-') else 180) if cfg.get('profile_only_followup') else 1800)
    rows,wall=generate_batch(url,payload,timeout=timeout)
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
    source_parity=None
    if cfg.get('profile_only_followup'):
        from atomic_stream import output_parity
        source=cfg['profile_source']['arms'][cfg['arm']]['production_cohorts'][0]
        original=stable_json(source['path'])
        source_parity=output_parity(original['requests'],rows)
        save(out/'profile_source_output_parity.json',source_parity)
        if source_parity.get('passed') is not True:
            raise RuntimeError('Profile-only follow-up output IDs/text differ from original arm production')
    graph=validate_receipts(after,cfg['layout'],'profile-',mtp=cfg['mtp'])
    scheduling=scheduler_proof(after,cfg,'profile-')
    activity=[]
    for rank in range(8):
        value=stable_json(out/f'activity_receipt_rank{rank}.json')
        if not value.get('passed') or value['rank']!=rank or value['target_forward_count']!=12 or not Path(value['trace']).is_file():
            raise RuntimeError('Bounded all-rank profile incomplete')
        activity.append(value)
    # Audit actual per-replay external Event intervals after the diagnostic ends.
    # GPU stream spans include dependency waits and event overhead, not kernel
    # busy time or a partition of the original source production TPOT.
    activity_summary=audit_activity(out)
    save(out/'stage_activity_summary.json',activity_summary)
    result=dict(arm=cfg['arm'],layout=cfg['layout'],mtp=cfg['mtp'],profile_only=True,
        performance_metrics_not_for_production_comparison=True,requests=rows,wall_s=wall,
        import_receipts=imported,native_cpu_ready_gpu_empty=proof,all_rank_after=after,
        actual_graph_proof=graph,actual_scheduler_proof=scheduling,activity_receipts=activity,
        profile_started_after_two_unprofiled_cohorts=True,source_bank_import_timed=False,hicache_h2d_timed=True,
        unprofiled_cohorts_are_source_run=bool(cfg.get('profile_only_followup')),
        source_performance=cfg.get('profile_source', {}).get('arms', {}).get(cfg['arm']),
        source_production_output_parity=source_parity,
        profile_valid=True,all_rank_gpu_activity_audited=True,
        measurement='external_cuda_event_stream_spans',cupti_enabled=False,diagnostic_cadence_changed=True,
        no_production_tpot_partition=True,
        activity_summary=str(out/'stage_activity_summary.json'))
    save(out/'p4096_decode_profile_results.json',result)
    return result

def audit_activity(out):
    import importlib.util
    spec=importlib.util.spec_from_file_location('fourway_saved_activity_audit',ROOT/'analysis/event_spans.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.analyze_arm(out,ranks=range(8))

def finish_diagnostics(url,run,cfg,production):
    """Only a profile failure may be isolated after both performance gates.

    Called only after timing, Graph/scheduler/source validation, and exact
    repeatability have succeeded. Any earlier failure still propagates.
    """
    if len(production)!=2 or not production[0].get('repeatability',{}).get('passed'):
        raise RuntimeError('Cannot isolate diagnostics before valid repeated production')
    out=run/cfg['arm'];failure=None
    try:
        flow_once(url,run,cfg,production[0])
        # This run qualifies all-forward unfenced flow only, not new A/E components.
        # Reuse the independently saved component profiles without adding replay fences.
        from flow_validation import validate_all
        validation=validate_all([stable_json(out/f'unfenced_flow_rank{rank}.json') for rank in range(8)],cfg['layout'])
        save(out/'flow_validation.json',validation)
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error),profile_only=True,
            performance_preserved=True,performance_repeated_and_valid=True,
            source_bank_import_timed=False,profiling_failure_not_a_performance_result=True)
        save(out/'profile_failure.json',failure)
    for index,row in enumerate(production):
        row.update(performance_valid=True,profile_valid=failure is None,profile_failure=failure,
                   profile_kind='unfenced_all_forward_timeline',new_component_profile=False)
        save(out/('p4096_decode_results.json' if index==0 else 'p4096_decode_repeat_results.json'),row)
    return [production[0]]


def flow_once(url,run,cfg,production):
    """Independent actual-graph cadence trace; never read/fence per replay."""
    import_cold(run,cfg,'flow_preparation')
    before=command(run,cfg,4096,'flow_cold','flow_cold')
    if not all(row['unfenced_flow']['armed'] and not row['profiler']['armed'] for row in before):
        raise RuntimeError('Unfenced flow mixed with fenced component profile')
    rows,wall=batch_request(url,cfg,'flow-decode-4096')
    after=command(run,cfg,4096,'flow_finish','flow_after')
    graph_proof=validate_receipts(after,cfg['layout'],'flow-',mtp=cfg['mtp'])
    placement_proof=scheduler_proof(after,cfg,'flow-')
    from flow_validation import validate_cpu_bank_placement
    validate_cpu_bank_placement(rows,cfg['layout'])
    from atomic_stream import output_parity
    parity=output_parity(production['requests'],rows)
    if not parity['passed']:raise RuntimeError('Flow diagnostic changed generated output')
    traces=[stable_json(run/cfg['arm']/f'unfenced_flow_rank{rank}.json') for rank in range(8)]
    if not all(row['passed'] and row['per_replay_fences']==0 for row in traces):
        raise RuntimeError('Unfenced flow incomplete')
    save(run/cfg['arm']/'unfenced_flow_results.json',dict(passed=True,requests=rows,wall_s=wall,
        diagnostic_only=True,output_parity=parity,all_rank_after=after,
        actual_graph_proof=graph_proof,actual_scheduler_proof=placement_proof,
        unprofiled_wall_s=production['wall_s'],per_replay_fences=0,
        trace_overhead_ratio=wall/production['wall_s'],
        no_gpu_idle_fraction_claim=True))

def warm_profile_followup(url,run,cfg):
    """Short untimed warmup before the first CUPTI session, not production."""
    imported=import_cold(run,cfg,'followup_warm_preparation')
    before=command(run,cfg,4096,'cold','followup_warm_cold')
    if any(row['profiler']['armed'] or row['profiler']['active'] for row in before):
        raise RuntimeError('Profile-only warmup must precede the first collector')
    rows,wall=batch_request(url,cfg,'warm-decode-4096',output_tokens=8)
    after=command(run,cfg,4096,'observe','followup_warm_after')
    graph=validate_receipts(after,cfg['layout'],'warm-',mtp=cfg['mtp'])
    scheduling=scheduler_proof(after,cfg,'warm-')
    if any(row.get('humming_validation',{}).get('passed') is not True for row in after):
        raise RuntimeError('Profile-only warmup Humming invariant proof missing')
    result=dict(arm=cfg['arm'],layout=cfg['layout'],mtp=cfg['mtp'],untimed=True,
        output_tokens_per_request=8,request_count=8,requests=rows,wall_s=wall,
        purpose='Warm request/sampling/speculative wrappers before the formal profile; not a production rerun',
        actual_graph_proof=graph,actual_scheduler_proof=scheduling,
        native_cpu_ready_gpu_empty=before,all_rank_after=after,bank_import_receipts=imported,
        source_bank_import_timed=False,cuda_profiler_active=False,
        formal_profile_requires_fresh_reimport=True,production_metric=False)
    save(run/cfg['arm']/'profile_followup_warm_results.json',result)
    return result


def profile_followup(url,run,cfg):
    """Short untimed warmup then missing diagnostic; no production reruns."""
    from profile_source import validate_source_once
    source=validate_source_once(cfg, arm=cfg['arm'])
    warm=warm_profile_followup(url,run,cfg)
    failure=None;profile=None
    try:
        profile=profile_once(url,run,cfg)
        if profile.get('profile_valid') is not True:
            raise RuntimeError('Strict GPU profile audit missing')
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error),profile_only=True,
            source_performance_preserved=True,profiling_failure_not_a_performance_result=True)
        save(run/cfg['arm']/'profile_failure.json',failure)
    # Do not copy prior metrics into a fresh measurement result. The saved
    # source descriptors identify their original run and exact byte hashes.
    result=dict(arm=cfg['arm'],layout=cfg['layout'],mtp=cfg['mtp'],profile_only_followup=True,
        source_performance_valid=True,source_performance=source,
        performance_valid=None,new_production_cohorts=0,profile_valid=failure is None,
        untimed_warm_output_tokens_per_request=8,warm_result_path=str(run/cfg['arm']/'profile_followup_warm_results.json'),
        profile_failure=failure,profile_result_path=str(run/cfg['arm']/'p4096_decode_profile_results.json') if profile else None)
    save(run/cfg['arm']/'profile_followup_results.json',result)
    return [result]


def measure_all(url,run,cfg):
    if cfg.get('profile_only_followup'):
        return profile_followup(url,run,cfg)
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
    return finish_diagnostics(url,run,cfg,production)
