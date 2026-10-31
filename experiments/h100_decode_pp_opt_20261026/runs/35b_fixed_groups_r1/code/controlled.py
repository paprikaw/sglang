"""Same request groups, unchanged kernels: EP reference and PP scheduling A/B."""
import hashlib
import json
import os
from pathlib import Path
import time


def write_byte(path,value):
    fd=os.open(path,os.O_WRONLY|os.O_NOFOLLOW)
    try:
        if os.write(fd,bytes([value]))!=1:raise RuntimeError('Policy write failed')
    finally:os.close(fd)


def install(base):
    import hopper_reference
    original_env=base.environment
    original_cohort=base.cohort
    epochs=[0]

    def environment(cfg,arm,inherited):
        result=original_env(cfg,arm,inherited)
        result.update(PP_READY_POLICY_PATH=cfg['policy_path'],
            PP_DECODE_OPT_POLICY_PATH=cfg['decode_opt_policy_path'],
            PP_TRANSFER_AUDIT_CONTROL=cfg['transfer_audit_control'],
            PP_FIXED_COHORT_CONTROL=cfg['fixed_cohort_control'])
        return result

    def cohort(cfg,key,payloads):
        arm=os.environ.get('EP2_PP2_ARM')  # controller has no arm environment
        # EP packets carry explicit DP ranks, PP packets do not.
        is_pp=not any('data_parallel_rank' in p for p in payloads)
        if is_pp and len(payloads)==8:
            epochs[0]+=1
            if epochs[0]>=255:raise RuntimeError('Fixed cohort epoch bound reached')
            write_byte(cfg['fixed_cohort_control'],epochs[0])
        else:write_byte(cfg['fixed_cohort_control'],0)
        result=original_cohort(cfg,key,payloads)
        # Outside client timing: finish original feedback before changing policy.
        info=None
        for _ in range(3):info=base.h.http_json(cfg['port'],'/get_server_info',key,timeout=30)
        if is_pp and len(payloads)==8:
            receipts=[s.get('experiment_fixed_group_receipt') for s in info['internal_states']
                      if s.get('experiment_fixed_group_receipt')]
            if len(receipts)!=1 or receipts[0]['epoch']!=epochs[0] or receipts[0]['ordered_markers']!=[
                    5000,5002,5004,5006,5001,5003,5005,5007]:
                raise RuntimeError('Fixed native PP admission was not acknowledged')
            result['fixed_group_receipt']=receipts[0]
        return result

    base.environment=environment;base.engine_args=hopper_reference.engine_args
    base.cohort=cohort

    def run_workloads(run,cfg,arm,key,health):
        import pp_ready_send
        from pp_transfer_audit import compare_transfers
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(cfg['model_path'],local_files_only=True)
        tokens=[]
        for index in range(8):
            head=[5000+index]+tokenizer.encode(f'Document {index}. ',add_special_tokens=False)
            body=tokenizer.encode('A distributed service keeps a record of independent requests. '
                'Explain the computation and communication clearly.\n',add_special_tokens=False)
            tokens.append((head+body*(4096//len(body)+2))[:4096])
        pref=[base.packet(cfg,arm,t,1,i) for i,t in enumerate(tokens)]
        decode=[base.packet(cfg,arm,t+[198],128,i) for i,t in enumerate(tokens)]
        base.h.save(run/arm/'prompt_fingerprints.json',dict(prefix_tokens=4096,
            sha256=[hashlib.sha256(str(t).encode()).hexdigest() for t in tokens]))
        write_byte(cfg['transfer_audit_control'],ord('0'))
        def policy(mode):
            pp_ready_send.set_policy(cfg['decode_opt_policy_path'],mode=='early_activation')
            pp_ready_send.set_policy(cfg['policy_path'],False)
        def cached(result):
            if len(result['requests'])!=8 or any(r['cached_tokens']!=4096 or r['output_tokens']!=128
                    for r in result['requests']):raise RuntimeError('Workload/cache mismatch')
        policy('baseline');base.flush_cache(cfg,key)
        base.h.save(run/arm/'prefix_preparation.json',dict(real_model_prefix=True,measured=False,
            result=base.cohort(cfg,key,pref)))
        for count in (2,4,6,8,8):
            health();warm=base.cohort(cfg,key,decode[:count])
            if any(r['cached_tokens']!=4096 for r in warm['requests']):raise RuntimeError('Warmup mismatch')
        modes=['baseline']*3 if arm=='ep2' else ['baseline','early_activation','early_activation',
                                                'baseline','baseline','early_activation']
        rows=[]
        for index,mode in enumerate(modes):
            health();policy(mode);cached(base.cohort(cfg,key,decode))
            base.progress(run,cfg,f'{arm}_fixed_groups_{index}_{mode}')
            result=base.cohort(cfg,key,decode);cached(result)
            row=dict(mode=mode,tag=f'formal_{index}',result=result,profiler_enabled=False,
                fixed_pp_admission=True,waiting_included_in_client_latency=True)
            base.h.save(run/arm/f'cohort.formal_{index}.{mode}.json',row);rows.append(row)
        for mode in sorted(set(modes)):
            values=[r['result'] for r in rows if r['mode']==mode]
            summary=base.summarize(values,'GPUcached4096/new1/output128/B8; fixed even/odd request groups')
            summary.update(mode=mode,confirmation=True,request_groups_fixed=True)
            base.h.save(run/arm/f'metrics.decode.{mode}.json',summary)
        by_mode={mode:[[r['output_ids'] for r in row['result']['requests']] for row in rows
                      if row['mode']==mode] for mode in sorted(set(modes))}
        reference=by_mode['baseline'][0]
        base.h.save(run/arm/'output_agreement.json',dict(
            baseline_repeat_equal=all(x==reference for x in by_mode['baseline']),
            all_modes_equal=all(x==reference for group in by_mode.values() for x in group),
            complete_output_ids=all(all(x is not None and len(x)==128 for x in cohort)
                                   for group in by_mode.values() for cohort in group),
            outputs_by_mode=by_mode))
        # Two independent diagnostic cohorts; cloning/hashing is NEVER timed.
        if arm=='pp2':
            for epoch,mode in enumerate(('baseline','early_activation'),1):
                policy(mode);cached(base.cohort(cfg,key,decode))
                write_byte(cfg['transfer_audit_control'],ord(str(epoch)))
                result=base.cohort(cfg,key,decode);cached(result)
                write_byte(cfg['transfer_audit_control'],ord('0'))
                paths=[run/arm/'transfer_audit'/f'epoch{epoch}.pp{rank}.json' for rank in (0,1)]
                deadline=time.monotonic()+45
                while not all(p.is_file() for p in paths):
                    health();base.h.http_json(cfg['port'],'/get_server_info',key,timeout=30)
                    if time.monotonic()>deadline:raise TimeoutError('Capture receipts missing')
                    time.sleep(.05)
                receipts=[json.loads(p.read_text()) for p in paths]
                audit=compare_transfers(receipts)
                expected={(5000,5002,5004,5006),(5001,5003,5005,5007)}
                for receipt in receipts:
                    groups={tuple(r['marker'] for r in batch['requests']) for batch in receipt['batches']
                        if len(batch['requests'])==4 and all(8<=r['generated']<=112 for r in batch['requests'])}
                    if groups!=expected:raise RuntimeError('Actual steady PP groups differ from EP')
                if not audit['passed']:raise RuntimeError('Payload comparison failed')
                base.h.save(run/arm/f'qualification.epoch{epoch}.json',dict(epoch=epoch,mode=mode,
                    performance_sample=False,cohort=result,audit=audit,fixed_actual_groups=True))
        for mode in sorted(set(modes)):
            policy(mode);cached(base.cohort(cfg,key,decode));health()
            directory=run/arm/'traces'/mode;directory.mkdir(parents=True)
            base.profile_action(cfg,key,'/start_profile',dict(output_dir=str(directory),num_steps=40,
                activities=['CPU','GPU'],record_shapes=True,with_stack=False,
                profile_id=arm+'-'+mode,merge_profiles=False,detailed_annotations=True))
            result=base.cohort(cfg,key,decode);cached(result)
            base.h.save(directory/'cohort.json',dict(performance_sample=False,result=result))
            exports=base.await_exports(directory,health,timeout=240)
            if len(exports)!=2:raise RuntimeError('Need both actual rank profiles')
            base.h.save(directory/'manifest.json',dict(files=exports,mode=mode,performance_sample=False))
        policy('baseline');write_byte(cfg['fixed_cohort_control'],0)
        base.h.save(run/arm/'completed.json',dict(passed=True,formal_cohorts=len(rows),
            request_groups_fixed=True))

    base.run_workloads=run_workloads
