"""Same-process scheduling A/Bs; semantic GPU prefixes and unprofiled timing."""
import hashlib
import os
from pathlib import Path
import statistics

MODES={'baseline':(False,False),'producer_ready':(False,True),
       'early_activation':(True,False),'early_and_ready':(True,True)}


def install(base):
    import hopper_reference
    original_env=base.environment

    def environment(cfg,arm,inherited):
        result=original_env(cfg,arm,inherited)
        result.update(PP_READY_POLICY_PATH=cfg['policy_path'],
                      PP_DECODE_OPT_POLICY_PATH=cfg['decode_opt_policy_path'])
        return result

    base.environment=environment
    base.engine_args=hopper_reference.engine_args

    def run_workloads(run,cfg,arm,key,health):
        import pp_ready_send
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

        def policy(mode):
            early,ready=MODES[mode]
            pp_ready_send.set_policy(cfg['decode_opt_policy_path'],early)
            pp_ready_send.set_policy(cfg['policy_path'],ready)

        def cached(result):
            if len(result['requests'])!=8: raise RuntimeError('Incomplete cohort')
            if any(r['cached_tokens']!=4096 or r['output_tokens']!=128 for r in result['requests']):
                raise RuntimeError('Expected exact cached4096 and128 real output tokens')

        def sample(mode,tag):
            health();policy(mode)
            base.progress(run,cfg,f'{arm}_{tag}_{mode}')
            result=base.cohort(cfg,key,decode);cached(result)
            row=dict(mode=mode,tag=tag,result=result,profiler_enabled=False)
            base.h.save(run/arm/f'cohort.{tag}.{mode}.json',row)
            return row

        policy('baseline');base.flush_cache(cfg,key)
        prime=base.cohort(cfg,key,pref)
        base.h.save(run/arm/'prefix_preparation.json',dict(real_model_prefix=True,measured=False,result=prime))
        # Exact local Decode shapes and two full cohorts warmed, outside timing.
        for count in [2,4,6,8,8]:
            health();warm=base.cohort(cfg,key,decode[:count])
            if any(r['cached_tokens']!=4096 for r in warm['requests']):
                raise RuntimeError('Warmup prefix mismatch')
        all_rows=[]
        if arm=='ep2':
            rows=[sample('baseline',f'formal_{i}') for i in range(3)]
            all_rows+=rows
            base.h.save(run/arm/'metrics.decode.baseline.json',base.summarize([r['result'] for r in rows],'GPUcached4096/new1/output128/B8'))
            profile_modes=['baseline']
        else:
            order=['baseline','producer_ready','early_activation','early_and_ready',
                   'early_and_ready','early_activation','producer_ready','baseline']
            pilot=[]
            for index,mode in enumerate(order):
                policy(mode);cached(base.cohort(cfg,key,decode))  # untimed mode warmup
                pilot.append(sample(mode,f'pilot_{index}'))
            all_rows+=pilot
            means={mode:statistics.mean(r['result']['output_tokens_per_s'] for r in pilot if r['mode']==mode) for mode in MODES}
            best=max((m for m in MODES if m!='baseline'),key=means.get)
            base.h.save(run/'pilot_selection.json',dict(means=means,candidate=best,
                criterion='Maximum pilot mean; independent confirmation follows, including negative results'))
            rows=[]
            for index,mode in enumerate(['baseline',best,best,'baseline','baseline',best]):
                policy(mode);cached(base.cohort(cfg,key,decode))
                rows.append(sample(mode,f'confirm_{index}'))
            all_rows+=rows
            for mode in MODES:
                chosen=[r['result'] for r in rows if r['mode']==mode]
                if not chosen: chosen=[r['result'] for r in pilot if r['mode']==mode]
                summary=base.summarize(chosen,'GPUcached4096/new1/output128/B8')
                summary.update(mode=mode,confirmation=mode in ('baseline',best))
                base.h.save(run/arm/f'metrics.decode.{mode}.json',summary)
            # No model math changed. Keep every A/A and A/B token sequence for audit.
            by_mode={m:[[r['output_ids'] for r in x['result']['requests']] for x in all_rows if x['mode']==m] for m in MODES}
            reference=by_mode['baseline'][0]
            base.h.save(run/arm/'output_agreement.json',dict(
                baseline_repeat_equal=all(x==reference for x in by_mode['baseline']),
                all_modes_equal=all(x==reference for values in by_mode.values() for x in values),
                complete_output_ids=all(all(x is not None and len(x)==128 for x in values) for group in by_mode.values() for values in group),
                outputs_by_mode=by_mode))
            profile_modes=['baseline',best]

        for mode in profile_modes:
            policy(mode);cached(base.cohort(cfg,key,decode));health()
            directory=run/arm/'traces'/mode;directory.mkdir(parents=True)
            base.profile_action(cfg,key,'/start_profile',dict(output_dir=str(directory),num_steps=40,
                activities=['CPU','GPU'],record_shapes=True,with_stack=False,
                profile_id=arm+'-'+mode,merge_profiles=False,detailed_annotations=True))
            result=base.cohort(cfg,key,decode);cached(result)
            base.h.save(directory/'cohort.json',dict(performance_sample=False,result=result))
            exports=base.await_exports(directory,health,timeout=240)
            if len(exports)!=2: raise RuntimeError('Need both actual rank exports')
            base.h.save(directory/'manifest.json',dict(files=exports,mode=mode,performance_sample=False))
        policy('baseline')
        base.h.save(run/arm/'completed.json',dict(passed=True,formal_or_pilot_cohorts=len(all_rows)))

    base.run_workloads=run_workloads
