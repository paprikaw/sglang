"""Same GPU-resident Decode workload, fixed graph policy, no timed Prefill."""
import hashlib
import time
from pathlib import Path
from previous_workloads import prompts
from lookahead_workloads import byte

def install(base):
    def overlap_state(cfg,key):
        info=base.h.http_json(cfg['port'],'/get_server_info',key,timeout=30)
        states=[s['experiment_decode_overlap'] for s in info['internal_states']]
        assert len(states)==2 and all(s['drained'] and s['policy']==cfg['decode_overlap_policy'] for s in states)
        return states

    def workloads(run,cfg,arm,key,health):
        from benchmark import policy as expert_policy
        from sampling import set_diagnostic
        import lookahead_workloads
        expert_policy(cfg,b'0');set_diagnostic(cfg,False)
        for case in cfg['workloads']:
            current=dict(cfg,requests=case['requests']);local=case['requests']//2
            root=run/arm/'workloads'/case['id'];root.mkdir(parents=True,exist_ok=True)
            Path(cfg['expert_workload_path']).write_text(case['id']+'\n')
            byte(cfg['fixed_cohort_size'],case['requests']);byte(cfg['gap_policy_path'],ord('0'))
            byte(cfg['transfer_audit_control'],ord('0'))
            if arm=='pp2':
                response=base.h.http_json(cfg['port'],'/set_internal_state',key,
                    dict(server_args=dict(pp_max_micro_batch_size=local)),timeout=30)
                assert not isinstance(response,dict) or response.get('success') is not False
            base.h.save(root/'server_info.json',base.h.safe_server_info(base.h.http_json(cfg['port'],'/get_server_info',key,timeout=30)))
            tokens=prompts(current,case)
            pref=[base.packet(current,arm,t,1,i) for i,t in enumerate(tokens)]
            dec=[base.packet(current,arm,t+[198],cfg['output_tokens'],i) for i,t in enumerate(tokens)]
            base.h.save(root/'protocol.json',dict(**case,cached_context_per_request=cfg['prefix_tokens'],
                new_input_tokens_per_request=1,output_tokens_per_request=cfg['output_tokens'],
                target_local_decode_tokens=local,prompt_sha256=[hashlib.sha256(str(t).encode()).hexdigest() for t in tokens],
                no_cpu_kv_restore=True,variant=cfg['variant'],graph_policy_fixed_before_capture=True))
            def check(group,cached=4096,output=128):
                assert len(group['requests'])==case['requests']
                for row in group['requests']:
                    assert row['output_tokens']==output and (row.get('cached_tokens') or 0)==cached
                    assert len(row.get('output_ids') or [])==output
            health();base.flush_cache(cfg,key)
            prime=base.cohort(current,key,pref);check(prime,0,1)
            base.h.save(root/'prefix_preparation.json',dict(performance_sample=False,real_model_prefix=True,
                exact_prefix_tokens_per_request=4096,cpu_kv_restore=False))
            if arm=='pp2':
                # Policy9 reuses schemas observed on the native Decode wire.
                # Prefill alone does not establish every typed Decode channel
                # or boundary shape; retain the established lookahead warmup.
                counts=sorted(set([2,4,6,8,local,case['requests']]))+[case['requests']]*2
                for count in counts:
                    health()
                    group=base.cohort(current,key,dec[:min(count,case['requests'])])
                    assert len(group['requests'])==min(count,case['requests'])
                    assert all(r['output_tokens']==128 and r['cached_tokens']==4096 for r in group['requests'])
                base.h.save(root/'native_transport_warmup.json',dict(performance_sample=False,
                    policy=0,requested_counts=counts,actual_counts=[min(c,case['requests']) for c in counts],
                    purpose='Warm native Decode schemas before policy9 typed NCCL transport'))
            if arm=='pp2':
                native_reference=[r['output_ids'] for r in group['requests']]
                base.h.save(root/'native_output_reference.json',dict(performance_sample=False,policy=0,output_ids=native_reference))
            if arm=='ep2':
                set_diagnostic(cfg,True);byte(run/'decode_overlap_verify.bin',ord('1'))
                probe=[base.packet(current,arm,t+[198],8,i) for i,t in enumerate(tokens)]
                result=base.cohort(current,key,probe);base.h.save(root/'numerical_probe.json',dict(performance_sample=False,result=result));check(result,4096,8)
                byte(run/'decode_overlap_verify.bin',ord('0'));set_diagnostic(cfg,False)
                states=overlap_state(cfg,key)
                quals=[base.h.load(p) if hasattr(base.h,'load') else __import__('json').loads(p.read_text())
                       for p in (root/'overlap_qualification').glob('rank*.json')]
                assert len(quals)==2 and all(len(q)==cfg['num_hidden_layers'] and all(r['passed'] for r in q.values()) for q in quals)
                base.h.save(root/'numerical_admission.json',dict(performance_sample=False,states=states,passed=True))
            policy=9 if arm=='pp2' else 0
            byte(cfg['gap_policy_path'],ord(str(policy)))
            for _ in range(3):
                health();warm=base.cohort(current,key,dec);check(warm)
                if arm=='pp2':
                    assert [r['output_ids'] for r in warm['requests']]==native_reference, 'PP lookahead changed the native fixed-group output'
            if arm=='pp2':base.h.save(root/'native_lookahead_equivalence.json',dict(passed=True,performance_sample=False,full_sequences_compared=case['requests']*3,output_tokens_per_request=128))
            values=[]
            for repeat in range(cfg['repeats']):
                health();base.progress(run,cfg,f'timing_{arm}_{case["id"]}',repeat=repeat+1)
                before=overlap_state(cfg,key) if arm=='ep2' else lookahead_workloads.states(run,cfg,policy) if hasattr(lookahead_workloads,'states') else None
                group=base.cohort(current,key,dec);check(group)
                if arm=='ep2':
                    after=overlap_state(cfg,key)
                    deltas=[sum(s['graph_replays'].values())-sum(b['graph_replays'].values()) for b,s in zip(before,after)]
                    assert all(n>0 for n in deltas)
                    group['overlap_receipts']=dict(states=after,graph_replay_deltas=deltas)
                else:
                    info=base.h.http_json(cfg['port'],'/get_server_info',key,timeout=30)
                    paths=[run/'pp2/gap_receipts'/f'policy9.pp{r}.json' for r in (0,1)]
                    from artifact_visibility import wait_for_stable_json
                    states=[wait_for_stable_json(p,timeout_s=20)[0] for p in paths]
                    assert all(s['ahead_produced']>0 and s['ahead_produced']==s['ahead_consumed'] and not s['pending'] and not s['ahead_pending'] for s in states)
                    group['scheduling_receipt']=dict(policy=9,states=states)
                values.append(group)
                base.h.save(root/'metrics.decode.json',base.summarize(values,cfg['protocol']))
            ref=[r['output_ids'] for r in values[0]['requests']]
            agreement=all([r['output_ids'] for r in g['requests']]==ref for g in values)
            base.h.save(root/'output_agreement.json',dict(all_formal_sequences_equal=agreement,output_tokens=128,
                requirement='PP fixed-group sequences must match; EP checks same-input per-layer native/serial/overlap equivalence separately',
                historical_native_ep_stable=False if arm=='ep2' else None))
            if arm=='pp2':assert agreement
            health();check(base.cohort(current,key,dec))
            directory=root/'traces/decode';directory.mkdir(parents=True,exist_ok=True)
            base.profile_action(cfg,key,'/start_profile',dict(output_dir=str(directory),num_steps=cfg['profile_steps'],
                activities=['CPU','GPU'],record_shapes=True,with_stack=False,
                profile_id=f'{arm}-{case["id"]}-{cfg["variant"]}',merge_profiles=False,detailed_annotations=True))
            group=base.cohort(current,key,dec);check(group)
            base.h.save(directory/'cohort.json',dict(performance_sample=False,result=group))
            files=base.await_exports(directory,health,timeout=240);assert len(files)==2
            base.h.save(directory/'manifest.json',dict(files=files,performance_sample=False,target_local_tokens=local,
                selected_forwards=cfg['selected_forwards'],drop_initial_graphs=cfg['drop_initial_graphs'],variant=cfg['variant']))
            byte(cfg['gap_policy_path'],ord('0'));byte(cfg['fixed_cohort_control'],0)
            base.h.save(root/'completed.json',dict(passed=True,formal_rounds=cfg['repeats']))
        base.h.save(run/arm/'completed.json',dict(passed=True,workloads=len(cfg['workloads'])))
    base.run_workloads=workloads
    def progress(run,cfg,stage,**extra):
        base.h.save(run/'task/progress.json',dict(experiment_id=cfg['id'],task_id='comparison',stage=stage,
            timestamp=time.time(),valid_points=len(list(run.glob('*/workloads/*/metrics.decode.json'))),expected=cfg['expected_points'],**extra))
    base.progress=progress
