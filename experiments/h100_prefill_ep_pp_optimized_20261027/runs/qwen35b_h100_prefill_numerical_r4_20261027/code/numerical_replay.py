"""PP-only cold first-token logprob diagnostics with explicit allocator headroom."""
import hashlib
import os
from pathlib import Path
import time

from lookahead_workloads import byte
from previous_workloads import prompts


def install(base):
    previous_args=base.engine_args
    def engine_args(cfg, arm):
        args=previous_args(cfg,arm)
        if cfg.get('prefill_numerical_only'):
            args[args.index('--mem-fraction-static')+1]=str(cfg['validation_mem_fraction'])
        return args
    base.engine_args=engine_args

    def progress(run,cfg,stage,**extra):
        base.h.save(run/'task/progress.json',dict(experiment_id=cfg['id'],task_id='comparison',
            stage=stage,timestamp=time.time(),controller_pid=os.getpid(),
            valid_points=len(list(run.glob('*/workloads/*/validation.prefill_logprobs.json'))),
            expected=cfg['expected_points'],**extra))

    def workloads(run,cfg,arm,key,health):
        assert cfg['prefill_numerical_only'] and arm=='pp2'
        from benchmark import policy as expert_policy
        from sampling import set_diagnostic
        expert_policy(cfg,b'0');set_diagnostic(cfg,False)
        byte(cfg['transfer_audit_control'],ord('0'))
        for case in cfg['workloads']:
            current=dict(cfg,requests=case['requests'],prefix_tokens=case['new_input'])
            root=run/arm/'workloads'/case['id'];root.mkdir(parents=True,exist_ok=True)
            Path(cfg['expert_workload_path']).write_text(case['id']+'\n')
            byte(cfg['fixed_cohort_size'],case['requests']);byte(cfg['gap_policy_path'],ord('0'))
            response=base.h.http_json(cfg['port'],'/set_internal_state',key,
                dict(server_args=dict(pp_max_micro_batch_size=case['requests']//2)),timeout=30)
            if isinstance(response,dict) and response.get('success') is False:raise RuntimeError('Microbatch update failed')
            tokens=prompts(current,case)
            base.h.save(root/'protocol.json',dict(**case,cached_context=0,output_tokens=1,
                input_sha256=[hashlib.sha256(str(t).encode()).hexdigest() for t in tokens],
                performance_sample=False,validation_mem_fraction=cfg['validation_mem_fraction']))
            values={}
            for p in (0,1):
                byte(cfg['gap_policy_path'],ord(str(p)))
                packets=[dict(base.packet(current,arm,t,1,i),return_logprob=True,
                    logprob_start_len=case['new_input'],top_logprobs_num=10) for i,t in enumerate(tokens)]
                rounds=[]
                for repeat in range(2):
                    health();base.flush_cache(cfg,key)
                    progress(run,cfg,f'numerical_{case["id"]}_p{p}',repeat=repeat+1)
                    group=base.cohort(current,key,packets)
                    for r in group['requests']:
                        assert (r.get('cached_tokens') or 0)==0 and r['prompt_tokens']==case['new_input'] and r['output_tokens']==1
                        assert r['validation_logits']['output_top_logprobs']
                    rounds.append(group)
                values[str(p)]=rounds
            base.h.save(root/'validation.prefill_logprobs.json',dict(performance_sample=False,
                cached_context=0,new_input=case['new_input'],output_tokens=1,by_policy=values,
                validation_mem_fraction=cfg['validation_mem_fraction']))
            base.h.save(root/'completed.json',dict(passed=True,performance_sample=False))
        byte(cfg['gap_policy_path'],ord('0'));byte(cfg['fixed_cohort_control'],0)
        base.h.save(run/arm/'completed.json',dict(passed=True,performance_sample=False,workloads=len(cfg['workloads'])))

    base.progress=progress
    base.run_workloads=workloads
