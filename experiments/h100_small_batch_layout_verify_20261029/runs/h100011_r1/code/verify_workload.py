"""One controlled Decode experiment: real full models and GPU-hot prefixes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import threading
import time


def byte(path,value):
    fd=os.open(path,os.O_WRONLY|os.O_NOFOLLOW)
    try:os.pwrite(fd,bytes([value]),0)
    finally:os.close(fd)


def install(base):
    def engine_args(cfg,arm,replica):
        row=base.layout(cfg,arm);ep=row['ep'];pp=row['pp']
        running=8*ep if pp==1 else 32//row['replicas']
        args=['--model-path',cfg['model_path'],'--host','127.0.0.1','--port',str(base.port(cfg,arm,replica)),
          '--dist-init-addr',f"127.0.0.1:{cfg['dist_port']+cfg['layouts'].index(row)*100+replica*10}",
          '--tp-size',str(ep),'--ep-size',str(ep),'--dp-size',str(ep),'--pp-size',str(pp),
          '--dtype','bfloat16','--kv-cache-dtype','bfloat16','--context-length','8192',
          '--mem-fraction-static','.90','--page-size','64','--max-total-tokens','49152',
          '--max-running-requests',str(running),'--max-mamba-cache-size',str(16*ep*pp),
          '--mamba-max-states-per-path','1','--mamba-ssm-dtype','float32',
          '--cuda-graph-max-bs-decode','16','--cuda-graph-backend-prefill','disabled',
          '--chunked-prefill-size',str(4096*ep),'--max-prefill-tokens',str(4096*ep),
          '--attention-backend','fa3','--linear-attn-backend','triton',
          '--moe-runner-backend','deep_gemm','--moe-a2a-backend','deepep' if ep>1 else 'none',
          '--disable-custom-all-reduce','--skip-server-warmup',
          '--enable-cache-report','--json-model-override-args','{"language_model_only":true}']
        if pp>1:args+=['--pp-max-micro-batch-size','8','--pp-async-batch-depth','0']
        if ep>1:args+=['--enable-dp-attention','--deepep-mode','auto','--deepep-dispatcher-output-dtype','bf16']
        return args
    base.engine_args=engine_args

    def states(cfg,arm,key):
        return base.each_server(cfg,arm,lambda c:base.h.http_json(c['port'],'/get_server_info',key,timeout=30))

    def counts(infos):
        rows=[]
        for replica,info in enumerate(infos):
            for s in info['internal_states']:
                rows.append(dict(replica=replica,shape=s['experiment_parallel_shape'],counts=s['experiment_forward_counts']))
        return rows

    def summarize(rounds):
        metrics={}
        for name,fn in [
            ('mean_request_tpot_ms',lambda g:statistics.mean(r['tpot_ms'] for r in g['requests'])),
            ('mean_request_e2e_ms',lambda g:statistics.mean(r['e2e_s']*1000 for r in g['requests'])),
            ('mean_request_ttft_ms',lambda g:statistics.mean(r['ttft_s']*1000 for r in g['requests'])),
            ('cohort_makespan_ms',lambda g:g['makespan_s']*1000),
            ('output_tokens_per_s',lambda g:g['output_tokens_per_s'])]:
            values=[fn(g) for g in rounds]
            metrics[name]=dict(mean=statistics.mean(values),samples=values,stdev=statistics.stdev(values) if len(values)>1 else 0)
        return dict(rounds=rounds,metrics=metrics)

    def workload(run,cfg,arm,key,case,health):
        row=base.layout(cfg,arm);current=dict(cfg,requests=case['requests'])
        out=run/'results'/arm/case['id'];out.mkdir(parents=True,exist_ok=True)
        from sampling import make_tokens
        tokens=make_tokens(current)
        base.h.save(out/'protocol.json',dict(**case,global_new_tokens_per_decode_step=case['requests'],
            P=4096,Q=1,O=128,cpu_kv_restore=False,hicache=False,gpu_hot=True,
            prompt_sha256=[hashlib.sha256(str(t).encode()).hexdigest() for t in tokens],
            pp_slots=row['pp'],target_microbatch=case['requests']//4,
            pp_policy=9 if row['pp']==2 else 0))
        byte(cfg['gap_policy_path'],ord('0'));byte(cfg['fixed_cohort_control'],0)
        byte(cfg['diagnostic_eager_policy_path'],ord('0'));byte(cfg['fragment_policy_path'],ord('0'))
        if row['pp']>1:
            answers=base.each_server(cfg,arm,lambda c:base.h.http_json(c['port'],'/set_internal_state',key,
                dict(server_args=dict(pp_max_micro_batch_size=case['requests']//4)),timeout=30))
            base.h.save(out/'microbatch_update.json',answers)
        base.flush(cfg,arm,key);health()
        prime=base.cohort(current,arm,key,tokens,1)
        assert len(prime['requests'])==case['requests']
        assert all((r.get('cached_tokens') or 0)==0 for r in prime['requests'])
        base.h.save(out/'prefix_preparation.json',dict(performance_sample=False,result=prime))
        decode=[t+[198] for t in tokens]
        def check(group):
            assert len(group['requests'])==case['requests']
            for r in group['requests']:
                assert r['prompt_tokens']==4097 and r['output_tokens']==128 and r['cached_tokens']==4096
                assert len(r['output_ids'])==128
                if r.get('cached_tokens_details'):
                    assert r['cached_tokens_details'].get('host',0)==0 and r['cached_tokens_details'].get('storage',0)==0
        # Native PP wire schemas and all selected graph sizes are warmed first.
        for n in sorted(set([4,8,case['requests']//2,case['requests']]))+[case['requests']]*2:
            health();group=base.cohort(current,arm,key,decode[:min(n,case['requests'])],128)
            assert all(r['cached_tokens']==4096 for r in group['requests'])
        if row['pp']==2:
            byte(cfg['fixed_cohort_size'],case['requests']//2)
            byte(cfg['gap_policy_path'],ord('9'))
        epoch=1
        for _ in range(3):
            if row['pp']==2:byte(cfg['fixed_cohort_control'],epoch);epoch+=1
            health();check(base.cohort(current,arm,key,decode,128))
        formal=[]
        for repeat in range(cfg['repeats']):
            health();base.progress(run,cfg,f'timing_{arm}_{case["id"]}',repeat=repeat+1)
            before=counts(states(cfg,arm,key))
            if row['pp']==2:byte(cfg['fixed_cohort_control'],epoch);epoch+=1
            group=base.cohort(current,arm,key,decode,128);check(group)
            after=counts(states(cfg,arm,key))
            delta=[]
            assert len(before)==len(after)==4
            for b,a in zip(before,after):
                assert b['shape']==a['shape'] and b['replica']==a['replica']
                d={k:v-b['counts'].get(k,0) for k,v in a['counts'].items() if v-b['counts'].get(k,0)}
                assert any(k.startswith('DECODE:') for k in d),d
                assert any(k.startswith('graph_eligible:') for k in d),d
                delta.append(dict(replica=a['replica'],shape=a['shape'],counts=d))
            group['forward_deltas']=delta
            formal.append(group)
            base.h.save(out/f'formal.r{repeat+1}.json',dict(performance_sample=True,result=group))
            base.h.save(out/'metrics.decode.json',summarize(formal))
        ref=[r['output_ids'] for r in formal[0]['requests']]
        agreement=all([r['output_ids'] for r in g['requests']]==ref for g in formal)
        base.h.save(out/'output_agreement.json',dict(all_formal_sequences_equal=agreement,
            requirement='PP2 fixed-group repeatability; cross-layout numerical batch effects may change generated sequences.'))
        if row['pp']==2:assert agreement
        byte(cfg['gap_policy_path'],ord('0'));byte(cfg['fixed_cohort_control'],0)
        base.h.save(out/'completed.json',dict(passed=True,formal_rounds=cfg['repeats']))

    def worker(run,cfg):
        incoming=json.loads(base.sys.stdin.readline());env=dict(os.environ,**incoming['environment'])
        base.check(run,cfg,env,True)
        def interrupted(signum,frame):raise RuntimeError('Guard/controller requested cleanup')
        signal.signal(signal.SIGTERM,interrupted)
        def watch_parent():base.sys.stdin.read();os.kill(os.getpid(),signal.SIGTERM)
        threading.Thread(target=watch_parent,daemon=True).start();key=base.h.read_key(cfg)
        topo=subprocess.run(['nvidia-smi','topo','-m'],capture_output=True,text=True,timeout=20)
        (run/'topology.txt').write_text(topo.stdout+topo.stderr)
        def health():base.check(run,cfg,env,True)
        for row in cfg['layouts']:
            arm=row['id'];children=[]
            try:
                base.progress(run,cfg,'loading_'+arm)
                for replica in range(row['replicas']):children.append(base.launch(run,cfg,arm,replica,env,key))
                deadline=time.monotonic()+cfg['startup_timeout_s']
                for replica,child in enumerate(children):
                    while True:
                        health()
                        if child['process'].poll() is not None:raise RuntimeError(arm+f'.r{replica} exited during startup')
                        try:info=base.h.http_json(base.port(cfg,arm,replica),'/get_server_info',key,timeout=3);break
                        except Exception:
                            if time.monotonic()>deadline:raise TimeoutError('Full model startup timeout')
                            time.sleep(2)
                    base.h.save(run/'evidence'/f'{arm}.r{replica}'/'server_info.json',base.h.safe_server_info(info))
                    qs=list((run/'evidence'/f'{arm}.r{replica}'/('ep2' if row['ep']>1 else 'pp2')/'qualification').glob('*.json'))
                    assert len(qs)==row['ep']*row['pp'] and all(json.loads(p.read_text())['passed'] for p in qs)
                    assert all(not s.get('enable_hierarchical_cache',False) for s in info['internal_states'])
                for case in cfg['workloads']:workload(run,cfg,arm,key,case,health)
            finally:
                if children:
                    evidence=base.h.cleanup(children);base.h.save(run/'evidence'/arm/'cleanup.json',evidence)
                    if not evidence['cleanup_verified']:raise RuntimeError('Owned process cleanup failed')
        return 0
    base.worker=worker
    def progress(run,cfg,stage,**extra):
        base.h.save(run/'task/progress.json',dict(experiment_id=cfg['id'],task_id='comparison',stage=stage,
            timestamp=time.time(),controller_pid=json.loads((run/'task/controller_identity.json').read_text())['pid']
            if (run/'task/controller_identity.json').exists() else os.getpid(),
            valid_points=len(list(run.glob('results/*/*/completed.json'))),expected=cfg['expected_points'],**extra))
    base.progress=progress
