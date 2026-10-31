"""Guard-owned heterogeneous resident instances and reproducible open-loop requests."""
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
    def instance(cfg,arm,replica):return base.layout(cfg,arm)['instances'][replica]
    def engine_args(cfg,arm,replica):
        inst=instance(cfg,arm,replica);ep,pp=inst['ep'],inst['pp'];p=cfg['profile']
        running=p['max_requests']*len(inst['gpus'])//4
        args=['--model-path',cfg['model_path'],'--host','127.0.0.1','--port',str(base.port(cfg,arm,replica)),
          '--dist-init-addr',f"127.0.0.1:{cfg['dist_port']+cfg['layouts'].index(base.layout(cfg,arm))*100+replica*10}",
          '--tp-size',str(ep),'--ep-size',str(ep),'--dp-size',str(ep),'--pp-size',str(pp),
          '--dtype','bfloat16','--kv-cache-dtype','bfloat16','--context-length',str(p['max_context']),
          '--mem-fraction-static',str(cfg['mem_fraction_static']),'--page-size','64',
          '--max-running-requests',str(running),'--max-mamba-cache-size',str(p['mamba_equiv']*ep*pp),
          '--max-total-tokens',str(p['kv_token_cap_per_full_gpu']*pp),
          '--mamba-max-states-per-path','1','--mamba-ssm-dtype','float32',
          '--cuda-graph-max-bs-decode',str(p['max_requests']//4),'--cuda-graph-backend-prefill','disabled',
          '--chunked-prefill-size',str(4096*ep),'--max-prefill-tokens',str(4096*ep),
          '--attention-backend','fa3','--linear-attn-backend','triton',
          '--moe-runner-backend','deep_gemm','--moe-a2a-backend','deepep' if ep>1 else 'none',
          '--disable-custom-all-reduce','--skip-server-warmup','--enable-cache-report',
          '--json-model-override-args','{"language_model_only":true}']
        if pp>1:args+=['--pp-max-micro-batch-size',str(cfg['pp_stream_microbatch']),'--pp-async-batch-depth','0']
        if ep>1:args+=['--enable-dp-attention','--deepep-mode','auto','--deepep-dispatcher-output-dtype','bf16']
        return args
    base.engine_args=engine_args

    def launch(run,cfg,arm,replica,inherited,key):
        row=base.layout(cfg,arm);inst=instance(cfg,arm,replica);alias='ep2' if inst['ep']>1 else 'pp2'
        inherited=dict(inherited,BENCH_PP2_LOOKAHEAD='1' if inst['pp']==2 else '0')
        env=base.b.environment(cfg,alias,inherited)
        evidence=run/'evidence'/cfg['node']/f'{arm}.r{replica}';evidence.mkdir(parents=True,exist_ok=True)
        env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,inst['gpus'])),EP2_PP2_RUN=str(evidence),
            EP_STEADY_HARDWARE='H100',PP_READY_POLICY_PATH=str(run/'pp_ready_policy.bin'),
            STEADY_EAGER_POLICY_PATH=str(run/'eager_policy.bin'),FRAGMENT_POLICY_PATH=str(run/'fragment_policy.bin'),
            FRAGMENT_EXPERT_MODE='triton_small',EXPERT_WORKLOAD_PATH=str(run/'active_workload.txt'),
            NVSHMEM_DISABLE_P2P='0',DEEPEP_LOW_LATENCY_NVLINK_ONLY='0',PP_GAP_POLICY_PATH=str(run/'gap_policy.bin'))
        env['BENCH_REPLICAS']=str(sum(s['pp']>1 for s in row['instances']) or 1)
        command=[cfg['python'],str(Path(cfg['guard_scripts'])/'model_cache_adapter.py'),'exec-local',
            '--bindings',env[base.h.GUARD_KEYS[3]],'--',cfg['python'],str(Path(base.__file__).resolve()),
            '--run',str(run),'--engine-entry','--arm',arm,'--replica',str(replica)]
        proc=subprocess.Popen(command,env=env,cwd=cfg['engine_root'],stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        identity=base.h.process_record(proc.pid)
        if not identity or identity['pgid']!=proc.pid:
            proc.terminate();raise RuntimeError('Owned model process group absent')
        thread=threading.Thread(target=base.h.stream_log,args=(proc.stdout,row['usage_logs'][replica]['path'],key),daemon=True)
        thread.start()
        return dict(process=proc,identity=identity,log_thread=thread,arm=arm,replica=replica)
    base.launch=launch

    def progress(run,cfg,stage,**extra):
        measured=len(list(run.glob('results/*/*/completed.json')))
        base.h.save(run/'task/progress.json',dict(experiment_id=cfg['id'],task_id='comparison',stage=stage,
            timestamp=time.time(),controller_pid=json.loads((run/'task/controller_identity.json').read_text())['pid']
            if (run/'task/controller_identity.json').exists() else os.getpid(),valid_points=measured,
            measured_points=measured,expected=cfg['expected_points'],**extra))
    base.progress=progress

    def states(cfg,arm,key):
        return base.each_server(cfg,arm,lambda c:base.h.http_json(c['port'],'/get_server_info',key,timeout=30))

    counter_epoch=[0]
    def counts(run,cfg,arm,key):
        from artifact_visibility import wait_for_stable_json
        states(cfg,arm,key)
        counter_epoch[0]=counter_epoch[0]%254+1;epoch=counter_epoch[0]
        byte(cfg['forward_counter_control'],epoch);rows=[]
        for replica,inst in enumerate(base.layout(cfg,arm)['instances']):
            for pp in range(inst['pp']):
                for tp in range(inst['tp']):
                    path=run/'evidence'/f'{arm}.r{replica}'/'forward_counters'/f'pp{pp}.tp{tp}.json'
                    deadline=time.monotonic()+25
                    while True:
                        value,_=wait_for_stable_json(path,timeout_s=25,poll_s=.03,min_stable_s=.03)
                        if value['epoch']==epoch:break
                        if time.monotonic()>deadline:raise TimeoutError('Forward snapshot epoch absent')
                        time.sleep(.03)
                    rows.append(dict(replica=replica,shape=value['shape'],counts=value['counts']))
        assert len(rows)==4
        return rows

    def set_micro(cfg,arm,key,size):
        for replica,inst in enumerate(base.layout(cfg,arm)['instances']):
            if inst['pp']>1:
                answer=base.h.http_json(base.port(cfg,arm,replica),'/set_internal_state',key,
                    dict(server_args=dict(pp_max_micro_batch_size=size)),timeout=30)
                assert answer and all(x is True for x in answer),answer

    bodies=None
    def requests(cfg,case):
        nonlocal bodies
        if bodies is None:
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(cfg['model_path'],local_files_only=True)
            texts=['Explain distributed inference, token routing and model execution. ',
                   'A botanist compares soil, rainfall and plants across seasons. ',
                   'Describe a Python algorithm to merge sorted database records. ',
                   'A historian studies trade routes, ships and medieval port cities. ']
            bodies=[tok.encode(t,add_special_tokens=False) for t in texts]
        rows=[]
        for i in range(case['requests']):
            if case['kind']=='mixed_arrivals':
                # Rotate P/D order every group so affinity does not coincide
                # with an arbitrary round-robin rank index.
                cls='P' if (i+(i//4))%4<int(case['p_fraction']*4) else 'D'
                q,o=case[cls+'_Q'],case[cls+'_O']
            else:cls='P' if case.get('O',1)==1 else 'D';q,o=case['Q'],case['O']
            body=bodies[i%len(bodies)]
            tokens=([5000+i]+body*(q//len(body)+2))[:q]
            rows.append(dict(index=i,kind=cls,Q=q,O=o,P=0,input_ids=tokens,
                arrival_s=i/case['rps'] if case['rps'] else 0,
                prompt_sha256=hashlib.sha256(str(tokens).encode()).hexdigest()))
        return rows

    class Immediate:
        def wait(self):return None

    def execute(cfg,arm,key,reqs,route='least_outstanding'):
        instances=base.layout(cfg,arm)['instances'];active=[[0]*s['dp'] for s in instances]
        lock=threading.Lock();barrier=threading.Barrier(len(reqs)+1);origin=[None]
        def one(r):
            barrier.wait()
            scheduled=origin[0]+r['arrival_s'];delay=scheduled-time.perf_counter()
            if delay>0:time.sleep(delay)
            with lock:
                if 'forced_replica' in r:
                    replica=r['forced_replica']
                elif route=='phase_pools':
                    half=len(instances)//2
                    eligible=range(half) if r['kind']=='P' else range(half,len(instances))
                    replica=min(eligible,key=lambda j:(sum(active[j]),(j-r['index'])%len(instances)))
                elif route=='least_outstanding':
                    replica=min(range(len(instances)),key=lambda j:(sum(active[j])/len(instances[j]['gpus']),
                        (j-r['index'])%len(instances)))
                else:
                    want_pp=(r['kind']=='P') if route=='p_to_pp' else (r['kind']=='D')
                    replica=next(j for j,s in enumerate(instances) if (s['pp']>1)==want_pp)
                rank=min(range(instances[replica]['dp']),key=lambda j:(active[replica][j],
                    (j-r['index'])%instances[replica]['dp']))
                occupancy=[list(x) for x in active];active[replica][rank]+=1
            p=dict(input_ids=r['input_ids'],sampling_params=dict(temperature=0,max_new_tokens=r['O'],ignore_eos=True),stream=True)
            if instances[replica]['dp']>1:p['data_parallel_rank']=rank
            try:
                value=base.b.stream_one(dict(cfg,port=base.port(cfg,arm,replica)),key,p,Immediate())
                assert value['prompt_tokens']==r['Q'] and value['output_tokens']==r['O']
                assert value['cached_tokens']==0,('unexpected prefix hit',value['cached_tokens'])
                assert len(value['output_ids'])==r['O']
                assert not any((value.get('cached_tokens_details') or {}).get(k,0) for k in ('host','storage'))
                value.update(index=r['index'],kind=r['kind'],replica=replica,dp_rank=rank,
                    planned_arrival_s=r['arrival_s'],client_arrival_lag_ms=(value['started']-scheduled)*1000,
                    scheduled=scheduled,prompt_sha256=r['prompt_sha256'],routing_occupancy=occupancy)
                return value
            finally:
                with lock:active[replica][rank]-=1
        with ThreadPoolExecutor(max_workers=len(reqs)) as pool:
            futures=[pool.submit(one,r) for r in reqs]
            origin[0]=time.perf_counter()+.10;barrier.wait()
            rows=[f.result() for f in futures]
        assert all(x==0 for counts_ in active for x in counts_)
        duration=max(r['finished'] for r in rows)-origin[0]
        return dict(requests=rows,origin=origin[0],makespan_s=duration,
            output_tokens_per_s=sum(r['output_tokens'] for r in rows)/duration,
            input_tokens_per_s=sum(r['prompt_tokens'] for r in rows)/duration,
            request_per_s=len(rows)/duration,route=route)

    def summarize(rounds):
        result={}
        for cls in ('all','P','D'):
            metrics={}
            for name,field,scale in [('mean_ttft_ms','ttft_s',1000),('mean_tpot_ms','tpot_ms',1),
                                      ('mean_e2e_ms','e2e_s',1000),('mean_client_lag_ms','client_arrival_lag_ms',1)]:
                values=[]
                for group in rounds:
                    v=[r[field]*scale for r in group['requests'] if (cls=='all' or r['kind']==cls) and r[field] is not None]
                    if v:values.append(statistics.mean(v))
                if values:metrics[name]=dict(mean=statistics.mean(values),samples=values,
                    stdev=statistics.stdev(values) if len(values)>1 else 0)
            result[cls]=metrics
        for name in ('makespan_s','request_per_s','output_tokens_per_s','input_tokens_per_s'):
            values=[g[name] for g in rounds]
            result[name]=dict(mean=statistics.mean(values),samples=values,stdev=statistics.stdev(values) if len(values)>1 else 0)
        return result

    def qualification(run,cfg,arm,key,health):
        haspp=any(s['pp']>1 for s in base.layout(cfg,arm)['instances'])
        if not haspp:return 0
        if cfg.get('controlled_qualification'):
            pp_replicas=[i for i,s in enumerate(base.layout(cfg,arm)['instances']) if s['pp']>1]
            case=dict(id='controlled_variable_lengths',kind='mixed_arrivals',requests=8*len(pp_replicas),
                p_fraction=.5,rps=0,P_Q=8192,P_O=8,D_Q=256,D_O=64)
            reqs=requests(cfg,case)
            for r in reqs:r['forced_replica']=pp_replicas[r['index']%len(pp_replicas)]
            set_micro(cfg,arm,key,4);byte(cfg['fixed_cohort_size'],8)
            records=[]
            try:
                for epoch,policy in enumerate((0,cfg['pp_candidate_policy']),start=1):
                    health();byte(cfg['gap_policy_path'],ord(str(policy)));base.flush(cfg,arm,key)
                    byte(cfg['fixed_cohort_control'],epoch)
                    value=execute(cfg,arm,key,reqs)
                    info=states(cfg,arm,key)
                    receipts=[s['experiment_fixed_group_receipt'] for replica,server in enumerate(info) if replica in pp_replicas
                              for s in server['internal_states'] if 'experiment_fixed_group_receipt' in s]
                    assert len(receipts)==len(pp_replicas) and all(r['epoch']==epoch for r in receipts),receipts
                    records.append(dict(policy=policy,result=value,admission_receipts=receipts))
            finally:byte(cfg['fixed_cohort_control'],0)
            identical=all(a['output_ids']==b['output_ids'] for a,b in zip(records[0]['result']['requests'],records[1]['result']['requests']))
            selected=cfg['pp_candidate_policy'] if identical else 0
            base.h.save(run/'evidence'/arm/'controlled_qualification.json',dict(performance_sample=False,
                exact_output_agreement=identical,selected_policy=selected,records=records,
                scope='Same initial admission, real 8192/256-token Prefill, unequal 8/64 output lengths, native variable-sized messages and batch shrinkage; policy1 early send only.'))
            byte(cfg['gap_policy_path'],ord(str(selected)))
            return selected
        # Dynamic widths/finish positions, tested with unchanged prompt/output
        # specifications. Every policy transition is after a drained trace.
        case=dict(id='variable_batch_qualification',kind='mixed_arrivals',requests=8,p_fraction=.5,
                  rps=20,P_Q=8192,P_O=8,D_Q=256,D_O=64)
        reqs=requests(cfg,case);records=[]
        for policy in (0,cfg['pp_candidate_policy']):
            health();byte(cfg['gap_policy_path'],ord(str(policy)));base.flush(cfg,arm,key)
            value=execute(cfg,arm,key,reqs)
            states(cfg,arm,key)
            records.append(dict(policy=policy,result=value))
        identical=all(a['output_ids']==b['output_ids'] for a,b in zip(records[0]['result']['requests'],records[1]['result']['requests']))
        selected=cfg['pp_candidate_policy'] if identical else 0
        base.h.save(run/'evidence'/arm/'variable_batch_qualification.json',dict(performance_sample=False,
            exact_output_agreement=identical,selected_policy=selected,records=records,
            limitation='Fixed-schema lookahead policy9 is not used. If variable batching changes output, conservatively retain native policy0; this alone is not a diagnosis of arithmetic error.'))
        byte(cfg['gap_policy_path'],ord(str(selected)))
        return selected

    def workload(run,cfg,arm,key,case,route,policy,health):
        name=case['id']+('__'+route if case['kind']=='mixed_arrivals' and (arm=='mixed' or route!='least_outstanding') else '')
        out=run/'results'/arm/name;out.mkdir(parents=True,exist_ok=True)
        reqs=requests(cfg,case)
        base.h.save(out/'protocol.json',dict(case=case,route=route,pp_policy=policy,cpu_kv_restore=False,
            hicache=False,requests=[{k:v for k,v in r.items() if k!='input_ids'} for r in reqs],
            allocation=base.layout(cfg,arm),workload_scope='complete cold P+D requests; no phase migration'))
        set_micro(cfg,arm,key,max(1,case['requests']//4) if case['kind']=='cold_prefill' else cfg['pp_stream_microbatch'])
        formal=[]
        for repeat in range(-1,cfg['repeats']):
            health();progress(run,cfg,f'{arm}_{name}',repeat=repeat+1,pp_policy=policy)
            base.flush(cfg,arm,key)
            before=counts(run,cfg,arm,key) if repeat>=0 else None
            def early_counts():
                return [s['experiment_gap_state']['early_sent'] for server in states(cfg,arm,key)
                        for s in server['internal_states'] if 'experiment_gap_state' in s]
            early_before=early_counts() if repeat>=0 and cfg.get('controlled_qualification') and policy==1 else []
            group=execute(cfg,arm,key,reqs,route)
            if repeat<0:
                base.h.save(out/'warmup.json',dict(performance_sample=False,result=group));continue
            after=counts(run,cfg,arm,key);delta=[]
            if early_before:
                early_after=early_counts()
                group['pp_early_send_deltas']=[b-a for a,b in zip(early_before,early_after)]
                assert len(early_before)==len(early_after) and all(x>0 for x in group['pp_early_send_deltas'])
            for a,b in zip(before,after):
                assert a['shape']==b['shape'] and a['replica']==b['replica']
                d={k:v-a['counts'].get(k,0) for k,v in b['counts'].items() if v-a['counts'].get(k,0)}
                assert any(k.startswith('EXTEND:') for k in d),d
                if case['kind']=='mixed_arrivals':assert any(k.startswith('DECODE:') for k in d),d
                delta.append(dict(replica=b['replica'],shape=b['shape'],counts=d))
            group['forward_deltas']=delta
            group['client_arrival_lag_max_ms']=max(r['client_arrival_lag_ms'] for r in group['requests'])
            group['arrival_valid']=group['client_arrival_lag_max_ms']<100
            formal.append(group)
            base.h.save(out/f'formal.r{repeat+1}.json',dict(performance_sample=True,result=group))
            base.h.save(out/'metrics.json',dict(summary=summarize(formal),rounds=len(formal)))
        base.h.save(out/'completed.json',dict(passed=True,formal_rounds=cfg['repeats'],
            arrival_valid=all(g['arrival_valid'] for g in formal)))

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
                byte(cfg['gap_policy_path'],ord('0'));byte(cfg['fixed_cohort_control'],0)
                progress(run,cfg,'loading_'+arm)
                for replica in range(row['replicas']):children.append(launch(run,cfg,arm,replica,env,key))
                deadline=time.monotonic()+cfg['startup_timeout_s'];admission=[]
                for replica,child in enumerate(children):
                    inst=instance(cfg,arm,replica)
                    while True:
                        health()
                        if child['process'].poll() is not None:raise RuntimeError(f'{arm}.r{replica} exited during startup')
                        try:info=base.h.http_json(base.port(cfg,arm,replica),'/get_server_info',key,timeout=3);break
                        except Exception:
                            if time.monotonic()>deadline:raise TimeoutError('Full model startup timeout')
                            time.sleep(2)
                    base.h.save(run/'evidence'/f'{arm}.r{replica}'/'server_info.json',base.h.safe_server_info(info))
                    path=run/'evidence'/f'{arm}.r{replica}'/('ep2' if inst['ep']>1 else 'pp2')/'qualification'
                    qs=list(path.glob('*.json'))
                    assert len(qs)==inst['ep']*inst['pp'] and all(json.loads(p.read_text())['passed'] for p in qs)
                    assert all(not s.get('enable_hierarchical_cache',False) for s in info['internal_states'])
                    startup=Path(row['usage_logs'][replica]['path']).read_text()
                    limits=re.findall(r'max_total_num_tokens=\d+.*?max_running_requests=(\d+)',startup)
                    assert limits,(arm,replica,'missing native concurrency evidence')
                    admission.append(dict(replica=replica,kv_tokens=base.capacity_values(info),
                        max_running=min(map(int,limits)),mamba_slots=list(map(int,re.findall(r'Mamba Cache is allocated\. max_mamba_cache_size: (\d+)',startup)))))
                base.h.save(run/'evidence'/arm/'actual_admission.json',admission)
                # One untimed all-rank warmup makes phase counters available.
                base.flush(cfg,arm,key)
                execute(cfg,arm,key,requests(cfg,dict(kind='warmup',requests=8,Q=512,O=16,rps=0)))
                policy=qualification(run,cfg,arm,key,health)
                for case in cfg['workloads']:
                    routes=['least_outstanding','p_to_pp','p_to_ep'] if arm=='mixed' and case['kind']=='mixed_arrivals' else ['least_outstanding']
                    if cfg.get('phase_isolation') and arm!='mixed' and case['kind']=='mixed_arrivals' and row['replicas']>1:
                        routes=(['least_outstanding'] if any(s['pp']>1 for s in row['instances']) else [])+['phase_pools']
                    for route in routes:workload(run,cfg,arm,key,case,route,policy,health)
            finally:
                if children:
                    evidence=base.h.cleanup(children);base.h.save(run/'evidence'/arm/'cleanup.json',evidence)
                    if not evidence['cleanup_verified']:raise RuntimeError('Owned process cleanup failed')
        return 0
    base.worker=worker
