"""Real SGLang P/D handoff and a bounded AgentX-derived screening workload."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import threading
import time
import urllib.request

def install(base):
    original_entry=base.engine_entry
    def engine_entry(run,cfg,arm,replica):
        if cfg.get('mooncake_dependency_path'):
            base.sys.path.insert(0,cfg['mooncake_dependency_path'])
        return original_entry(run,cfg,arm,replica)
    base.engine_entry=engine_entry
    original_args=base.engine_args
    def engine_args(cfg,arm,replica):
        args=original_args(cfg,arm,replica)
        index=args.index('--max-total-tokens');del args[index:index+2]
        args+=['--enable-hierarchical-cache','--hicache-size',str(cfg['hicache_gb_per_rank']),
            '--hicache-host-memory-mode','cache','--hicache-write-policy','write_through',
            '--hicache-io-backend','kernel','--hicache-mem-layout','page_first',
            '--enable-metrics','--enable-metrics-for-all-schedulers']
        if arm=='pd':
            args+=['--disaggregation-mode','prefill' if replica==0 else 'decode',
                '--disaggregation-transfer-backend','mooncake',
                '--disaggregation-bootstrap-port',str(cfg['port']+800),
                '--disaggregation-ib-device','mlx5_bond_0']
            if replica==1:args+=['--disaggregation-decode-enable-radix-cache']
        return args
    base.engine_args=engine_args

    def stream(cfg,port,key,payload):
        req=urllib.request.Request(f'http://127.0.0.1:{port}/generate',data=json.dumps(payload).encode(),
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        start=time.perf_counter();first=last=None;final=None
        with urllib.request.urlopen(req,timeout=cfg['request_timeout_s']) as response:
            for line in response:
                if not line.startswith(b'data: '):continue
                if line[6:].strip()==b'[DONE]':break
                value=json.loads(line[6:]);meta=value.get('meta_info',{})
                if 'error' in value:raise RuntimeError(str(value['error']))
                finish=meta.get('finish_reason')
                if isinstance(finish,dict) and finish.get('type') in ('abort','error'):raise RuntimeError(str(finish))
                if meta.get('completion_tokens',0):
                    now=time.perf_counter();first=now if first is None else first;last=now
                final=value
        assert first is not None and final is not None,final
        m=final['meta_info'];n=m['completion_tokens']
        assert n==payload['sampling_params']['max_new_tokens'],(n,payload['sampling_params'])
        return dict(ttft_ms=1000*(first-start),tpot_ms=1000*(last-first)/(n-1) if n>1 else None,
            e2e_ms=1000*(last-start),started=start,finished=last,output_ids=final.get('output_ids'),
            prompt_tokens=m.get('prompt_tokens'),output_tokens=n,cached_tokens=m.get('cached_tokens'),
            cached_tokens_details=m.get('cached_tokens_details'),meta_info=m)

    def one(cfg,arm,key,tokens,output,lane):
        row=base.layout(cfg,arm);ep=row['instances'][0]['ep']
        p=dict(input_ids=tokens,sampling_params=dict(temperature=0,max_new_tokens=output,ignore_eos=True),stream=True)
        if ep>1:p['routed_dp_rank']=(lane//2 if arm=='colocated' else lane)%ep
        if arm=='colocated':return stream(cfg,base.port(cfg,arm,lane%2),key,p)
        p.update(bootstrap_host='127.0.0.1',bootstrap_port=cfg['port']+800,bootstrap_room=random.randrange(1,2**62))
        if ep>1:p['disagg_prefill_dp_rank']=lane%ep
        # Same protocol as the pinned upstream mini_lb: submit both sides with
        # the same bootstrap identity. Prefill returns after transfer; Decode
        # stream contains the complete client-visible generated sequence.
        pre=dict(p,stream=False)
        with ThreadPoolExecutor(max_workers=1) as pool:
            started=time.perf_counter()
            future=pool.submit(base.h.http_json,base.port(cfg,arm,0),'/generate',key,pre,cfg['request_timeout_s'])
            result=stream(cfg,base.port(cfg,arm,1),key,p)
            pre_result=future.result()
        assert isinstance(pre_result,dict) and 'error' not in pre_result,pre_result
        result['prefill_meta']=pre_result.get('meta_info',{})
        overhead=(result['started']-started)*1000
        result['ttft_ms']+=overhead;result['e2e_ms']+=overhead;result['started']=started
        return result

    def tokenize(window,body):
        blocks={}
        result=[]
        for req in window['requests']:
            tokens=[]
            for h in req['hash_ids']:
                if h not in blocks:
                    digest=hashlib.sha256(f"{window['session_id']}:{h}".encode()).digest()
                    marker=[1000+int.from_bytes(digest[i:i+2],'little')%20000 for i in range(0,16,2)]
                    blocks[h]=(marker+body*64)[:64]
                tokens.extend(blocks[h])
            assert len(tokens)==req['in'];result.append(tokens)
        return result

    def episode(cfg,arm,key,windows,prompts,concurrency,health):
        origin=time.perf_counter()
        def session(index):
            w=windows[index];rows=[]
            for turn,(r,tokens) in enumerate(zip(w['requests'],prompts[index])):
                health()
                if turn:time.sleep(w['waits_s'][turn-1])
                value=one(cfg,arm,key,tokens,r['out'],index)
                value.update(session_id=w['session_id'],turn=turn,primer=turn==0,
                             source_index=w['start_index']+turn,input_sha256=hashlib.sha256(str(tokens).encode()).hexdigest())
                rows.append(value)
            return rows
        with ThreadPoolExecutor(max_workers=concurrency) as pool:groups=list(pool.map(session,range(len(windows))))
        rows=[r for g in groups for r in g];measured=[r for r in rows if not r['primer']]
        return dict(requests=rows,wall_s=time.perf_counter()-origin,
            means={k:statistics.mean(r[k] for r in measured) for k in ('ttft_ms','tpot_ms','e2e_ms')},
            mean_cached_tokens=statistics.mean((r.get('prefill_meta',{}).get('cached_tokens') if arm=='pd' else r.get('cached_tokens')) or 0 for r in measured))

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
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(cfg['model_path'],local_files_only=True)
        body=tok.encode('def merge_records(items):\n    return sorted(items)\n',add_special_tokens=False)
        windows=json.loads(Path(cfg['screening_windows']).read_text())['windows']
        prompts=[tokenize(w,body) for w in windows]
        smoke=[([5012]+body*512)[:n] for n in (512,513,2048,2049,4096,4097)]
        reference=None
        if cfg.get('colocated_reference'):
            reference=[r['output_ids'] for r in json.loads(Path(cfg['colocated_reference']).read_text())['responses']]
            assert len(reference)==6 and all(len(x)==32 for x in reference)
        for row in cfg['layouts']:
            arm=row['id'];children=[]
            try:
                Path(cfg['gap_policy_path']).write_bytes(b'0');Path(cfg['fixed_cohort_control']).write_bytes(bytes([0]))
                base.progress(run,cfg,'loading_'+arm)
                for i in range(row['replicas']):children.append(base.launch(run,cfg,arm,i,env,key))
                deadline=time.monotonic()+cfg['startup_timeout_s']
                for i,child in enumerate(children):
                    while True:
                        health()
                        if child['process'].poll() is not None:raise RuntimeError(f'{arm}.r{i} exited during startup')
                        try:info=base.h.http_json(base.port(cfg,arm,i),'/get_server_info',key,timeout=3);break
                        except Exception:
                            if time.monotonic()>deadline:raise TimeoutError('Full model startup timeout')
                            time.sleep(2)
                    base.h.save(run/'evidence'/f'{arm}.r{i}'/'server_info.json',base.h.safe_server_info(info))
                    assert all(s.get('enable_hierarchical_cache') for s in info['internal_states']), 'HiCache not active'
                base.progress(run,cfg,'qualifying_'+arm);base.flush(cfg,arm,key)
                responses=[one(cfg,arm,key,t,32,i) for i,t in enumerate(smoke)]
                ids=[r['output_ids'] for r in responses]
                assert all(v is not None and len(v)==32 for v in ids)
                qualified=reference is None or ids==reference
                base.h.save(run/'evidence'/arm/'output_qualification.json',dict(passed=qualified,
                    exact_serial_token_match=qualified if reference is not None else None,responses=responses))
                if reference is None:reference=ids
                if not qualified:raise RuntimeError('PD serial outputs differ from colocated reference; performance withheld')
                for concurrency in (1,8):
                    out=run/'results'/arm/f'c{concurrency}';out.mkdir(parents=True,exist_ok=True)
                    rounds=[]
                    for repeat in range(cfg['repeats']):
                        base.progress(run,cfg,f'{arm}_c{concurrency}',repeat=repeat+1)
                        health();base.flush(cfg,arm,key)
                        value=episode(cfg,arm,key,windows,prompts,concurrency,health);rounds.append(value)
                        base.h.save(out/f'formal.r{repeat+1}.json',value)
                    base.h.save(out/'metrics.json',dict(rounds=len(rounds),
                        means={k:statistics.mean(v['means'][k] for v in rounds) for k in rounds[0]['means']},
                        mean_wall_s=statistics.mean(v['wall_s'] for v in rounds)))
                    base.h.save(out/'completed.json',dict(passed=True,repeats=len(rounds)))
            finally:
                if children:
                    evidence=base.h.cleanup(children);base.h.save(run/'evidence'/arm/'cleanup.json',evidence)
                    if not evidence['cleanup_verified']:raise RuntimeError('Owned process cleanup failed')
        return 0
    base.worker=worker
