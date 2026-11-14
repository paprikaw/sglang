"""One bounded, Guard-owned AgentX/BurstGPT reactive comparison arm."""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import copy
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[1]
sys.path.insert(0, str(ENGINE/'research/pp_elastic'))
import pd_elastic_smoke as q
h = q.h
PYTHON, GUARD = q.PYTHON, q.GUARD
PRIOR = q.PRIOR
PARENT = PRIOR.parent.parent

def save(path, value):
    h.save(path, value)

def endpoint(cfg):
    row = dict(name=cfg['server_id'], url=f"http://{cfg['node']}:{cfg['port']}",attention_dp_size=1)
    if cfg['role']=='prefill':
        row.update(bootstrap_host=cfg['node'],bootstrap_port=cfg['bootstrap_port'])
    return row

def prepare(run, mechanism):
    # Reuse the authenticated canonical-cache contracts and qualified r23 profile.
    q.prepare(run)
    root = json.loads((run/'config.json').read_text())
    spec = json.loads((run/'dashboard_spec.json').read_text())
    old = json.loads((PRIOR/'config.json').read_text())
    cfgs = list(root['node_configs'].values())
    servers = {}
    for sid, base, gpus in [('p',cfgs[0],[0,1]),('d0',cfgs[1],[0,1,2] if mechanism=='inplace' else [0,1])]:
        cfg = copy.deepcopy(base)
        pp = len(gpus)
        cfg['server_id']=sid
        cfg['layouts'][0].update(pp=pp,gpu_groups=[gpus])
        cfg['optimized_elastic_pp'] = sid=='d0' and mechanism=='inplace'
        cfg['cuda_graph'] = sid.startswith('d')
        cfg['hicache_size_gib']=32
        cfg['layouts'][0]['usage_logs']=[dict(id=f'{run.name}.{sid}',path=str(run/'cache'/f'{sid}.log'),expected_ranks=[f'PP{i} TP0' for i in range(pp)])]
        cfg['launch_env'].update(CUDA_MODULE_LOADING='EAGER',CUDA_MODULE_DATA_LOADING='EAGER')
        servers[sid]=cfg
    if mechanism=='replica':
        cfg=copy.deepcopy(servers['d0']); cfg.update(server_id='d1',port=29501,dist_port=29502)
        cfg['layouts'][0]['gpu_groups']=[[2,3]]
        cfg['layouts'][0]['usage_logs']=[dict(id=f'{run.name}.d1',path=str(run/'cache/d1.log'),expected_ranks=['PP0 TP0','PP1 TP0'])]
        servers['d1']=cfg
    cache=spec['model_cache']
    cache['usage_logs']={n:[] for n in root['node_configs']}
    for cfg in servers.values():
        cache['usage_logs'][cfg['node']].extend(cfg['layouts'][0]['usage_logs'])
    cache['hicache_reserve_bytes_per_node']={cfgs[0]['node']:160*1024**3,cfgs[1]['node']:400*1024**3}
    for cfg in servers.values():
        cfg['model_cache']=cache
    # Keep node-config authentication bound to all declared logs, not only d0.
    root['node_configs']={n:copy.deepcopy(next(c for c in servers.values() if c['node']==n)) for n in cache['usage_logs']}
    for n,cfg in root['node_configs'].items():
        cfg['layouts']=[row for c in servers.values() if c['node']==n for row in c['layouts']]
    root.update(mechanism=mechanism,servers=servers,agentx_python=old['agentx_python'],
        agentx_tokenizer=old['agentx_tokenizer'],tokenizer_cache=old['tokenizer_cache'],
        model_path=old['model_path'],client_node=cfgs[1]['node'],per_instance_admission=4,
        policy=dict(high_wait_s=2,up_hold_s=2,down_hold_s=30,residence_s=15,max_gpus=8),
        slo=dict(ttft_s=15,tpot_s=.150,joint_target=.90),
        context_length=65536,max_total_tokens=262144,max_running_requests=4,
        baseline_startup='new PP2 process using admitted ready node-local canonical CPU weights',
        max_scale_cycles=1,full_drain=True,wall_timeout_s=1080)
    save(run/'config.json',root)
    plan=json.loads((PRIOR/'plan.json').read_text())
    with Path(plan['source']['burstgpt_path']).open() as f:
        rows=list(x for i,x in enumerate(csv.DictReader(f)) if 109<=i<=114)
    stamps=[float(x['Timestamp']) for x in rows]
    plan['capacity']=10
    plan['source'].update(burstgpt_rows_zero_based=list(range(109,115)),burstgpt_timestamps_s=stamps,
        selection='Six consecutive original arrivals, no time compression; short pilot')
    plan['arrivals']=[dict(id=f'new-{i}',at_s=45+t-stamps[0],trace_id=plan['source']['selected_family']) for i,t in enumerate(stamps)]
    save(run/'plan.json',plan)
    import shutil
    (run/'data').mkdir()
    shutil.copyfile(PRIOR/'data/traces.jsonl',run/'data/traces.jsonl')
    (run/'harness').mkdir()
    for source in (PRIOR/'harness').glob('*.py'):
        shutil.copyfile(source,run/'harness'/source.name)
    shutil.copyfile(HERE/'router.py',run/'harness/reactive_router.py')
    shutil.copyfile(HERE/'stable_placement.py',run/'harness/stable_placement.py')
    p=run/'harness/sglang_adapter.py'
    s=p.read_text()+ '\nfrom reactive_router import install as install_reactive\ninstall_reactive(Adapter)\n'
    s=s.replace('first_content_s = time.monotonic() - started','first_content_s = time.monotonic() - started\n                            self.reactive.streaming.add(rid)')
    s=s.replace('self.active.pop(rid, None)','self.active.pop(rid, None)\n            self.reactive.streaming.discard(rid)')
    p.write_text(s)
    p=run/'harness/run.py';s=p.read_text()
    s=s.replace('web.TCPSite(server, "127.0.0.1", 0)','web.TCPSite(server, "0.0.0.0", 0)')
    s=s.replace('port = site._server.sockets[0].getsockname()[1]', 'port = site._server.sockets[0].getsockname()[1]\n        (args.output.parent / "router_ready.json").write_text(json.dumps(dict(host=socket.gethostname().split(".")[0],port=port,unix=__import__("time").time())))')
    p.write_text(s)
    ep=dict(mode='pd',require_hicache=True,api_key_file=root['admin_key_file'],
        prefill=[endpoint(servers['p'])],decode=[endpoint(c) for n,c in servers.items() if n.startswith('d')],
        request_timeout_s=600,backend_force_close=True,provenance=dict(pilot=True),reactive_admission=4)
    save(run/'endpoints.json',ep)
    registry=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/kv_checkpoints/registry.json')
    registry_value=json.loads(registry.read_text())
    # Existing checkpoints use fixed synthetic prefixes / old engine fingerprints.
    # This AgentX protocol explicitly primes fresh cache-busted warm snapshots.
    save(run/'checkpoint_lookup.json',dict(registry_path=str(registry),registry_sha256=q.describe(registry)['sha256'],
        reused=False,reason='Fresh AgentX first_turn_prefix namespace and changed engine fingerprint; native existing-cohort snapshot warmup retained',
        lookup_kind='registry inventory incompatibility audit',entries=len(registry_value.get('entries',registry_value.get('checkpoints',[])))))
    spec.update(title=f'122B AgentX/BurstGPT reactive {mechanism}',port=29470)
    spec['tasks']=[dict(id='comparison',label=f'Short reactive {mechanism}',root=str(run/'task'),expected=1,
        nodes=list(root['node_configs']),controller_script=str(Path(__file__).resolve()))]
    paths=[Path(x['path']) for x in spec['frozen_inputs'] if Path(x['path'])!=run/'config.json']
    paths += [run/'config.json',run/'plan.json',run/'endpoints.json',run/'data/traces.jsonl',*sorted((run/'harness').glob('*.py')),
        *sorted(HERE.glob('*.py')),ENGINE/'python/sglang/srt/elastic_pp/pd_reactive.py']
    spec['frozen_inputs']=[q.describe(p) for p in dict.fromkeys(paths)]
    save(run/'dashboard_spec.json',spec)
    command=[PYTHON,str(GUARD/'dashboard_guard.py'),'run','--spec',str(run/'dashboard_spec.json'),
        '--resource-choice','auto','--exit-after-run','--',PYTHON,str(Path(__file__).resolve()),'--run',str(run)]
    save(run/'preparation.json',dict(command=command))
    print(json.dumps(dict(run=str(run),mechanism=mechanism,command=command)))

def engine(run, root, sid):
    cfg=root['servers'][sid]
    import sglang.srt.server_args as server_args
    original=server_args.prepare_server_args
    def arguments(argv):
        values={'--context-length':str(root['context_length']),'--max-total-tokens':str(root['max_total_tokens']),
            '--max-running-requests':'4','--max-mamba-cache-size':'48','--hicache-size':'32',
            '--chunked-prefill-size':'4096','--max-prefill-tokens':'8192'}
        flags=[]
        remove={'--enable-elastic-pp','--elastic-pp-disagg-decode','--disable-cuda-graph'} if sid.startswith('d') and not cfg['optimized_elastic_pp'] else set()
        pairs={'--elastic-pp-initial-partition','--elastic-pp-transition-strategy','--elastic-pp-transition-backing'} if remove else set()
        i=0
        while i<len(argv):
            k=argv[i]
            if k in remove:
                i+=1;continue
            if k in pairs:
                i+=2;continue
            if k in values:
                flags.extend([k,values[k]]);i+=2;continue
            flags.append(k);i+=1
        if remove:
            flags.extend(['--cuda-graph-bs-decode','1'])
        save(run/'servers'/sid/'effective_launch.json',dict(argv=flags,source=root['engine_commit']))
        args=original(flags)
        from launch_checks import validate_prefill_alignment
        validate_prefill_alignment(args)
        return args
    server_args.prepare_server_args=arguments
    q.engine_entry(run,cfg)

def install_telemetry():
    from sglang.srt.managers.scheduler import Scheduler
    old=Scheduler.get_internal_state
    def state(scheduler,*args,**kw):
        result=old(scheduler,*args,**kw)
        live=getattr(scheduler,'_elastic_pp_live_requests',None)
        reqs=live() if live else getattr(getattr(scheduler,'running_batch',None),'reqs',[])
        reqs={r.rid:r for r in reqs if not r.finished()}
        pd_pending=0
        if os.environ['ELASTIC_PD_ROLE']=='decode':
            from sglang.srt.elastic_pp.pd_registration import pending_pd_work
            pd_pending=pending_pd_work(scheduler)
        result.internal_state['e2e_runtime']=dict(running=len(reqs),waiting=len(scheduler.waiting_queue),
            pd_pending=pd_pending,output_tokens={r:len(x.output_ids) for r,x in reqs.items()})
        return result
    Scheduler.get_internal_state=state


if os.environ.get("ELASTIC_PD_ROLE"):
    install_telemetry()


def worker(run,root,sid):
    cfg=root['servers'][sid]
    packet=json.loads(sys.stdin.readline());os.environ.update(packet['environment'])
    h.authenticate(run,cfg,os.environ,local=True)
    directory=run/'servers'/sid
    stop=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set())
    threading.Thread(target=lambda:(sys.stdin.read(),stop.set()),daemon=True).start()
    children=[]
    lifecycle=dict(server=sid,worker_started=time.time(),gpus=cfg['layouts'][0]['gpu_groups'][0],node=cfg['node'])
    try:
        cmd=[PYTHON,str(GUARD/'model_cache_adapter.py'),'exec-local','--bindings',os.environ[h.GUARD_KEYS[3]],'--',
            PYTHON,str(Path(__file__).resolve()),'--engine','--server',sid,'--run',str(run)]
        env=q.environment(cfg)
        proc=subprocess.Popen(cmd,env=env,cwd=ENGINE,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,start_new_session=True)
        lifecycle['engine_spawned']=time.time()
        identity=h.process_record(proc.pid)
        if not identity or identity['pgid']!=proc.pid:raise RuntimeError('owned process group missing')
        thread=threading.Thread(target=h.stream_log,args=(proc.stdout,cfg['layouts'][0]['usage_logs'][0]['path'],h.read_key(cfg)),daemon=True)
        thread.start();children.append(dict(process=proc,identity=identity,log_thread=thread))
        save(directory/'identity.json',identity);save(directory/'lifecycle.json',lifecycle)
        deadline=time.monotonic()+1200
        while not stop.wait(.5):
            if proc.poll() is not None:raise RuntimeError('engine exited before readiness')
            try:
                h.http_json(cfg['port'],'/get_server_info',h.read_key(cfg),timeout=2)
                break
            except Exception:
                if time.monotonic()>deadline:raise TimeoutError('engine ready deadline')
        if stop.is_set():raise InterruptedError('stopped before readiness')
        lifecycle['ready']=time.time();save(directory/'lifecycle.json',lifecycle)
        save(directory/'ready.json',dict(passed=True,unix=time.time()))
        while not stop.wait(1):
            h.authenticate(run,cfg,os.environ,local=True)
            if proc.poll() is not None:raise RuntimeError('engine exited during replay')
    except BaseException as error:
        save(directory/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        cleanup=h.cleanup(children)
        lifecycle['engine_cleaned']=time.time();save(directory/'lifecycle.json',lifecycle)
        save(directory/'cleanup.json',cleanup)
        if not cleanup['cleanup_verified']:raise RuntimeError('owned GPU cleanup failed')

def client(run,root):
    packet=json.loads(sys.stdin.readline())
    cfg=root['node_configs'][socket.gethostname().split('.')[0]]
    env=dict(os.environ,**packet['environment'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_CACHE=root['tokenizer_cache'],PYTHONPATH=str(run/'harness'),
        AGENTX_MIXED_CONFIG_PATH=str(run/'plan.json'),AIPERF_DATASET_WEKA_PARALLEL_WORKERS='2')
    h.authenticate(run,cfg,env,local=True)
    stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set())
    threading.Thread(target=lambda:(sys.stdin.read(),stop.set()),daemon=True).start()
    cmd=[root['agentx_python'],str(run/'harness/run.py'),'run','--config',str(run/'endpoints.json'),
        '--model',root['model_path'],'--tokenizer',root['agentx_tokenizer'],'--local-dataset',str(run/'data/traces.jsonl'),
        '--output',str(run/'replay'),'--entries','1','--concurrency','10','--duration','720',
        '--profile-grace','150','--warmup-grace','180','--warmup-prefill-limit','2','--wall-timeout','1050','--smoke']
    children=[]
    try:
        proc=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        identity=h.process_record(proc.pid)
        thread=threading.Thread(target=h.stream_log,args=(proc.stdout,str(run/'replay_driver.log'),''),daemon=True)
        thread.start();children=[dict(process=proc,identity=identity,log_thread=thread)]
        while proc.poll() is None:
            if stop.wait(1):raise InterruptedError('owned replay stopped')
        if proc.returncode:raise RuntimeError('AgentX replay failed')
    finally:
        save(run/'client_cleanup.json',h.cleanup(children))

def coordinator(run,root):
    sys.path.insert(0,str(ENGINE/'python'))
    from sglang.srt.elastic_pp.pd_reactive import ReactivePDPolicy,Parameters,Observation
    import agentx_replay as replay
    key=Path(root['admin_key_file']).read_text().strip()
    workers={};handles=[];errors=[];actions=[];samples=[];expanded=False;generation=0;cycles=0
    stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    client_process=None;completed_at=None;formal_start=None;router=None
    packet=json.dumps(dict(environment={k:os.environ[k] for k in h.GUARD_KEYS}))+'\n'
    pool=cf.ThreadPoolExecutor(max_workers=1);future=None
    policy=ReactivePDPolicy(root['mechanism'],Parameters(**root['policy']))
    def health():
        if stop.is_set():raise InterruptedError('Guard requested owned cleanup')
        for sid,proc in list(workers.items()):
            if proc.poll() is not None:raise RuntimeError(f'worker exited: {sid}')
        for cfg in root['node_configs'].values():h.authenticate(run,cfg,os.environ)
    def progress(stage):
        ident=h.process_record(os.getpid())
        save(run/'task/progress.json',dict(experiment_id=root['id'],task_id='comparison',timestamp=time.time(),
            stage=stage,controller_pid=os.getpid(),controller_start_ticks=ident['start_ticks'],process_start_ticks=ident['start_ticks'],
            valid_points=int(stage == "complete"),expected=1))
    def visible(path,seconds=1200,shutdown=False):
        deadline=time.monotonic()+seconds
        while not path.exists():
            if not shutdown:health()
            if time.monotonic()>deadline:raise TimeoutError(str(path))
            stop.wait(.5)
        subprocess.run([PYTHON,str(GUARD/'artifact_visibility.py'),str(path),'--timeout-s','20',
            '--receipt',str(path.with_name(path.stem+'.visibility.json'))],check=True,stdout=subprocess.DEVNULL,timeout=25)
        return json.loads(path.read_text())
    def launch(sid):
        cfg=root['servers'][sid];directory=run/'servers'/sid;directory.mkdir(parents=True,exist_ok=True)
        cmd=[PYTHON,str(Path(__file__).resolve()),'--worker','--server',sid,'--run',str(run)]
        log=(directory/'worker.log').open('wb');handles.append(log)
        proc=subprocess.Popen(['ssh','-T','-o','BatchMode=yes',cfg['node'],shlex.join(cmd)],
            stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
        proc.stdin.write(packet.encode());proc.stdin.flush();workers[sid]=proc
    def terminate(sid):
        proc=workers.pop(sid)
        proc.stdin.close()
        try:proc.wait(timeout=85)
        except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10)
        return visible(run/'servers'/sid/'cleanup.json',20,shutdown=True)
    def info(sid):
        cfg=root['servers'][sid]
        return replay.http(cfg['node'],cfg['port'],'/get_server_info',key,None,10)
    def rhttp(path,body=None):
        return replay.http(router['host'],router['port'],path,key,body,10)
    def action(decision):
        nonlocal generation
        record=dict(decision=asdict(decision),started=time.time(),mechanism=root['mechanism'])
        actions.append(record);save(run/'actions.json',actions)
        success=False
        try:
            if root['mechanism']=='inplace':
                rhttp('/reactive/control',dict(paused=True))
                deadline=time.monotonic()+180
                while True:
                    health()
                    r=rhttp('/reactive/state');state=info('d0')
                    if r['prefilling']==0 and all(s['e2e_runtime']['pd_pending']==0 for s in state['internal_states']):break
                    if time.monotonic()>deadline:raise TimeoutError('quiescent handoff drain')
                    stop.wait(.25)
                record['handoff_drained']=time.time();record['before']=state
                generation+=1
                cfg=root['servers']['d0']
                record['control_started']=time.time()
                reply=replay.http(cfg['node'],cfg['port'],'/reconfigure_elastic_pp',key,
                    dict(layer_counts=[16,16,16] if decision.direction=='up' else [24,0,24],generation=generation),360)
                record['control_finished']=time.time();record['answer']=reply
                if not reply.get('success'):raise RuntimeError('migration rejected')
                deadline=time.monotonic()+90
                while True:
                    state=info('d0')
                    if all(s.get('transition_metrics',{}).get('phase')=='done' for s in state['internal_states']):break
                    if time.monotonic()>deadline:raise TimeoutError('transition final cleanup')
                    stop.wait(.25)
                record['after']=state
                rhttp('/reactive/control',dict(paused=False))
            elif decision.direction=='up':
                launch('d1');visible(run/'servers/d1/ready.json')
                record['replica_ready']=time.time()
                record['routing']=rhttp('/reactive/control',dict(enabled=['d0','d1']))
            else:
                rhttp('/reactive/control',dict(enabled=['d0']))
                deadline=time.monotonic()+180
                while rhttp('/reactive/state')['active']['d1']:
                    health()
                    if time.monotonic()>deadline:raise TimeoutError('replica drain')
                    stop.wait(.5)
                record['cleanup']=terminate('d1')
            success=True
            return decision.direction=='up'
        finally:
            record.update(success=success,finished=time.time())
            save(run/'actions.json',actions)
    try:
        progress('loading_P2_D2')
        launch('p');launch('d0')
        visible(run/'servers/p/ready.json');visible(run/'servers/d0/ready.json')
        initial=dict(prefill=info('p'),decode=info('d0'))
        if not all('e2e_runtime' in state for state in initial['decode']['internal_states']):
            raise RuntimeError('spawned GPU workers lack queue telemetry')
        save(run/'state.initial.json',initial)
        p,d=[dict(host=root['servers'][n]['node'],port=root['servers'][n]['port'],ep=1,
            bootstrap_port=root['servers'][n]['bootstrap_port']) for n in ['p','d0']]
        tokens=[1000+i%123 for i in range(4608)]
        smoke=[replay.request_one(p,d,key,tokens,8,0,True) for _ in range(2)]
        if smoke[0]['output_ids']!=smoke[1]['output_ids']:raise AssertionError('static P/D warm output mismatch')
        save(run/'start_smoke.json',dict(passed=True,responses=smoke))
        cmd=[PYTHON,str(Path(__file__).resolve()),'--client','--run',str(run)]
        log=(run/'client.log').open('wb');handles.append(log)
        client_process=subprocess.Popen(['ssh','-T','-o','BatchMode=yes',root['client_node'],shlex.join(cmd)],
            stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
        client_process.stdin.write(packet.encode());client_process.stdin.flush()
        router=visible(run/'router_ready.json',60)
        progress('native_warmup_then_finite_burst')
        deadline=time.monotonic()+root['wall_timeout_s']
        last_state=None
        while True:
            health()
            now=time.monotonic()
            if now>deadline:raise TimeoutError('bounded E2E deadline')
            if future and future.done():
                expanded=future.result();future=None
                policy.complete(now=now,success=True)
                if not expanded:cycles+=1
            if formal_start is None:
                for f in (run/'replay/events').glob('*.jsonl'):
                    for line in f.read_text().splitlines():
                        row=json.loads(line)
                        if row['event']=='phase_start':formal_start=row['unix_ns']/1e9
                if formal_start:save(run/'formal_start.json',dict(unix=formal_start))
            route=rhttp('/reactive/state')
            state=info('d0');last_state=state
            runtime=[s['e2e_runtime'] for s in state['internal_states']]
            pending=max(s['pd_pending'] for s in runtime)
            # Router slots include TTFT. Only call it Decode overload once all
            # admitted slots are actually streaming, not waiting on Prefill/RDMA.
            overloaded=(route['waiting']>0 and route['oldest_wait_s']>=2 and route['streaming']>=root['per_instance_admission'])
            internal_overloaded=max(s['waiting'] for s in runtime)>0 and pending==0
            sample=dict(unix=time.time(),monotonic=now,formal=bool(formal_start),route=route,
                decode_runtime=runtime,expanded=expanded,operation_pending=future is not None)
            samples.append(sample)
            with (run/'telemetry.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
            fits=(route['waiting']==0 and sum(route['active'].values())<=1 and pending==0)
            if root['mechanism']=='replica':fits=fits and route['active'].get('d1',0)==0
            observation=Observation('decode',now,expanded=expanded,compute_queue=route['waiting'],
                overloaded=overloaded or internal_overloaded,next_layout_improves=True,
                target_fits_state=True,target_fits_load=fits,in_flight_handoff=bool(pending or route['prefilling']),
                metrics_valid=bool(formal_start) and cycles<root['max_scale_cycles'])
            decision=policy.observe([observation],now=now,occupied_gpus=4+(1 if root['mechanism']=='inplace' else 2)*int(expanded),
                free_by_role=dict(decode=4,prefill=0))
            if decision:
                future=pool.submit(action,decision)
                progress('reactive_'+decision.direction)
            if client_process.poll() is not None:
                if client_process.returncode:raise RuntimeError('AgentX client failed')
                if completed_at is None:completed_at=time.time();save(run/'replay_done.json',dict(unix=completed_at))
                if not expanded and future is None:break
                if time.time()-completed_at>75 and future is None:
                    raise RuntimeError('normal reactive down did not become safe after drain')
            stop.wait(1)
        save(run/'state.final.json',last_state)
        save(run/'completed.json',dict(passed=True,formal_start=formal_start,replay_done=completed_at,
            comparison_finished=time.time(),mechanism=root['mechanism'],actions=len(actions),
            scaling_exercised=any(a['decision']['direction']=='up' and a['success'] for a in actions),
            expected_tasks=8,slo=root['slo']))
        progress('complete')
    except BaseException as error:
        errors.append(repr(error));save(run/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        stop.set()
        if client_process:
            if client_process.stdin and not client_process.stdin.closed:client_process.stdin.close()
            try:client_process.wait(timeout=60)
            except subprocess.TimeoutExpired:client_process.terminate();client_process.wait(timeout=10)
        # Never free the servers while a migration/replica startup thread uses them.
        if future:
            try:future.result(timeout=400)
            except BaseException as e:errors.append('operation cleanup: '+repr(e))
        pool.shutdown(wait=True)
        clean=[]
        for sid in list(workers):
            try:clean.append(terminate(sid))
            except BaseException as e:errors.append('cleanup '+sid+': '+repr(e))
        for handle in handles:handle.close()
        good=bool(clean) and all(c.get('cleanup_verified') for c in clean)
        save(run/'task/cleanup.json',dict(experiment_id=root['id'],task_id='comparison',timestamp=time.time(),
            passed=good,model_processes_exited=good,gpu_idle=good,errors=errors))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--mechanism',choices=['inplace','replica'])
    p.add_argument('--worker',action='store_true');p.add_argument('--engine',action='store_true');p.add_argument('--client',action='store_true')
    p.add_argument('--server',choices=['p','d0','d1']);a=p.parse_args()
    if a.prepare:prepare(a.run,a.mechanism)
    else:
        root=json.loads((a.run/'config.json').read_text())
        if a.engine:engine(a.run,root,a.server)
        elif a.worker:worker(a.run,root,a.server)
        elif a.client:client(a.run,root)
        else:coordinator(a.run,root)
