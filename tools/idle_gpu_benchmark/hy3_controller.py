#!/usr/bin/env python3
"""Guard-owned continuous prefill for the existing two-node HY3 weight bank."""
import argparse
import asyncio
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parent.parent
RUNTIME=WORKSPACE/'experiments/hy3_idle_runtime_20261019/runtime'
sys.path.insert(0,str(RUNTIME))
from common import GUARD, ROOT, config, environment, save
from auth import authenticate
from controller import ssh_command, BOOTSTRAP, call
sys.path.insert(0,str(HERE))
from prefill_trace import PrefillTrace
from prefill_worker import run_prefill_batch, batch_summary
import aiohttp


class RankSession:
    """Route native requests to the Attention DP replica on this physical node."""
    def __init__(self, session, rank):
        self.session,self.rank=session,rank
    def post(self,url,**kwargs):
        kwargs['json']=dict(kwargs['json'],data_parallel_rank=self.rank)
        return self.session.post(url,**kwargs)


class LeaseCheck:
    def __init__(self,run,cfg,cancelled):
        self.run,self.cfg,self.cancelled=run,cfg,cancelled
        self.last=0.;self.stopped=False;self.children=[]
        self.spec_sha=hashlib.sha256((run/'dashboard_spec.json').read_bytes()).hexdigest()
        from gpu_registry import SharedGpuRegistry,load_receipt
        self.receipt=load_receipt(os.environ['EXPERIMENT_GUARD_LEASE_RECEIPT'])
        self.registry=SharedGpuRegistry(self.receipt['registry_dir'])
    def check(self):
        if any(child.poll() is not None for child in self.children) or self.cancelled['signal'] or (self.run/'STOP').exists() or time.time()>=self.cfg['stop_at']:
            self.stopped=True
        now=time.monotonic()
        if not self.stopped and now-self.last>=.05:
            self.last=now
            try:
                lease=self.registry.assert_owned(self.receipt,self.cfg['id'],self.spec_sha)
                if sorted(lease['nodes'])!=sorted(self.cfg['nodes']) or lease.get('preemption_requested'):
                    self.stopped=True
            except Exception:
                self.stopped=True
        return self.stopped
    async def __call__(self):
        return self.check()


async def benchmark(run,cfg,stop):
    root=Path(cfg['benchmark_root']);nodes=cfg['nodes'];counts={}
    started=time.time()
    async def one_node(node,rank):
        state=dict(dispatched_requests=0,completed_requests=0,successful_requests=0,input_tokens=0)
        counts[node]=state
        stream=PrefillTrace(**cfg['trace'],cursor_path=root/'cursors'/(node+'.json'),node=node,nodes=nodes)
        sequence=0;rows=[];stage=None;inflight=0;offered=0
        def publish(status='running'):
            elapsed=max(.001,time.time()-started)
            row=dict(timestamp=time.time(),status=status,mode='continuous_prefill',node=node,run=str(run),
                deployment_id=cfg['id'],model='Hy3-BF16',attention_tp=4,attention_dp_rank=rank,moe_ep=8,
                coupled_nodes=nodes,concurrent_requests_per_instance=8,inflight=inflight,cumulative=dict(state),
                input_tokens_per_s=state['input_tokens']/elapsed,completed_rps=state['successful_requests']/elapsed,
                target_rps=None,queue_policy='observe_only',stage_dir=str(stage) if stage else None)
            save(root/'nodes'/(node+'.json'),row)
            total={key:sum(c[key]for c in counts.values())for key in state}
            save(run/'task/progress.json',dict(timestamp=time.time(),stage='running',valid_points=1,
                controller_pid=os.getpid(),case='Hy3 BF16 AttentionTP4 DP2 MoEEP8; two nodes',
                completed_requests=total['completed_requests'],successful_requests=total['successful_requests'],
                dispatched_requests=total['dispatched_requests'],benchmark_requests=total['dispatched_requests'],
                note='Continuous prefill, 8 lanes per Attention DP replica; output1; no queue/RPS gate'))
        try:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=8,limit_per_host=8))as session:
                scoped=RankSession(session,rank)
                endpoint=dict(node=node,url=f"http://{nodes[0]}:{cfg['api_port']}",model_id=cfg['id'])
                while not await stop():
                    batch=stream.peek_batch();requests=batch['requests']
                    if not requests:
                        stream.commit();await asyncio.sleep(0);continue
                    sequence+=1;rows=[];inflight=0;offered=len(requests)
                    stage=root/'stages'/f'{run.name}-{sequence:06d}'/node
                    stage.mkdir(parents=True,exist_ok=False,mode=0o700)
                    save(stage/'workload.json',batch);batch_started=time.time()
                    with (stage/'requests.jsonl').open('a',buffering=1)as out:
                        def on_start(_):
                            nonlocal inflight
                            inflight+=1;state['dispatched_requests']+=1
                        def on_row(row):
                            nonlocal inflight
                            inflight-=1;state['completed_requests']+=1;rows.append(row)
                            if row['status']=='ok':
                                state['successful_requests']+=1;state['input_tokens']+=row['prompt_tokens']
                            out.write(json.dumps(row,allow_nan=False)+'\n')
                        done=asyncio.Event()
                        async def heartbeat():
                            while not done.is_set():
                                publish()
                                try:await asyncio.wait_for(done.wait(),.5)
                                except asyncio.TimeoutError:pass
                        beat=asyncio.create_task(heartbeat())
                        try:await run_prefill_batch(scoped,endpoint,requests,stop,concurrency=8,request_timeout_s=600,on_start=on_start,on_row=on_row)
                        finally:done.set();await beat
                    summary=batch_summary(rows,batch_started,time.time(),offered,concurrency=8)
                    save(stage/'summary.json',summary)
                    fields=['node','sample_id','status','prompt_tokens','completion_tokens','ttft_s','e2e_s','tpot_s','server_queue_s','dispatch_epoch','finished_epoch','error']
                    with (stage/'full_metrics.csv').open('w',newline='')as out:
                        writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
                        writer.writerows({key:r.get(key)for key in fields}for r in rows)
                    if summary['valid']:stream.commit()
                    elif not await stop():
                        for _ in range(50):
                            if await stop():break
                            await asyncio.sleep(.1)
        finally:
            stream.close();publish('paused' if stop.check()else'failed')
    await asyncio.gather(*(one_node(node,rank)for rank,node in enumerate(nodes)))


def main():
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run=p.parse_args().run.resolve()
    cfg=config(run);authenticate(run,cfg)
    keys=('EXPERIMENT_GUARD_LEASE_RECEIPT','EXPERIMENT_GUARD_ALLOCATION_FILE','EXPERIMENT_GUARD_ACQUIRED_NODES','EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS')
    guard_env={key:os.environ[key]for key in keys};cancelled={'signal':0}
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,lambda n,_:cancelled.update(signal=n))
    stop=LeaseCheck(run,cfg,cancelled);children=[];cleanups=[];failure=None;ready=False
    try:
        save(run/'task/progress.json',dict(timestamp=time.time(),stage='loading',valid_points=0,controller_pid=os.getpid(),note='Loading full cached HY3 on168/169'))
        for rank,node in enumerate(cfg['nodes']):
            command=[cfg['python'],str(GUARD/'model_cache_adapter.py'),'exec-local','--bindings',guard_env['EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS'],'--',cfg['python'],str(ROOT/'runtime/model_node.py'),'--run',str(run),'--layout','ep8','--node-rank',str(rank)]
            env=environment(cfg,'ep8',rank,guard_env);log=(run/'ep8'/f'{node}.ssh.log').open('x')
            child=subprocess.Popen(ssh_command(node,['/usr/bin/python3','-c',BOOTSTRAP]),stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,text=True)
            children.append((node,child,log));stop.children.append(child);child.stdin.write(json.dumps(dict(env=env,command=command))+'\n');child.stdin.flush()
        deadline=time.monotonic()+1800
        while not stop.check():
            if any(child.poll()is not None for _,child,_ in children):raise RuntimeError('Remote model owner exited')
            try:
                call(f"http://{cfg['nodes'][0]}:{cfg['api_port']}/health_generate",timeout=5);ready=True;break
            except Exception:pass
            if time.monotonic()>=deadline:raise TimeoutError('HY3 readiness timeout')
            save(run/'task/progress.json',dict(timestamp=time.time(),stage='loading',valid_points=0,controller_pid=os.getpid()))
            time.sleep(1)
        if ready:asyncio.run(benchmark(run,cfg,stop))
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,message=str(exc)if isinstance(exc,(RuntimeError,TimeoutError))else type(exc).__name__)
        save(run/'failure.json',failure)
    finally:
        # Inference has stopped before remote model owners receive shutdown.
        for _,child,_ in children:
            if child.stdin and not child.stdin.closed:
                try:child.stdin.write('stop\n');child.stdin.flush()
                except (BrokenPipeError,OSError):pass
                child.stdin.close()
        from artifact_visibility import wait_for_stable_json
        for node,child,log in children:
            try:child.wait(timeout=65)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
            log.close()
            try:
                row,receipt=wait_for_stable_json(run/'ep8'/node/'exit.json',timeout_s=20)
                save(run/'ep8'/f'{node}.visibility.json',receipt);cleanups.append(row)
            except Exception:cleanups.append(dict(node=node,owned_children_exited=False))
        clean=len(cleanups)==2 and all(r.get('owned_children_exited')and r.get('gpu_cleanup',{}).get('passed')for r in cleanups)
        save(run/'task/progress.json',dict(timestamp=time.time(),stage='complete'if clean else'failed',valid_points=int(ready),note='Background round ended'))
        save(run/'task/cleanup.json',dict(timestamp=time.time(),experiment_id=cfg['id'],task_id='qualification',passed=clean,gpu_idle=clean,model_processes_exited=clean,gpu_resources_released=clean,benchmark_workers_exited=True,nodes=cleanups))
    return 0 if clean else 1

if __name__=='__main__':raise SystemExit(main())
