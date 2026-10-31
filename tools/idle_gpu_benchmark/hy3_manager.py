#!/usr/bin/env python3
"""Reconcile the two existing HY3 node-local banks as a background Guard group."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid
HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parent.parent
sys.path.insert(0,str(HERE.parent/'idle_gpu_service'))
from manager import discover_bookings,registry_blocks,atomic_json,read_json,process_start_ticks,same_process
BASE=WORKSPACE/'experiments/hy3_a100_qualification_20261018/node_local/ep8_layers80_reuse_r1_20261019'


def eligibility(cfg,bookings,blocked,ledger,now):
    entries={row['node']:row for row in ledger.get('entries',[])}
    for node in cfg['nodes']:
        book=bookings.get(node)
        if book is None:return f'{node}: no current RUNNING booking'
        if book.end_at-now<=90:return f'{node}: booking ending'
        if node in blocked:return f'{node}: {blocked[node]}'
        bank=entries.get(node,{})
        if bank.get('state')!='active' or bank.get('source_commit')!=cfg['engine_commit'] or bank.get('cache_scheme')!='node_local_weight' or str(bank.get('expected_job_id'))!=book.job_id:
            return f'{node}: existing HY3 cache identity differs; preserve bank'
        if str(cfg['jobs'][node])!=book.job_id:return f'{node}: booking changed; preserve original cache binding'
        if not bank.get('status',{}).get('entries')or any(row.get('state')!='ready'for row in bank['status']['entries']):return f'{node}: recorded cache not ready'
    return None


def prepare(cfg,root,stop_at):
    ident='idle-hy3-'+str(int(time.time()))+'-'+uuid.uuid4().hex[:10]
    run=root/'runs'/ident
    for path in (run/'task',run/'ep8'):
        path.mkdir(parents=True,mode=0o700)
    source=read_json(BASE/'config.json');source.update(id=ident,run_dir=str(run),benchmark_root=str(root),
        timeout_s=604800,stop_at=stop_at,api_port=19270,dist_port=19271,trace=cfg['trace'])
    source['guard_scripts']=str((WORKSPACE/'experiments/hy3_idle_runtime_20261019/guard_candidate/scripts').resolve())
    source['benchmark_policy']=dict(concurrent_requests_per_dp_replica=8,output_tokens=1,queue_policy='observe_only',max_context_tokens=2048)
    atomic_json(run/'config.json',source)
    spec=read_json(BASE/'dashboard_spec.json');spec.update(id=ident,title='Background HY3 full80 continuous prefill on168/169',
        runtime_dir=str(run/'dashboard_guard'),port=20000+int(uuid.uuid4().hex[:6],16)%10000,
        resource_priority='background',preemptible=True,gpu_lease_ttl_s=15,preemption_stop_grace_s=45)
    spec['workload']=dict(kind='burstgpt_continuous_prefill',model_loaded=True,effective_num_hidden_layers=80,
        full_model=True,output_tokens=1,concurrent_requests_per_attention_dp=8,performance_benchmark=True)
    task=spec['tasks'][0];task.update(root=str(run/'task'),controller_script=str(HERE/'hy3_controller.py'),expected=1,label='HY3 full80 EP8 AttentionTP4 DP2 prefill')
    spec['model_cache']['on_miss']='fail';spec['model_cache']['reuse_only']=True
    for node in cfg['nodes']:
        (run/'ep8'/node).mkdir(mode=0o700)
        (run/'ep8'/node/'server.log').touch(mode=0o600)
        spec['model_cache']['usage_logs'][node][0]['path']=str(run/'ep8'/node/'server.log')
    newfiles=list((WORKSPACE/'experiments/hy3_idle_runtime_20261019').rglob('*.py'))
    newfiles+=[run/'config.json',HERE/'hy3_controller.py',HERE/'hy3_manager.py']
    newfiles.extend(HERE/name for name in ('prefill_trace.py','trace_stream.py','prefill_worker.py','prefill_client.py','worker.py','loadgen.py'))
    newfiles.append(HERE.parent/'idle_gpu_service/node_runner_benchmark.py')
    have={x['path']for x in spec['frozen_inputs']}
    for path in newfiles:
        if str(path)not in have:
            spec['frozen_inputs'].append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()));have.add(str(path))
    atomic_json(run/'dashboard_spec.json',spec)
    return run,source,spec


def main():
    os.umask(0o077);p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);args=p.parse_args();cfg=read_json(args.config)
    root=Path(cfg['runtime_root']);root.mkdir(parents=True,exist_ok=True,mode=0o700)
    for name in ('nodes','cursors','stages','validation'):(root/name).mkdir(exist_ok=True,mode=0o700)
    lock=(root/'manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop={'value':False}
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.update(value=True))
    started=time.time();deadline=started+cfg['duration_s'];child=None;run=None;log=None;retry_at=0;rounds=0
    def publish(state,reason=None):
        atomic_json(root/'state.json',dict(schema_version=1,state=state,reason=reason,updated_at=time.time(),
            started_at=started,deadline=deadline,manager_pid=os.getpid(),manager_start_ticks=process_start_ticks(os.getpid()),
            manager_host=socket.gethostname().split('.')[0],nodes=cfg['nodes'],rounds=rounds,
            run=str(run)if run else None,guard_pid=child.pid if child else None))
    try:
        while True:
            now=time.time();stopping=stop['value']or now>=deadline;reason=None;bookings={}
            try:bookings=discover_bookings(cfg,now)
            except Exception as exc:reason='booking discovery unavailable: '+type(exc).__name__
            if child is not None:
                if child.poll()is None:
                    invalid=reason or any(node not in bookings or bookings[node].job_id!=cfg['jobs'][node]or bookings[node].end_at-now<=90 for node in cfg['nodes'])
                    if stopping or invalid:
                        (run/'STOP').touch(exist_ok=True)
                        if not stop_sent and same_process(child.pid,child_ticks):
                            os.kill(child.pid,signal.SIGTERM);stop_sent=True
                    publish('stopping'if stopping else'running',reason)
                    time.sleep(1 if stopping else 5);continue
                atomic_json(run/'guard_exit.json',dict(timestamp=now,returncode=child.returncode));log.close();child=None;retry_at=now+5
            if stopping:publish('stopped');break
            if reason is None:
                try:
                    blocked=registry_blocks(Path(cfg['gpu_registry_dir']),now)
                    ledger=read_json(Path(cfg['gpu_registry_dir']).parent/'cpu_weight_daemons/registry.json')
                    reason=eligibility(cfg,bookings,blocked,ledger,now)
                except Exception as exc:reason='registry/cache admission unavailable: '+type(exc).__name__
            if reason or now<retry_at:
                publish('waiting',reason or'retry backoff');time.sleep(5);continue
            stop_at=min(deadline,*(bookings[node].end_at-90 for node in cfg['nodes']))
            run,source,spec=prepare(cfg,root,stop_at);rounds+=1
            command=[cfg['python'],str(Path(source['guard_scripts'])/'dashboard_guard.py'),'run','--exit-after-run','--spec',str(run/'dashboard_spec.json'),'--resource-choice','user','--node-limit','2','--requested-gpus','8','--',cfg['python'],str(HERE/'hy3_controller.py'),'--run',str(run)]
            log=(run/'guard.log').open('ab',buffering=0)
            env=dict(os.environ,SGLANG_MEMORY_BANK_ROOT=str(WORKSPACE/'experiments/hy3_a100_qualification_20261018/memory_bank_candidate'))
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
            stop_sent=False;child_ticks=process_start_ticks(child.pid);publish('starting');time.sleep(5)
    finally:
        if child and child.poll()is None:
            (run/'STOP').touch(exist_ok=True)
            if same_process(child.pid,child_ticks):os.kill(child.pid,signal.SIGTERM)
            try:child.wait(timeout=120)
            except subprocess.TimeoutExpired:publish('cleanup_pending','Guard remains owner; cleanup not proven')
        if log and not log.closed:log.close()

if __name__=='__main__':main()
