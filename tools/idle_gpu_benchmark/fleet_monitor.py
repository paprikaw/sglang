#!/usr/bin/env python3
"""Read-only aggregate for independent GLM replicas and the coupled HY3 group."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from prefill_monitor import read,save,query_gpu,fresh
ROOT=Path(__file__).resolve().parents[2]/'experiments/idle_gpu_fleet_20261019'
GROUP=['spartan-gpgpu168','spartan-gpgpu169']
REGISTRY=Path('/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/gpu_registry/registry.json')

def snapshot(now):
    glm=read(ROOT/'prefill_20261019/state.json');hy3=read(ROOT/'hy3_prefill_20261019/state.json');registry=read(REGISTRY)
    rows={key:dict(value,model='GLM-4.7-AWQ',coupled_nodes=[key])for key,value in glm.get('nodes',{}).items()}
    ident=Path(hy3['run']).name if hy3.get('run')else None
    leases=[l for l in registry.get('leases',[])if l.get('state')=='active'and l.get('experiment_id')==ident and sorted(l.get('nodes',[]))==GROUP]
    authorized=(len(leases)==1 and leases[0].get('preemptible')is True and leases[0].get('resource_priority')=='background'and not leases[0].get('preemption_requested')and fresh(leases[0].get('heartbeat_at'),now,float(leases[0].get('ttl_s',0))))
    for node in GROUP:
        path=ROOT/'hy3_prefill_20261019/nodes'/(node+'.json');worker=read(path)if path.exists()else{}
        if worker.get('deployment_id')!=ident:worker={}
        current=authorized and worker.get('deployment_id')==ident and fresh(worker.get('timestamp'),time.time(),10)
        row=dict(worker,authorized=bool(current),worker_status=worker.get('status','loading'if authorized else'waiting'),worker_stale=not current,model='Hy3-BF16',coupled_nodes=GROUP)
        # No model traffic or registry changes. Sample only the exact current group.
        if current:
            gpu=query_gpu(node);row.update(gpus=gpu['gpus'],gpu_error=gpu['error'],gpu_sampled_at=gpu['sampled_at'])
        else:row.update(gpus=None,gpu_error='group_not_ready_or_preempted')
        rows[node]=row
    values=[gpu['utilization_gpu_pct']for row in rows.values()if row.get('authorized')and not row.get('worker_stale')for gpu in row.get('gpus')or[]if gpu.get('utilization_gpu_pct')is not None]
    return dict(schema_version=1,timestamp=time.time(),policy='all four currently booked nodes default to preemptible continuous prefill',nodes=rows,glm_manager=glm.get('state'),hy3_manager=hy3['state'],hy3_reason=hy3.get('reason'),valid_gpu_utilization_readings=len(values),gpu_utilization_avg_pct=sum(values)/len(values)if values else None)

def main():
    os.umask(0o077);lock=(ROOT/'fleet_monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            state=snapshot(time.time());save(ROOT/'all_nodes_state.json',state)
            with (ROOT/'all_nodes_history.jsonl').open('a')as out:out.write(json.dumps(state,allow_nan=False)+'\n')
        except Exception as exc:save(ROOT/'fleet_monitor_error.json',dict(timestamp=time.time(),error=type(exc).__name__))
        time.sleep(10)
if __name__=='__main__':main()
