"""Recompute the finite replay pilot from preserved official exports and lifecycle logs."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import shlex
import subprocess

def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default

def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

def describe(values):
    if not values:return dict(n=0,mean=None,p50=None,p95=None)
    v=sorted(values)
    def quantile(q):
        x=(len(v)-1)*q;i=int(x);j=min(i+1,len(v)-1)
        return v[i]+(v[j]-v[i])*(x-i)
    return dict(n=len(v),mean=statistics.mean(v),p50=quantile(.5),p95=quantile(.95))

def overlap(a,b,c,d):return max(0,min(b,d)-max(a,c))

def verify_remote_exports(run):
    cfg=read(run/'config.json')
    code="""import hashlib,json,sys
from pathlib import Path
run=Path(sys.argv[1])
paths=[run/'replay/agentx/profile_export.jsonl',run/'replay/routing.jsonl',*sorted((run/'replay/events').glob('*.jsonl')),*sorted((run/'servers').glob('*/lifecycle.json')),*[p for p in [run/'replay_workload_done.json',run/'replay/manifest.json'] if p.exists()]]
rows=[]
for path in paths:
 data=path.read_bytes()
 if path.suffix=='.jsonl':
  count=sum(1 for line in data.splitlines() if line and isinstance(json.loads(line),dict))
 else:
  json.loads(data);count=1
 rows.append(dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),records=count))
(run/'analysis_export_manifest.json').write_text(json.dumps(dict(files=rows),indent=2)+chr(10))
"""
    python='/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python'
    subprocess.run(['ssh','-T','-o','BatchMode=yes',cfg['client_node'],shlex.join([python,'-c',code,str(run)])],check=True,timeout=30)
    manifest=run/'analysis_export_manifest.json'
    subprocess.run([python,'/home/bxb1/.codex/skills/llm-experiment-guard/scripts/artifact_visibility.py',str(manifest),
        '--receipt',str(run/'analysis_export_manifest.visibility.json')],check=True,stdout=subprocess.DEVNULL,timeout=20)
    for row in read(manifest)['files']:
        data=Path(row['path']).read_bytes()
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise RuntimeError('Remote export is not fully visible or changed: '+row['path'])

def analyze(run):
    run=run.resolve()
    verify_remote_exports(run)
    cfg=read(run/'config.json');done=read(run/'completed.json',{})
    audit=read(run/'analysis/evidence_audit.json',{})
    if audit:
        for name,digest in audit['files'].items():
            if hashlib.sha256((run/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Audited evidence changed: '+name)
    posts=[read(p) for p in (run/'dashboard_guard/checks').glob('*post.json')]
    supervisor_passed=bool(done.get('passed') and len(posts)==1 and posts[0].get('passed'))
    state='complete' if supervisor_passed else ('measured_supervisor_failed' if audit.get('measurements_complete') else 'incomplete')
    frozen=read(run/'dashboard_spec.json')['frozen_inputs']
    core=sorted((r['path'].split('/python/sglang/',1)[1],r['sha256']) for r in frozen if '/python/sglang/' in r['path'])
    core_hash=hashlib.sha256(json.dumps(core).encode()).hexdigest()
    workload_definition={f:hashlib.sha256((run/f).read_bytes()).hexdigest() for f in ['plan.json','data/traces.jsonl','harness/run.py','harness/mixed_replay.py','harness/sglang_adapter.py']}
    records=jsonl(run/'replay/agentx/profile_export.jsonl')
    routes={r['x_request_id']:r for r in jsonl(run/'replay/routing.jsonl')}
    events=[row for file in (run/'replay/events').glob('*.jsonl') for row in jsonl(file)]
    events.sort(key=lambda r:r['unix_ns'])
    starts=[r['unix_ns']/1e9 for r in events if r['event']=='phase_start']
    if not starts:return dict(run=str(run),status='no_formal_phase',failure=read(run/'failure.json'))
    start=starts[0]
    formal=[r for r in records if r['metadata'].get('benchmark_phase')=='profiling']
    family={};lane_task={};offered={};tasks=[]
    for r in events:
        if r['event']=='offer':offered[r['task_id']]=r['unix_ns']/1e9
        if r['event'] in ['existing','admit']:
            task=r['task_id'];lane_task[r['lane']]=task
            family[r['root']]=task
        if r['event']=='tree_drained':
            task=lane_task.get(r['lane'])
            tasks.append(dict(task=task,end=r['unix_ns']/1e9,elapsed_s=r['unix_ns']/1e9-(offered.get(task,start))))
    end=max([t['end'] for t in tasks]+[r['metadata'].get('request_end_ns',0)/1e9 for r in formal]+[start])
    request=[];signatures=[]
    def value(row,key,default=None):return row.get('metrics',{}).get(key,{}).get('value',default)
    for r in formal:
        meta=r['metadata'];route=routes.get(meta['x_request_id'],{})
        ttft=value(r,'time_to_first_token');tpot=value(r,'inter_token_latency')
        osl=value(r,'output_sequence_length',0)
        good=bool(route.get('success')) and not meta.get('was_cancelled') and not meta.get('context_overflow_skip')
        slo=good and ttft is not None and ttft<=1000*cfg['slo']['ttft_s'] and (osl==1 or (tpot is not None and tpot<=1000*cfg['slo']['tpot_s']))
        usage=route.get('prefill_usage') or {}
        detail=usage.get('_sglang_cached_tokens_details') or {}
        task=family.get(meta.get('root_correlation_id'),'unmapped')
        row=dict(task=task,source_outer_idx=meta.get('source_outer_idx'),source_inner_idx=meta.get('source_inner_idx'),
            turn_index=meta.get('turn_index'),source_kind=meta.get('source_kind'),input_tokens=value(r,'input_sequence_length'),
            output_tokens=osl,ttft_ms=ttft,tpot_ms=tpot,request_e2e_ms=value(r,'request_latency'),
            http_e2e_ms=1000*route['elapsed_s'] if route.get('elapsed_s') is not None else None,
            admission_wait_ms=1000*route.get('admission_wait_s',0),success=good,slo=bool(slo),target=route.get('target'),
            prefill_device_cached=detail.get('device'),prefill_host_cached=detail.get('host'),
            prefill_prompt_tokens=usage.get('prompt_tokens'),started=meta.get('request_start_ns',0)/1e9)
        signatures.append([row[k] for k in ['task','source_outer_idx','source_inner_idx','turn_index','source_kind']])
        request.append(row)
    lifecycles=[read(p) for p in (run/'servers').glob('*/lifecycle.json')]
    worker_s=sum(len(r['gpus'])*overlap(start,end,r['engine_spawned'],r['engine_cleaned']) for r in lifecycles)
    worker_total=sum(len(r['gpus'])*(r['engine_cleaned']-r['engine_spawned']) for r in lifecycles)
    actions=read(run/'actions.json',[])
    service_end=done.get('comparison_finished',audit.get('service_end_unix',end))
    worker_service=sum(len(r['gpus'])*overlap(start,service_end,r['engine_spawned'],r['engine_cleaned']) for r in lifecycles)
    logical=4*(end-start)
    for up in actions:
        if up['success'] and up['decision']['direction']=='up':
            downs=[a['finished'] for a in actions if a['success'] and a['decision']['direction']=='down' and a['finished']>up['finished']]
            logical+=up['decision']['gpu_delta']*overlap(start,end,up['finished'],min(downs) if downs else end)
    cache_rows=[r for r in request if r.get('prefill_device_cached') is not None and r.get('prefill_host_cached') is not None]
    pdenom=sum(r.get('prefill_prompt_tokens') or 0 for r in cache_rows)
    cached=sum(r['prefill_device_cached']+r['prefill_host_cached'] for r in cache_rows)
    expected_requests=cfg.get('expected_profile_requests',2*17+6*21)
    expected_tasks=cfg.get('expected_tasks',8)
    result=dict(expected_requests=expected_requests,expected_tasks=expected_tasks,qualification_scope=cfg.get('qualification_scope'),complete_workload=(len(tasks)==expected_tasks and len(request)==expected_requests and all(r['success'] for r in request)),run=str(run),status=state,supervisor_passed=supervisor_passed,production_engine_definition_sha256=core_hash,source=cfg['engine_commit'],mechanism=cfg.get('mechanism','static'),
        start_unix=start,end_unix=end,formal_wall_s=end-start,service_through_scale_down_s=service_end-start,formal_requests=len(request),successful=sum(r['success'] for r in request),
        slo_count=sum(r['slo'] for r in request),slo_rate=sum(r['slo'] for r in request)/len(request) if request else None,
        task_count=len(tasks),task_e2e_s=describe([t['elapsed_s'] for t in tasks]),
        request_e2e_ms=describe([r['request_e2e_ms'] for r in request if r['request_e2e_ms'] is not None]),
        http_e2e_ms=describe([r['http_e2e_ms'] for r in request if r['http_e2e_ms'] is not None]),
        ttft_ms=describe([r['ttft_ms'] for r in request if r['ttft_ms'] is not None]),
        tpot_ms=describe([r['tpot_ms'] for r in request if r['tpot_ms'] is not None]),
        admission_wait_ms=describe([r['admission_wait_ms'] for r in request]),
        prefill_cache_observed_requests=len(cache_rows),prefill_cache_hit_fraction=cached/pdenom if pdenom else None,
        prefill_device_hit_tokens=sum(r['prefill_device_cached'] for r in cache_rows) if cache_rows else None,
        prefill_host_hit_tokens=sum(r['prefill_host_cached'] for r in cache_rows) if cache_rows else None,
        input_tokens=sum(r.get('input_tokens') or 0 for r in request),output_tokens=sum(r['output_tokens'] for r in request),
        routed_requests=dict(Counter(r['target'] for r in request)),
        arrival_lag_ms=[r['lag_s']*1000 for r in events if r['event']=='offer'],
        scaling_exercised=done.get('scaling_exercised',audit.get('scaling_round_trip_observed',False)),
        actions=[dict(direction=a['decision']['direction'],seconds=a['finished']-a['started'],success=a['success'],
            control_seconds=a.get('control_finished',0)-a.get('control_started',0) if 'control_started' in a else None,
            replica_startup_seconds=a.get('replica_ready',0)-a['started'] if 'replica_ready' in a else None) for a in actions],
        gpu_seconds=dict(worker_service_through_scale_down=worker_service,node_reserved_service=8*(service_end-start),worker_formal_including_standby_and_startup=worker_s,worker_total_setup_through_cleanup=worker_total,
            logical_layout_formal=logical,node_reserved_formal=8*(end-start)),
        workload_definition=workload_definition,policy=cfg.get('policy'),per_instance_admission=cfg.get('per_instance_admission'),
        workload_signature=sorted(signatures,key=str),lifecycle=lifecycles,tasks=tasks,
        limitations=[cfg.get('qualification_scope','multi-request candidate'), 'one paired exploratory run','P fixed at PP2','pool capacity fixed across PP layouts',
            'inplace standby worker remains resident','replica launched from CPU bank on demand','per-rank HiCache settings recorded, not assumed equal physical bytes'])
    (run/'analysis').mkdir(exist_ok=True)
    (run/'analysis/requests.json').write_text(json.dumps(request,indent=2)+'\n')
    (run/'analysis/summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[analyze(r) for r in a.runs]
    result=dict(runs=rows)
    if len(rows)==2 and all(r['status'] in ['complete','measured_supervisor_failed'] for r in rows):
        one,two=sorted(rows,key=lambda r:r['mechanism'])
        result['same_workset']=one['workload_signature']==two['workload_signature']
        result['same_workload_definition']=one['workload_definition']==two['workload_definition']
        result['same_policy']=one['policy']==two['policy'] and one['per_instance_admission']==two['per_instance_admission']
        result['same_token_work']=all(one[k]==two[k] for k in ['input_tokens','output_tokens'])
        result['same_production_engine']=one['production_engine_definition_sha256']==two['production_engine_definition_sha256']
        result['fully_qualified_pair']=all(r['supervisor_passed'] for r in rows)
        if all(result[k] for k in ['same_workset','same_workload_definition','same_policy','same_token_work','same_production_engine']) and all(r['scaling_exercised'] and r['complete_workload'] for r in rows):
            result['observed_differences_percent']={k:100*(b-a)/b for k,a,b in [
                ('request_e2e',one['request_e2e_ms']['mean'],two['request_e2e_ms']['mean']),
                ('p95_request_e2e',one['request_e2e_ms']['p95'],two['request_e2e_ms']['p95']),
                ('task_e2e',one['task_e2e_s']['mean'],two['task_e2e_s']['mean']),
                ('worker_gpu_time',one['gpu_seconds']['worker_service_through_scale_down'],two['gpu_seconds']['worker_service_through_scale_down'])]}
    if result.get('fully_qualified_pair') and 'observed_differences_percent' in result:result['improvements_percent']=result['observed_differences_percent']
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'}))
    for r in rows:print(json.dumps({k:v for k,v in r.items() if k not in ['workload_signature','lifecycle','tasks','limitations']}))
