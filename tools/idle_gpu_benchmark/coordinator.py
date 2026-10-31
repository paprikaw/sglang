#!/usr/bin/env python3
"""CPU-only phase planner; all inference is sent by exact-node Guard children."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'idle_gpu_service'))
from node_runner_benchmark import read, save
from loadgen import percentiles
from trace_stream import TraceStream


class CohortChanged(RuntimeError):
    pass


def capacity_failure(result):
    reasons=result.get('reasons',[])
    queue=('backlog','queue_growth','queue_snapshot','retraction','queue_above','sustained_waiting')
    technical=('telemetry_failed','missing_','stale_','stop_check','dispatch_lag','token_count_mismatch','http_','counter reset')
    return any(any(k in r for k in queue) for r in reasons) and not any(any(k in r for k in technical) for r in reasons)


def members(config):
    state = read(Path(config['fleet_runtime']) / 'state.json')
    registry = read(Path(config['gpu_registry_dir']) / 'registry.json')
    leases = {row['lease_id']: row for row in registry['leases']}
    result = {}
    for node in config['nodes']:
        attempt = state.get('attempts', {}).get(node, {})
        status = attempt.get('status', {})
        if not status.get('ready') or time.time() - status.get('timestamp', 0) > 20:
            continue
        run = Path(attempt['run'])
        try:
            spec = read(run / 'dashboard_spec.json')
            receipt = read(Path(spec['runtime_dir']) / 'gpu_registry_lease.json')
            deployment = read(run / 'deployment.json')
            worker = read(run / 'benchmark_worker.json')
            lease = leases.get(receipt['lease_id'], {})
            if (lease.get('state') != 'active' or lease.get('nodes') != [node]
                    or lease.get('resource_priority') != 'background' or lease.get('preemption_requested')
                    or time.time() - lease.get('heartbeat_at', 0) >= lease.get('ttl_s', 0)
                    or deployment.get('status') != 'healthy' or worker.get('state') != 'running'):
                continue
            result[node] = dict(run=str(run), lease_id=lease['lease_id'],
                                deployment_id=deployment['deployment_id'])
        except (OSError, ValueError, KeyError):
            continue
    return result


def schedule(rows, total_rps, mode):
    if total_rps <= 0 or len(rows) < 2:
        raise ValueError('A positive RPS and at least two trace records are required')
    span = float(rows[-1]['offset_s']) - float(rows[0]['offset_s'])
    if mode == 'burstgpt' and span <= 0:
        # Simultaneous arrivals cannot be rate-scaled without changing the trace.
        raise ValueError('All trace timestamps tied; no finite time compression defines an RPS')
    result = []
    for index, row in enumerate(rows):
        relative = index if mode == 'smooth' else (float(row['offset_s']) - float(rows[0]['offset_s'])) / span * (len(rows)-1)
        result.append(dict(row, offset_s=relative / total_rps))
    return result, len(rows)/total_rps


def aggregate(root, plan, summaries):
    records = []
    for node in plan['members']:
        path = root / 'stages' / plan['id'] / node / 'requests.jsonl'
        if path.exists():
            records.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    csv_path=root/'stages'/plan['id']/'full_metrics.csv'
    with csv_path.open('w',newline='') as handle:
        fields=['phase_id','node','sample_id','status','planned_prompt_tokens','prompt_tokens','planned_completion_tokens','completion_tokens','ttft_ms','tpot_ms','e2e_ms','server_queue_ms','dispatch_lag_s','error']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in records:
            out={key:row.get(key) for key in fields};out['phase_id']=plan['id']
            for key in ('ttft','tpot','e2e','server_queue'):
                value=row.get(key+'_s');out[key+'_ms']=value*1000 if value is not None else None
            writer.writerow(out)
    successful = [row for row in records if row['status'] == 'ok']
    reasons = [f'{node}:{reason}' for node, summary in summaries.items() for reason in summary.get('reasons', [])]
    counts = {node: sum(row['node'] == node for row in records) for node in plan['members']}
    planned = {node: sum(index % len(plan['members']) == list(sorted(plan['members'])).index(node)
                        for index in range(len(plan['requests']))) for node in plan['members']}
    if any(counts[n] != planned[n] for n in counts):
        reasons.append('incomplete_round_robin_dispatch')
    end = max((summary.get('finished_epoch', plan['start_at']) for summary in summaries.values()), default=plan['start_at'])
    return dict(id=plan['id'], phase=plan['label'], total_rps=plan['total_rps'],
        per_instance_rps=plan['total_rps']/len(plan['members']), arrival_mode=plan['arrival_mode'],
        workload=plan['workload'], valid=not reasons and all(s['valid'] for s in summaries.values()),
        reasons=reasons, offered_requests=len(plan['requests']), successful_requests=len(successful),
        per_node_counts=counts, planned_per_node_counts=planned, completed_rps=len(successful)/max(.001,end-plan['start_at']),
        latency_s={name:percentiles([r[name+'_s'] for r in successful if r.get(name+'_s') is not None])
                   for name in ('ttft','tpot','e2e','server_queue')},
        token_totals={node:dict(prompt=sum(r.get('prompt_tokens') or 0 for r in records if r['node']==node),
            completion=sum(r.get('completion_tokens') or 0 for r in records if r['node']==node)) for node in counts},
        members=plan['members'], request_records=str(root / 'stages' / plan['id']),
        start_at=plan['start_at'], finished_at=end,
        queue_policy='no sustained backlog in observed test window; admission wait recorded separately')


class Coordinator:
    def __init__(self, config):
        self.config, self.root = config, Path(config['runtime_root'])
        self.stopped = False
        self.deadline = time.time()+config.get('duration_s',604800)
        self.sequence = 0
        self.state = dict(schema_version=1, state='waiting_for_guard_workers', pid=os.getpid(),
                          started_at=time.time(), history=[], highest_validated_rps=None)
    def publish(self, **updates):
        self.state.update(updates, updated_at=time.time())
        save(self.root/'state.json', self.state)
    def deactivate(self):
        try:
            plan = read(self.root/'phase.json')
            plan['active'] = False
            save(self.root/'phase.json', plan)
        except (OSError, ValueError):
            pass
    def wait(self, seconds):
        until = time.monotonic()+seconds
        if time.time()>=self.deadline:self.stopped=True
        while not self.stopped and time.monotonic()<until:
            time.sleep(min(.2,max(0,until-time.monotonic())))
            if time.time()>=self.deadline:self.stopped=True
    def settle(self, cohort):
        # Control-plane reads only; inference remains confined to Guard workers.
        deadline=time.time()+self.config.get('settle_timeout_s',120)
        while not self.stopped and time.time()<deadline:
            if members(self.config)!=cohort:raise CohortChanged()
            idle=True
            for node,member in cohort.items():
                endpoint=read(Path(member['run'])/'deployment.json')['endpoint']
                try:
                    with urlopen(endpoint+'/v1/loads',timeout=2) as response:loads=json.load(response)['loads']
                    if not loads or any(row.get('num_running_reqs',0) or row.get('num_waiting_reqs',0) for row in loads):idle=False
                except (OSError,ValueError,KeyError):idle=False
            if idle:return
            self.wait(.5)
        raise RuntimeError('Previous traffic did not drain; no next phase admitted')

    def stage(self, cohort, rows, rps, label, workload, mode='burstgpt'):
        if self.stopped or members(self.config) != cohort:
            raise CohortChanged()
        self.settle(cohort)
        self.sequence += 1
        stage_id = f'{int(time.time())}-{self.sequence:05d}-{uuid.uuid4().hex[:8]}'
        planned, window = schedule(rows,rps,mode)
        now = time.time()
        max_output = max(row['output_tokens'] for row in rows)
        drain = max(self.config.get('drain_timeout_s',600), max_output*1.5)
        plan = dict(schema_version=1,id=stage_id,active=True,label=label,members=cohort,
            requests=planned,total_rps=rps,dispatch_window_s=window,start_at=now+5,
            deadline=now+5+window+drain,workload=workload,arrival_mode=mode)
        save(self.root/'phase.json',plan)
        self.publish(state='running',phase=label,phase_id=stage_id,target_total_rps=rps,
                     per_instance_rps=rps/len(cohort),members=cohort)
        results = {}
        try:
            while not self.stopped and time.time()<plan['deadline']:
                if members(self.config)!=cohort:
                    raise CohortChanged()
                for node in cohort:
                    path=self.root/'stages'/stage_id/node/'summary.json'
                    if node not in results and path.exists():
                        results[node]=read(path)
                    error=self.root/'stages'/stage_id/node/'worker_failure.json'
                    if error.exists():
                        raise RuntimeError('Worker preparation failed: '+str(error))
                if len(results)==len(cohort):
                    break
                self.wait(.2)
            if self.stopped:
                raise CohortChanged()
            if len(results)!=len(cohort):
                raise RuntimeError('Stage result deadline expired')
            summary=aggregate(self.root,plan,results)
            save(self.root/'stages'/stage_id/'summary.json',summary)
            self.state['history'].append(summary)
            self.publish(last_result=summary)
            return summary
        finally:
            self.deactivate()
    def tune(self,cohort,workload,label,start):
        rows=workload['requests'][:self.config.get('search_requests',32)]
        low,high,rate=0.,None,start
        for iteration in range(self.config.get('max_search_stages',12)):
            result=self.stage(cohort,rows,rate,label,workload['path'])
            if result['valid']:
                low=rate
            else:
                # Only observed server queue overload provides an upper bound.
                if not capacity_failure(result):
                    raise RuntimeError('Measurement invalid; not a server-capacity bound: '+str(result['reasons']))
                high=rate
            self.publish(search_lower_rps=low,search_upper_rps=high)
            if high is not None and low and (high-low)/low<=self.config.get('relative_precision',.1):
                break
            rate=(low+high)/2 if high is not None and low else (rate/2 if high else rate*1.8)
            if rate<self.config.get('min_rps',.005) or rate>self.config.get('max_rps',64):
                break
            self.wait(self.config.get('inter_stage_pause_s',3))
        if not low:
            raise RuntimeError('No safe observed RPS yet; preserve diagnostics and retry')
        # Longer independent confirmation may lower the observed safe point.
        for attempt in range(8):
            result=self.stage(cohort,workload['requests'],low,label+'-confirmation',workload['path'])
            if result['valid']:
                self.publish(highest_validated_rps=low,upper_observed_rps=high,
                    validated_cohort=cohort,validation_result=result)
                return low
            if not capacity_failure(result):
                raise RuntimeError('Confirmation invalid; capacity not established: '+str(result['reasons']))
            high=low
            low*=.8
            self.wait(3)
        raise RuntimeError('Longer confirmation did not find a safe RPS')
    def run(self):
        calibration=read(self.config['calibration_workload']);calibration['path']=self.config['calibration_workload']
        full=read(self.config['full_workload']);full['path']=self.config['full_workload']
        stream_cfg=self.config.get('trace_stream')
        stream=TraceStream(**stream_cfg) if stream_cfg else None
        while not self.stopped:
            try:
                cohort=members(self.config)
                if len(cohort)<self.config.get('min_nodes',1):
                    self.publish(state='waiting_for_guard_workers',members=cohort)
                    self.wait(2);continue
                rate=self.tune(cohort,calibration,'calibration',self.config.get('initial_total_rps',.5))
                self.publish(state='full_length_validation',subset_validated_rps=rate)
                # Full-length trace has its own rate bound; do not reuse subset RPS as a claim.
                rate=self.tune(cohort,full,'full-length',rate)
                self.publish(state='steady',full_length_validated_rps=rate)
                cycle=0
                while not self.stopped and members(self.config)==cohort:
                    cycle+=1
                    batch=stream.peek_batch() if stream else full
                    result=self.stage(cohort,batch['requests'],rate,'full-trace-steady',batch.get('dataset',{}).get('path',full['path']))
                    if not result['valid']:
                        if not capacity_failure(result):
                            raise RuntimeError('Steady measurement invalid: '+str(result['reasons']))
                        rate*=.8
                        self.publish(adaptive_backoff_rps=rate)
                    elif stream:
                        stream.commit()
                        self.publish(trace_cursor=str(stream_cfg['cursor_path']))
                    if cycle%self.config.get('retune_every_cycles',12)==0:
                        break
                    self.wait(3)
            except CohortChanged:
                self.deactivate()
                self.publish(state='paused_for_resource_change',highest_validated_rps=None)
                self.wait(2)
            except (OSError,ValueError,RuntimeError) as exc:
                self.deactivate()
                self.publish(state='diagnostic_pause',error=str(exc))
                self.wait(self.config.get('error_retry_s',30))
        if stream:stream.close()
        self.deactivate();self.publish(state='stopped')


def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=read(args.config);root=Path(config['runtime_root']);root.mkdir(parents=True,exist_ok=True,mode=0o700)
    lock=(root/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    coordinator=Coordinator(config)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,lambda *_:setattr(coordinator,'stopped',True))
    try:coordinator.run()
    finally:coordinator.deactivate()

if __name__=='__main__':main()
