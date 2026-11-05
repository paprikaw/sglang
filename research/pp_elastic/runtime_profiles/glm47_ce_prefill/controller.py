"""Guard child: run the two single-node arms sequentially."""
import argparse
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
from common import ARMS, ROOT, SG, PYTHON, save, environment, server_args
from guard_authorization import (GUARD_KEYS, authorization_signature, close_remote_workers,
                                 process_ticks, validate_guard, visible_json)


def validate_cleanup(row, selected, arm, rank=0):
    assert row['node'] == selected['nodes'][rank] and row['node_rank'] == rank and row['arm'] == arm
    assert row['returncode'] == 0, dict(node=row.get('node'),arm=row.get('arm'),returncode=row.get('returncode'),launch_error=row.get('launch_error'),reason=row.get('reason'))
    assert all(row.get(key) is True for key in ('worker_exited', 'owned_processes_exited', 'gpu_idle'))
    assert row.get('gpu_query', {}).get('passed') is True
    return row


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    cfg = json.loads((run/'config.json').read_text())
    arms=tuple(cfg['arms'])
    receipt = validate_guard(run, cfg, os.environ, run/'artifact_visibility/controller', require_parent=True)
    save(run/'guard_authorization.json', dict(lease_id=receipt['lease_id'],
         experiment_id=cfg['id'], nodes=cfg['nodes']))
    cancelled = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: cancelled.set())
    controller_pid, controller_ticks = os.getpid(), process_ticks(os.getpid())
    task = run/'a100_pair/task'
    receipts = run/'artifact_visibility/comparison'
    completed, launched, evidence = [], [], []
    procs = []; client = None
    handles = []
    valid = False
    started = time.time()
    requests_per_arm = sum(w['concurrency'] for w in cfg['workloads'].values()) * (cfg['benchmark']['warmup_trials'] + cfg['benchmark']['formal_trials'])
    planned_requests = requests_per_arm * len(arms)

    def progress(stage, arm, note):
        state = {}
        path = run/arm/'client_progress.json'
        if arm not in completed and path.exists():
            try: state = visible_json(path, receipts, timeout=.3)
            except (OSError, ValueError, TimeoutError): pass
        save(task/'progress.json', dict(timestamp=time.time(), controller_pid=controller_pid,
            process_start_ticks=controller_ticks, stage=stage, case=arm, valid_points=len(completed),
            elapsed_s=time.time()-started, note=note, phase=state.get('phase', 'loading'),
            planned_requests=planned_requests, completed_requests=min(planned_requests,requests_per_arm*len(completed)+state.get('completed_requests',0)),
            inflight_requests=0, failed_requests=0))

    try:
        for arm in arms:
            if cancelled.is_set(): raise InterruptedError('Comparison cancelled')
            selected = cfg['cases'][arm]
            out = run/arm
            out.mkdir(exist_ok=False)
            helper = '/home/bxb1/.codex/skills/spartan-model-serving/scripts/create_deployment_manifest.py'
            command = [sys.executable, helper, '--root', str(run/'deployments'), '--engine', 'sglang',
                '--model', 'GLM-4.7-AWQ', '--model-revision', cfg['model_revision'],
                '--source-repo', str(SG), '--source-tag', cfg['source_tag'],
                '--source-commit', cfg['source_commit'], '--hardware', '8xa100-'+arm, '--tp', str(selected['tp']),
                '--job-id', ','.join(selected['jobs']), '--node', ','.join(selected['nodes']),
                '--port', str(selected['api_port']), '--context', '16384']
            command += ['--launch-arg='+value for value in server_args(arm, 0, run)]
            response = subprocess.check_output(command, text=True)
            manifest = Path(next(line.split('=', 1)[1] for line in response.splitlines()
                                 if line.startswith('manifest=')))
            progress('running', arm, 'CPU缓存加载；启动不计入性能')
            procs=[];handles=[]
            for node_rank,node in enumerate(selected['nodes']):
                payload = dict(guard_env={key: os.environ[key] for key in GUARD_KEYS},
                    model_cache_bindings=os.environ['EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS'],
                    lease_id=receipt['lease_id'], node=node, arm=arm,
                    capacity=selected['ep_capacity'], node_rank=node_rank, controller_pid=controller_pid,
                    controller_start_ticks=controller_ticks, controller_host=socket.gethostname())
                command = ['python3', str(ROOT/'model_node.py'), '--run', str(run),
                           '--arm', arm, '--node-rank', str(node_rank), '--controller-stdin']
                handle = (out/f'node{node_rank}_controller.log').open('w');handles.append(handle)
                proc = subprocess.Popen(['ssh', '-o', 'ConnectTimeout=5', node, shlex.join(command)],
                    stdin=subprocess.PIPE, stdout=handle, stderr=subprocess.STDOUT,
                    text=True, start_new_session=True);procs.append(proc)
                launched.append((arm,node_rank))
                proc.stdin.write(json.dumps(dict(payload=payload,
                    signature=authorization_signature(payload, receipt['token'])))+'\n')
                proc.stdin.flush()
            with (out/'client.log').open('w') as log:
                client = subprocess.Popen([PYTHON, str(ROOT/'client.py'), '--arm', arm, '--run', str(run),
                    '--base-url', selected['base_url']], env=environment(arm, arm, 0, run),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic()+cfg['arm_timeout_s']
                while client.poll() is None:
                    if cancelled.is_set(): raise InterruptedError('Comparison cancelled')
                    if any(proc.poll() is not None for proc in procs): raise RuntimeError('Owned model worker exited early')
                    if time.monotonic() >= deadline: raise TimeoutError('Arm deadline')
                    progress('running', arm, 'Prefill C4：预热1轮，正式1轮')
                    time.sleep(.5)
                assert client.returncode == 0
                result = visible_json(out/'benchmark_result.json', receipts, 60)
                assert result.get('passed') is True and result['formal_trials'] == cfg['benchmark']['formal_trials']
            if selected['transport']=='ce':
                progress('running',arm,'正式请求结束；通过本地Unix socket收集CE CPU计数，不计时')
                from ce_receipts import collect
                collect(run,arm,selected)
            save(out/'STOP', dict(timestamp=time.time(), reason='arm completed'))
            close_remote_workers(procs)
            for handle in handles: handle.close()
            for node_rank in range(2):
                row=validate_cleanup(visible_json(out/f'node{node_rank}_end.json',receipts,60),selected,arm,node_rank)
                evidence.append(row)
            record = json.loads(manifest.read_text())
            record.update(status='benchmark_complete_stopped', experiment_id=cfg['id'], arm=arm,
                scope='temporary benchmark; gateway unchanged', guard_lease_id=receipt['lease_id'],
                pipeline_parallel_size=selected['pp'],expert_parallel_size=selected['ep'],attention_tp_size=4,
                validation=dict(benchmark_completed=True, model_processes_exited=True))
            save(manifest, record)
            completed.append(arm)
            procs = []; client = None
            handles = []
        subprocess.run([sys.executable, str(ROOT/'analyze.py'), '--run', str(run)], check=True)
        valid = True
    except BaseException as error:
        save(task/'failure.json', dict(error_type=type(error).__name__, error=str(error),
             completed_arms=completed, timestamp=time.time()))
    finally:
        for child in (client,):
            if child is not None and child.poll() is None:
                try: os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
        close_remote_workers(procs)
        for handle in handles:
            if not handle.closed:handle.close()
        exited = idle = True
        for arm,node_rank in launched:
            try:
                row = visible_json(run/arm/f'node{node_rank}_end.json', receipts, 60)
                exited &= row.get('worker_exited') is True and row.get('owned_processes_exited') is True
                idle &= row.get('gpu_idle') is True
                if not any(old.get('arm') == arm and old.get('node_rank') == node_rank for old in evidence): evidence.append(row)
            except (OSError, ValueError, TimeoutError):
                exited = idle = False
        progress('complete' if valid else 'failed', completed[-1] if completed else arms[0],
                 'EP8 / PP2 EP4 对照已保存' if valid else '查看失败证据')
        save(task/'cleanup.json', dict(passed=bool(exited and idle),
             model_processes_exited=bool(exited), gpu_idle=bool(idle), experiment_id=cfg['id'],
             task_id='a100_pair', timestamp=time.time(), evidence=evidence))
    return 0 if valid and exited and idle else 1


if __name__ == '__main__':
    raise SystemExit(main())
