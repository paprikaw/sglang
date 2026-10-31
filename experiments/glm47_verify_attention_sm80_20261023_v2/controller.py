#!/usr/bin/env python3
"""Supervise the one authorized SM80 synthetic Attention worker over SSH."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
GUARD = HERE / 'guard_candidate/scripts'
BOOTSTRAP = '''import json,os,sys
packet=json.load(sys.stdin)
os.environ.update(packet["env"])
os.execv(packet["command"][0],packet["command"])
'''
CLEANUP_PROBE = '''import json,os,subprocess,sys,time
from pathlib import Path
run=Path(sys.argv[1])
identity=json.loads((run/'worker_identity.json').read_text()) if (run/'worker_identity.json').exists() else None
receipt=json.loads((run/'worker_exit.json').read_text()) if (run/'worker_exit.json').exists() else None
same_process=False
if identity:
    stat=Path('/proc')/str(identity['pid'])/'stat'
    if stat.exists():
        raw=stat.read_text();same_process=int(raw[raw.rfind(')')+2:].split()[19])==identity['start_ticks']
probe=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
print(json.dumps({'timestamp':time.time(),'node':os.uname().nodename.split('.')[0],
 'identity_recorded':identity is not None,'worker_process_exited':identity is not None and not same_process,
 'owned_children_exited':bool(receipt and receipt.get('owned_children_exited')),
 'gpu_query_returncode':probe.returncode,'gpu_compute_processes':probe.stdout.strip().splitlines(),
 'gpu_idle':probe.returncode==0 and not probe.stdout.strip()}))
'''


def save(path, value):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    tmp.replace(path)


def start_ticks(pid):
    raw = Path('/proc', str(pid), 'stat').read_text()
    return int(raw[raw.rfind(')') + 2:].split()[19])


def authenticate(run, cfg):
    sys.path.insert(0, str(GUARD))
    import model_cache_adapter as adapter
    import gpu_registry
    spec_path = run / 'dashboard_spec.json'
    spec = json.loads(spec_path.read_text())
    adapter.validate_contract(spec, launch=True)
    adapter.validate_command(spec, [sys.executable, str(Path(__file__).resolve()), '--run', str(run)])
    receipt_path = Path(os.environ['EXPERIMENT_GUARD_LEASE_RECEIPT'])
    receipt = gpu_registry.load_receipt(receipt_path)
    registry = gpu_registry.SharedGpuRegistry(receipt['registry_dir'])
    lease = registry.assert_owned(receipt, cfg['id'], hashlib.sha256(spec_path.read_bytes()).hexdigest())
    registry.assert_owned_nodes(receipt, [cfg['node']])
    if (time.time() - lease['heartbeat_at'] > lease['ttl_s']
            or os.environ.get('EXPERIMENT_GUARD_ACQUIRED_NODES') != cfg['node']):
        raise RuntimeError('Exact-node supervising lease is not live')
    return lease


def ssh_command(node, remote_command):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', node,
            shlex.join(remote_command)]


def private_allocation_snapshot(run, cfg, lease):
    """Keep the worker's strict private-file check; do not change the registry.

    The registry's allocation is a public, token-free rendering. Authenticate
    every authority-bearing field against the live lease before copying its
    exact bytes into this run's private directory. The worker repeats the live
    lease check, so neither rendering grants authority by itself.
    """
    raw = Path(os.environ['EXPERIMENT_GUARD_ALLOCATION_FILE']).read_bytes()
    allocation = json.loads(raw)
    if (allocation.get('experiment_id') != cfg['id']
            or allocation.get('lease_id') != lease['lease_id']
            or allocation.get('nodes') != [cfg['node']]
            or allocation.get('tasks') != lease['allocation']
            or allocation.get('tasks') != {cfg['task_id']: [cfg['node']]}):
        raise RuntimeError('Public allocation differs from the authenticated live lease')
    target = run / 'private_allocation_snapshot.json'
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def progress(run, cfg, stage, note=''):
    completed = passed = 0
    for name in cfg['case_ids']:
        path = run / f'{name}.json'
        if path.exists():
            try:
                value = json.loads(path.read_text())
                completed += 1
                passed += value.get('status') == 'passed'
            except (OSError, ValueError):
                pass
    save(run / 'task/progress.json', {
        'timestamp': time.time(), 'controller_pid': os.getpid(),
        'process_start_ticks': start_ticks(os.getpid()), 'stage': stage,
        'phase': 'attention-kernel-qualification', 'valid_points': passed,
        'completed_cases': completed, 'expected_cases': len(cfg['case_ids']),
        'request_completed': completed, 'request_total': len(cfg['case_ids']),
        'progress_percent': 100 * completed / len(cfg['case_ids']), 'note': note})


def stop_owned(proc):
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run
    if not run.is_absolute():
        raise ValueError('An absolute frozen run directory is required')
    cfg = json.loads((run / 'worker_config.json').read_text())
    lease = authenticate(run, cfg)
    for name in ('task',):
        (run / name).mkdir(exist_ok=True)
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    proc = None
    returncode = 1
    error = ''
    progress(run, cfg, 'starting')
    try:
        keys = ('EXPERIMENT_GUARD_LEASE_RECEIPT', 'EXPERIMENT_GUARD_ALLOCATION_FILE',
                'EXPERIMENT_GUARD_ACQUIRED_NODES', 'EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS')
        env = {key: os.environ[key] for key in keys}
        env['EXPERIMENT_GUARD_ALLOCATION_FILE'] = str(private_allocation_snapshot(run, cfg, lease))
        cuda = '/apps/easybuild-2022/easybuild/software/Core/CUDA/12.8.0'
        env.update({
            'CUDA_VISIBLE_DEVICES': cfg['cuda_visible_devices'],
            'CUDA_HOME': cuda, 'CUDA_PATH': cuda,
            'PATH': f'{Path(cfg["python"]).parent}:{cuda}/bin:/usr/local/bin:/usr/bin:/bin',
            'LD_LIBRARY_PATH': f'/home/bxb1/conda/lib:{cuda}/lib64:/apps/easybuild-2022/easybuild/software/Core/OpenSSL/1.1/lib',
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'HF_HUB_DISABLE_TELEMETRY': '1',
            'XDG_CACHE_HOME': '/tmp/bxb1-glm47-verify-sm80-cache',
        })
        packet = {'env': env, 'command': [cfg['python'], str(HERE / 'worker.py'), '--run', str(run)]}
        with (run / 'worker_ssh.log').open('x') as log:
            proc = subprocess.Popen(ssh_command(cfg['node'], ['/usr/bin/python3', '-c', BOOTSTRAP]),
                                    stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, text=True)
            proc.stdin.write(json.dumps(packet))
            proc.stdin.close()
            while proc.poll() is None:
                authenticate(run, cfg)
                progress(run, cfg, 'running')
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        returncode = proc.returncode
        from artifact_visibility import wait_for_stable_json
        summary, receipt = wait_for_stable_json(run / 'worker_summary.json', timeout_s=15)
        save(run / 'worker_summary_visibility.json', receipt)
        if summary['id'] != cfg['id'] or summary['node'] != cfg['node']:
            raise RuntimeError('Remote summary identity differs')
        save(run / 'task/results.json', summary)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        save(run / 'controller_error.json', {'timestamp': time.time(), 'error': error})
        stop_owned(proc)
    finally:
        stop_owned(proc)
        progress(run, cfg, 'complete' if returncode == 0 and not error else 'failed', error)
        probe = None
        for _attempt in range(3):
            try:
                observed = subprocess.run(ssh_command(cfg['node'],
                    ['/usr/bin/python3', '-c', CLEANUP_PROBE, str(run)]),
                    capture_output=True, text=True, timeout=25)
                if observed.returncode == 0:
                    probe = json.loads(observed.stdout)
                    if probe.get('worker_process_exited') and probe.get('owned_children_exited') and probe.get('gpu_idle'):
                        break
            except (ValueError, OSError, subprocess.TimeoutExpired):
                pass
            time.sleep(2)
        save(run / 'cleanup_probe.json', probe or {'error': 'cleanup_not_verified'})
        clean = bool(probe and probe.get('node') == cfg['node'] and probe.get('worker_process_exited')
                     and probe.get('owned_children_exited') and probe.get('gpu_idle'))
        save(run / 'task/cleanup.json', {
            'timestamp': time.time(), 'experiment_id': cfg['id'], 'task_id': cfg['task_id'],
            'passed': clean, 'gpu_idle': clean, 'model_processes_exited': clean,
            'gpu_resources_released': clean, 'node': cfg['node'], 'worker_returncode': returncode})
    return returncode if returncode != 0 else (1 if error else 0)


if __name__ == '__main__':
    raise SystemExit(main())
