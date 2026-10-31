"""Pure EP locality phases under one Guard lease; no native build or model."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import shutil
import subprocess
import sys
import time
from common import ROOT, save, sha
from guard_authorization import (GUARD_KEYS, authorization_signature, close_remote_workers,
    process_ticks, validate_guard, visible_json)
from node import phase_for


def validate_exit(row, node, phase_id):
    if row.get('node') != node or row.get('phase') != phase_id or row.get('returncode') != 0:
        raise ValueError('Phase cleanup identity or return code mismatch')
    if not all(row.get(k) is True for k in ('worker_exited', 'owned_processes_exited', 'gpu_idle')):
        raise ValueError('Owned workers or GPU state not clean')
    if row.get('gpu_query', {}).get('passed') is not True or row['gpu_query'].get('post_exit_only') is not True:
        raise ValueError('Missing post-exit GPU proof')
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    run = args.run.resolve()
    cfg = json.loads((run / 'config.json').read_text())
    task, receipts = run / 'task', run / 'artifact_visibility/controller'
    receipt = validate_guard(run, cfg, os.environ, receipts, require_parent=True)
    save(task / 'guard_authorization.json', {'lease_id': receipt['lease_id'],
        'experiment_id': cfg['id'], 'nodes': cfg['nodes'], 'scope': 'communication_only'})
    cancelled = {'signal': 0}
    def interrupted(signum, _frame):
        cancelled['signal'] = signum
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    pid, ticks, started = os.getpid(), process_ticks(os.getpid()), time.time()
    completed, launched, endings = [], [], []
    procs, logs, valid, rc = [], [], False, 1

    def progress(stage, phase, note):
        save(task / 'progress.json', {'timestamp': time.time(), 'controller_pid': pid,
            'process_start_ticks': ticks, 'stage': stage, 'phase': phase, 'case': phase,
            'valid_points': len(completed), 'completed_phases': len(completed),
            'elapsed_s': time.time() - started, 'note': note})

    try:
        for phase in cfg['phases']:
            phase = phase_for(cfg, phase['id'])
            if cancelled['signal']:
                raise InterruptedError('Pure communication comparison cancelled')
            out = run / phase['id']
            out.mkdir(exist_ok=False)
            progress('running', phase['id'], '纯通信：初始化不计时，之后验证并采样')
            procs, logs = [], []
            for node in phase['nodes']:
                payload = {'guard_env': {key: os.environ[key] for key in GUARD_KEYS},
                    'lease_id': receipt['lease_id'], 'node': node, 'phase': phase['id'],
                    'config_sha256': sha(run / 'config.json')}
                cmd = ['python3', str(ROOT / 'node.py'), '--run', str(run),
                       '--phase', phase['id'], '--controller-stdin']
                log = (out / f'{node}.log').open('w')
                logs.append(log)
                proc = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'ConnectTimeout=5', node, shlex.join(cmd)],
                    stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                    text=True, start_new_session=True)
                procs.append(proc)
                launched.append((phase['id'], node))
                proc.stdin.write(json.dumps({'payload': payload,
                    'signature': authorization_signature(payload, receipt['token'])}) + '\n')
                proc.stdin.flush()
            deadline = time.monotonic() + 650
            while any(proc.poll() is None for proc in procs):
                if cancelled['signal']:
                    raise InterruptedError('Comparison cancelled')
                if any(proc.poll() not in (None, 0) for proc in procs):
                    raise RuntimeError('Communication worker failed: ' + phase['id'])
                if time.monotonic() >= deadline:
                    raise TimeoutError('Communication phase deadline')
                point_files=list(out.glob('point_*_rank0.json'))
                progress('running', phase['id'], f'纯通信；当前布局已输出 {len(point_files)}/{len(phase["workloads"])} 档结果，等待所有 rank 验证')
                time.sleep(.5)
            close_remote_workers(procs)
            for log in logs:
                log.close()
            if any(proc.returncode != 0 for proc in procs):
                raise RuntimeError('Nonzero phase exit: ' + phase['id'])
            for node in phase['nodes']:
                endings.append(validate_exit(visible_json(out / f'{node}_exit.json', receipts, timeout=30), node, phase['id']))
            for rank in range(phase['world']):
                result = visible_json(out / f'result_rank{rank}.json', receipts, timeout=30)
                if result.get('passed') is not True or result.get('model_loaded') is not False:
                    raise ValueError('Invalid or model-loading communication result')
            completed.append(phase['id'])
        subprocess.run(['/home/bxb1/conda/bin/python', str(ROOT / 'analyze.py'), '--run', str(run)], check=True)
        valid = visible_json(run / 'comparison.json', receipts).get('passed') is True
        rc = 0 if valid else 1
    except BaseException as error:
        save(run / 'controller_failure.json', {'timestamp': time.time(),
            'error_type': type(error).__name__, 'error': str(error), 'completed_phases': completed})
    finally:
        close_remote_workers(procs)
        for log in logs:
            if not log.closed:
                log.close()
        all_clean = bool(launched)
        proof = []
        for phase_id, node in launched:
            try:
                row = visible_json(run / phase_id / f'{node}_exit.json', receipts, timeout=30)
                clean = all(row.get(k) is True for k in ('worker_exited', 'owned_processes_exited', 'gpu_idle'))
                clean = clean and row.get('phase') == phase_id and row.get('node') == node
                proof.append(row)
            except (OSError, ValueError, TimeoutError):
                clean = False
            all_clean = all_clean and clean
        progress('complete' if valid else 'failed', completed[-1] if completed else 'init',
                 '纯通信对比已保存' if valid else '保留失败记录；未接受计时')
        save(task / 'cleanup.json', {'passed': all_clean, 'model_processes_exited': all_clean,
            'gpu_idle': all_clean, 'experiment_id': cfg['id'], 'task_id': 'local',
            'timestamp': time.time(), 'evidence': proof})
    return 0 if valid and all_clean else rc or 1


if __name__ == '__main__':
    raise SystemExit(main())
