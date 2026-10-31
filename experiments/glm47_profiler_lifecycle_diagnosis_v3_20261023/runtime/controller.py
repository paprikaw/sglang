"""Guard's direct CPU child; starts no model and mutates no weight/KV bank."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from common import ROOT, MODES, save
from guard_authorization import GUARD_KEYS, close_remote_workers, validate_guard, visible_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    run = parser.parse_args().run.resolve()
    cfg = json.loads((run / 'config.json').read_text())
    validate_guard(run, cfg, os.environ, run / 'visibility/controller', require_parent=True)
    task = run / 'task'
    save(task / 'progress.json', dict(stage='running', timestamp=time.time(), valid_points=0, planned_points=2))
    cancelled = {'signal': 0}
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, lambda number, _: cancelled.update(signal=number))
    procs, failure, row = [], None, None
    try:
        remote = shlex.join(['/usr/bin/python3', str(ROOT / 'runtime/node_runner.py'), '--run', str(run)])
        with (run / 'node_ssh.log').open('x') as log:
            proc = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ControlMaster=no',
                '-o', 'ControlPath=none', '-o', 'ConnectTimeout=5', cfg['nodes'][0], remote],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        procs.append(proc)
        proc.stdin.write(json.dumps(dict(guard_env={key: os.environ[key] for key in GUARD_KEYS})) + '\n')
        proc.stdin.flush()
        deadline = time.monotonic() + cfg['worker_timeout_s'] * len(MODES) + 90
        while proc.poll() is None:
            if cancelled['signal'] or (run / 'STOP').exists():
                raise InterruptedError('Lifecycle diagnosis cancelled')
            if time.monotonic() > deadline:
                raise TimeoutError('Controller lifecycle deadline')
            time.sleep(.2)
        if proc.returncode:
            raise RuntimeError('Owned lifecycle supervisor failed')
        row = visible_json(run / 'node_exit.json', run / 'visibility/controller', 30)
        if row.get('passed') is not True:
            raise RuntimeError('Lifecycle communication/cleanup evidence failed')
        ranks = {}
        for mode in MODES:
            ranks[mode] = []
            for rank in range(2):
                path = run / mode / f'result_rank{rank}.json'
                ranks[mode].append(visible_json(path, run / 'visibility/controller', 30) if path.exists()
                    else dict(rank=rank, passed=False, profile_valid=False, result_missing=True))
        valid_profiles = sum(all(item.get('profile_valid') is True for item in ranks[mode]) for mode in MODES)
        save(task / 'results.json', dict(passed=True, diagnostic_cases_completed=len(row['results']),
            valid_profile_modes=valid_profiles, node_outcomes=row['results'], modes=ranks,
            note='Passed means both diagnostic cases attempted and owned cleanup passed; crashed/missing profiles remain invalid, not measured results'))
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
        save(task / 'failure.json', failure)
    finally:
        close_remote_workers(procs)
        try:
            row = visible_json(run / 'node_exit.json', run / 'visibility/controller', 30)
        except Exception:
            row = None
        clean = bool(row and row.get('cleanup_passed'))
        save(task / 'progress.json', dict(stage='complete' if failure is None else 'failed',
            timestamp=time.time(), valid_points=2 if failure is None else 0, planned_points=2))
        # Guard requires terminal cleanup evidence after the final task update.
        save(task / 'cleanup.json', dict(passed=clean, gpu_idle=clean,
            model_processes_exited=clean, gpu_resources_released=clean,
            experiment_id=cfg['id'], task_id='lifecycle', timestamp=time.time()))
    return 0 if failure is None and clean else 1


if __name__ == '__main__':
    raise SystemExit(main())
