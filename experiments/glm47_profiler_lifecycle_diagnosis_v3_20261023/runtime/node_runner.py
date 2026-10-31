"""One exact-node, lease-authenticated supervisor for two fresh Torch modes."""
import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

from common import ROOT, MODES, environment, save
from guard_authorization import validate_guard
from process_cleanup import enable_subreaper, process_record, track_descendants, stop_owned, gpu_cleanup_query


def command(cfg, run, mode):
    if mode not in MODES:
        raise ValueError('Unknown mode')
    return [cfg['python'], '-m', 'torch.distributed.run', '--nnodes=1', '--nproc-per-node=2',
        '--master-addr=127.0.0.1', f'--master-port={cfg["master_port"]}', '--max-restarts=0',
        str(ROOT / 'runtime/worker.py'), '--run', str(run), '--mode', mode]


def can_continue(row, cancelled, stop_requested):
    """A crashed diagnostic may continue only after owned GPU cleanup passes."""
    return (not cancelled and not stop_requested and row.get('owned_processes_exited') is True
            and row.get('gpu_cleanup', {}).get('passed') is True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = json.loads((run / 'config.json').read_text())
    packet = json.loads(sys.stdin.readline(65536))
    receipt = validate_guard(run, cfg, packet['guard_env'], run / 'visibility/node')
    node = os.uname().nodename.split('.')[0]
    if cfg['nodes'] != [node]:
        raise RuntimeError('Remote node differs from exact Guard allocation')
    owner_pid = os.getpid()
    owner = process_record(owner_pid)
    if owner is None or not enable_subreaper():
        raise RuntimeError('Owned process identity/subreaper unavailable')
    cancelled = {'signal': 0}
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, lambda number, _: cancelled.update(signal=number))
    link = select.poll()
    link.register(sys.stdin.fileno(), select.POLLIN | select.POLLHUP | select.POLLERR)
    results = []
    try:
        for mode in MODES:
            output = run / mode
            output.mkdir(parents=True, exist_ok=False)
            child, owned = None, {}
            reason, failure, rc, exited = 'launch_error', None, 1, False
            try:
                with (output / 'worker.log').open('x') as log:
                    child = subprocess.Popen(command(cfg, run, mode), env=environment(cfg, mode, run),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                identity = process_record(child.pid)
                if identity:
                    owned[child.pid] = identity['ticks']
                deadline = time.monotonic() + cfg['worker_timeout_s']
                while child.poll() is None:
                    track_descendants(owner_pid, owner['ticks'], owned)
                    if cancelled['signal'] or (run / 'STOP').exists():
                        raise InterruptedError('Guard/owner cancelled diagnosis')
                    if time.monotonic() > deadline:
                        raise TimeoutError('Bounded lifecycle worker deadline')
                    if link.poll(100):
                        data = os.read(sys.stdin.fileno(), 4096)
                        if not data or data.strip().upper() == b'STOP':
                            raise InterruptedError('Controller ownership channel closed')
                rc, reason = child.returncode, 'completed' if child.returncode == 0 else 'worker_failed'
            except BaseException as error:
                failure = dict(type=type(error).__name__, message=str(error))
            finally:
                exited = stop_owned(child, owned, owner_pid, owner['ticks']) if child else True
                cleanup = gpu_cleanup_query(20)
                row = dict(mode=mode, node=node, lease_id=receipt['lease_id'], returncode=rc,
                    reason=reason, failure=failure, owned_processes_exited=exited,
                    gpu_cleanup=cleanup, passed=rc == 0 and exited and cleanup['passed'])
                save(output / 'node_exit.json', row)
                results.append(row)
            if not can_continue(row, cancelled['signal'], (run / 'STOP').exists()):
                break
    finally:
        clean = bool(results) and all(row['owned_processes_exited'] and row['gpu_cleanup']['passed'] for row in results)
        save(run / 'node_exit.json', dict(node=node, results=results, cleanup_passed=clean,
            passed=len(results) == len(MODES) and clean,
            diagnostic_cases_completed=len(results), valid_worker_modes=sum(r['passed'] for r in results)))
    return 0 if len(results) == len(MODES) and clean else 1


if __name__ == '__main__':
    raise SystemExit(main())
