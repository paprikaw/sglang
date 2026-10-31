"""Start prepared E6 controls only after the repair run and its watcher finish.

No GPU work is admitted here: the existing queue invokes Guard for every run,
verifies frozen hashes, and stops on any failed qualification or cleanup gate.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_remaining_matrix import completed_and_clean
from wait_then_qualify_v2 import identity

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predecessor-handoff', type=Path, required=True)
    ap.add_argument('--predecessor-watcher', type=Path, required=True)
    ap.add_argument('--predecessor-run', type=Path, required=True)
    ap.add_argument('--upstream-waiter', type=Path)
    ap.add_argument('--successor', type=Path, action='append', required=True)
    ap.add_argument('--status', type=Path, required=True)
    ap.add_argument('--queue-status', type=Path, required=True)
    args = ap.parse_args()
    assert not args.status.exists() and not args.queue_status.exists()
    state = dict(pid=os.getpid(), identity=identity(os.getpid()),
        successors=[str(p.resolve()) for p in args.successor])

    def save(stage, **extra):
        state.update(stage=stage, updated_at=time.time(), **extra)
        tmp = args.status.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(args.status)

    deadline = time.monotonic()+12*3600
    try:
        while True:
            if args.status.with_suffix('.cancel').exists():
                save('cancelled_before_successor')
                return
            if args.upstream_waiter and args.upstream_waiter.exists():
                upstream = read(args.upstream_waiter)
                if upstream.get('stage') in ('stopped_on_failed_gate', 'cancelled_before_successor'):
                    raise RuntimeError('Upstream controls did not pass: '+upstream['stage'])
            if not args.predecessor_handoff.exists():
                assert time.monotonic() < deadline, 'Predecessor queue creation deadline exceeded'
                save('waiting_for_predecessor_queue_creation')
                time.sleep(10)
                continue
            previous = read(args.predecessor_handoff)
            previous.setdefault('stage', 'initializing_queue')
            if previous['stage'] in ('stopped_on_failed_gate', 'cancelled_before_successor'):
                raise RuntimeError('Predecessor queue did not pass: '+previous['stage'])
            observer = read(args.predecessor_watcher) if args.predecessor_watcher.exists() else {}
            if previous['stage'] == 'matrix_complete' and observer.get('stage') == 'finished':
                break
            assert time.monotonic() < deadline, 'Predecessor queue deadline exceeded'
            save('waiting_for_verified_repair', predecessor_stage=previous['stage'])
            time.sleep(10)
        completed_and_clean(args.predecessor_run.resolve())
        assert str(args.predecessor_run.resolve()) in previous['completed']
        if args.status.with_suffix('.cancel').exists():
            save('cancelled_before_successor')
            return
        command = [sys.executable, str(ROOT/'run_remaining_matrix.py'),
            '--predecessor', str(args.predecessor_run.resolve()), '--status', str(args.queue_status.resolve())]
        for run in args.successor:
            command += ['--successor', str(run.resolve())]
        save('starting_verified_controls')
        with args.queue_status.with_suffix('.log').open('xb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            save('controls_queue_running', child=identity(child.pid))
            code = child.wait()
        assert code == 0, f'Controls queue exited {code}'
        save('controls_complete')
    except BaseException as exc:
        save('stopped_on_failed_gate', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
