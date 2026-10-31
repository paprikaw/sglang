"""Launch prepared controls only after an exact direct run and observer finish."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predecessor', type=Path, required=True)
    ap.add_argument('--predecessor-watcher', type=Path, required=True)
    ap.add_argument('--successor', type=Path, action='append', required=True)
    ap.add_argument('--status', type=Path, required=True)
    ap.add_argument('--queue-status', type=Path, required=True)
    args = ap.parse_args()
    assert not args.status.exists() and not args.queue_status.exists()
    state = dict(pid=os.getpid(), identity=identity(os.getpid()),
                 predecessor=str(args.predecessor.resolve()),
                 successors=[str(p.resolve()) for p in args.successor])

    def save(stage, **extra):
        state.update(stage=stage, updated_at=time.time(), **extra)
        temp = args.status.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(args.status)

    deadline = time.monotonic()+12*3600
    try:
        while True:
            if args.status.with_suffix('.cancel').exists():
                save('cancelled_before_successor')
                return
            observer = json.loads(args.predecessor_watcher.read_text())
            assert observer['run'] == args.predecessor.name
            if observer['stage'] == 'finished_direct_guard':
                assert observer['run_accepted'], 'Predecessor did not pass complete qualification and cleanup'
                break
            assert observer['stage'] != 'watcher_deadline', 'Predecessor observer expired'
            assert time.monotonic() < deadline, 'Predecessor deadline exceeded'
            save('waiting_for_verified_direct_run', predecessor_stage=observer.get('run_stage'),
                 predecessor_episodes=observer.get('completed_episodes', 0))
            time.sleep(10)
        completed_and_clean(args.predecessor.resolve())
        command = [sys.executable, str(ROOT/'run_remaining_matrix.py'),
            '--predecessor', str(args.predecessor.resolve()), '--status', str(args.queue_status.resolve())]
        for run in args.successor:
            command += ['--successor', str(run.resolve())]
        if args.status.with_suffix('.cancel').exists():
            save('cancelled_before_successor')
            return
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
