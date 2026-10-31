"""Read-only observer for a gated queue; never launches or stops GPU work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
TERMINAL = {'matrix_complete', 'stopped_on_failed_gate', 'cancelled_before_successor'}


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix('.observer.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--handoff', type=Path, required=True)
    ap.add_argument('--waiter', type=Path, required=True)
    ap.add_argument('--status', type=Path, required=True)
    ap.add_argument('--prior-watcher', type=Path, required=True)
    ap.add_argument('--queue-exec-session', type=int, required=True)
    ap.add_argument('--engine-commit', required=True)
    ap.add_argument('--analysis-script', type=Path, required=True)
    args = ap.parse_args()
    assert not args.status.exists()
    deadline = time.monotonic()+12*3600
    previous = {}
    aggregate_signature = None
    while time.monotonic() < deadline:
        if not args.handoff.exists():
            waiter = read(args.waiter)
            stopped = waiter['stage'] in {'stopped_on_failed_gate', 'cancelled_before_successor'}
            save(args.status, dict(pid=os.getpid(), updated_at=time.time(),
                stage='finished_without_queue' if stopped else 'waiting_for_queue',
                waiter_stage=waiter['stage']))
            if stopped:
                return
            time.sleep(15)
            continue
        handoff = read(args.handoff)
        handoff.setdefault('stage', 'initializing_queue')
        status = []
        for path in handoff['queue']:
            run = Path(path)
            cfg = read(run/'config.json')
            manifest = read(run/'launch_manifest.json')
            assert manifest['engine_commit'] == args.engine_commit
            count = len(list(run.glob('results/*/*/*/completed.r*.json')))
            progress = read(run/'task/progress.json')
            clean = read(run/'task/cleanup.json') if (run/'task/cleanup.json').exists() else {}
            posts = sorted((run/'dashboard_guard/checks').glob('*-post.json'))
            # The queue appends completed only after Guard has exited cleanly.
            accepted = bool(str(run) in handoff.get('completed', [])
                and count == cfg['expected_points'] and progress['stage'] == 'complete'
                and clean.get('passed') and clean.get('gpu_idle')
                and posts and read(posts[-1])['passed'])
            signature = count, accepted, handoff['stage'] in TERMINAL
            # At full count the queue itself writes the final per-run analyses.
            if count and count < cfg['expected_points'] and signature != previous.get(run.name):
                for script in ('analyze_cluster.py', 'analyze_exposure.py', 'analyze_tradeoffs.py'):
                    subprocess.run([sys.executable, str(ROOT/script), '--run', str(run)], check=True)
            previous[run.name] = signature
            status.append(dict(run=run.name, stage=progress['stage'], episodes=count,
                planned=cfg['expected_points'], accepted=accepted))
        signature = tuple((s['run'], s['stage'], s['episodes'], s['accepted']) for s in status)
        if signature != aggregate_signature:
            subprocess.run([sys.executable, str(args.analysis_script.resolve())], check=True)
            aggregate_signature = signature
        terminal = handoff['stage'] in TERMINAL
        observed = dict(pid=os.getpid(), updated_at=time.time(),
            stage='finished' if terminal else 'watching', queue_stage=handoff['stage'],
            engine_commit=args.engine_commit, runs=status)
        save(args.status, observed)
        prior = read(args.prior_watcher)
        if prior.get('stage') == 'finished_direct_guard':
            current = Path(handoff['current']) if handoff.get('current') else None
            state = read(ROOT/'STATUS.json')
            state.update(updated_at=time.time(), goal_complete=False,
                source_commit=args.engine_commit, matrix_queue_stage=handoff['stage'],
                matrix_queue_status=args.handoff.name,
                matrix_queue_cancel=args.handoff.with_suffix('.cancel').name,
                matrix_queue_exec_session=args.queue_exec_session,
                matrix_analysis_watcher_pid=os.getpid(), analysis_watcher_status=args.status.name,
                direct_guard_exec_session=None, active_run=current.name if current else None,
                current_control_runs=status,
                previous_goal_turn_classification='progress: fixed P/D, capacity and role-split controls; dynamic policy still conditional')
            if current:
                progress = read(current/'task/progress.json')
                cfg = read(current/'config.json')
                state.update(active_stage=progress['stage'], active_completed_episodes=progress.get('valid_points', 0),
                    active_protocol=cfg['protocol'])
            else:
                state['active_stage'] = handoff['stage']
            save(ROOT/'STATUS.json', state)
        if terminal:
            return
        time.sleep(30)
    save(args.status, dict(pid=os.getpid(), stage='watcher_deadline', updated_at=time.time()))


if __name__ == '__main__':
    main()
