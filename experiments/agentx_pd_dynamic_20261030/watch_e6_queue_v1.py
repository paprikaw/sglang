"""Observe the E6 queue without mixing its measurements into the E5 matrix."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
COMMIT = 'fc014e3553c4d234f38f3088a3158c0333e64830'
TERMINAL = {'matrix_complete', 'stopped_on_failed_gate', 'cancelled_before_successor'}


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--handoff', type=Path, required=True)
    ap.add_argument('--status', type=Path, required=True)
    ap.add_argument('--prior-watcher', type=Path, required=True)
    ap.add_argument('--queue-exec-session', type=int, required=True)
    args = ap.parse_args()
    assert not args.status.exists()
    deadline = time.monotonic() + 12 * 3600
    previous = {}
    aggregate_signature = None
    while time.monotonic() < deadline:
        handoff = read(args.handoff)
        # The queue creates its identity/plan file before its first stage save.
        # A deferred observer can legitimately arrive in that short window.
        handoff.setdefault('stage', 'initializing_queue')
        runs = [Path(p) for p in handoff['queue']]
        status = []
        for run in runs:
            cfg = read(run/'config.json')
            manifest = read(run/'launch_manifest.json')
            assert manifest['engine_commit'] == COMMIT
            markers = sorted(run.glob('results/*/*/*/completed.r*.json'))
            count = len(markers)
            progress = read(run/'task/progress.json') if (run/'task/progress.json').exists() else {}
            cleanup = read(run/'task/cleanup.json') if (run/'task/cleanup.json').exists() else {}
            posts = sorted((run/'dashboard_guard/checks').glob('*-post.json'))
            accepted = bool(count == cfg['expected_points'] and progress.get('stage') == 'complete'
                and cleanup.get('passed') and cleanup.get('gpu_idle')
                and posts and read(posts[-1])['passed'])
            signature = count, accepted, handoff['stage'] in TERMINAL
            # At full completion the queue runs these analyses itself before
            # announcing its terminal state. Avoid racing its output writers.
            if signature != previous.get(run.name) and count and (
                    count < cfg['expected_points'] or handoff['stage'] in TERMINAL):
                for script in ('analyze_cluster.py', 'analyze_exposure.py', 'analyze_tradeoffs.py'):
                    subprocess.run([sys.executable, str(ROOT/script), '--run', str(run)], check=True)
                previous[run.name] = signature
            status.append(dict(run=run.name, engine_commit=COMMIT, stage=progress.get('stage', 'prepared'),
                episodes=count, planned=cfg['expected_points'], accepted=accepted,
                analysis=str(run/'analysis/TRADEOFFS.md')))
        current_signature = tuple((s['run'], s['stage'], s['episodes'], s['accepted']) for s in status)
        if current_signature != aggregate_signature:
            subprocess.run([sys.executable, str(ROOT/'analyze_e6_controls.py')], check=True)
            aggregate_signature = current_signature
        watcher_state = dict(pid=os.getpid(), updated_at=time.time(),
            stage='finished' if handoff['stage'] in TERMINAL else 'watching',
            queue_stage=handoff['stage'], runs=status,
            scope='E6 only; E5 baselines remain separately audited.')
        save(args.status, watcher_state)
        prior = read(args.prior_watcher)
        if prior.get('stage') in {'finished_direct_guard', 'finished'}:
            current = Path(handoff['current']) if handoff.get('current') else None
            state = read(ROOT/'STATUS.json')
            state.update(updated_at=time.time(), goal_complete=False,
                source_commit=COMMIT,
                baseline_source_commit='37e0828c642ce4f3795c4c4394f4ce913bcd1a35',
                matrix_queue_stage=handoff['stage'], matrix_queue_status=args.handoff.name,
                matrix_queue_cancel=args.handoff.with_suffix('.cancel').name,
                matrix_queue_exec_session=args.queue_exec_session,
                matrix_analysis_watcher_pid=os.getpid(), analysis_watcher_status=args.status.name,
                direct_guard_exec_session=None, active_run=current.name if current else None,
                e6_validation_runs=status,
                previous_goal_turn_classification='progress: E5 baselines retained; E6 optional-checkpoint repair undergoing independent GPU validation')
            if current:
                progress = read(current/'task/progress.json') if (current/'task/progress.json').exists() else {}
                cfg = read(current/'config.json')
                state.update(active_stage=progress.get('stage', 'prepared'),
                    active_completed_episodes=progress.get('valid_points', 0), active_protocol=cfg['protocol'])
            else:
                state.update(active_stage=handoff['stage'])
            save(ROOT/'STATUS.json', state)
        if handoff['stage'] in TERMINAL:
            return
        time.sleep(30)
    save(args.status, dict(pid=os.getpid(), stage='watcher_deadline', updated_at=time.time()))


if __name__ == '__main__':
    main()
