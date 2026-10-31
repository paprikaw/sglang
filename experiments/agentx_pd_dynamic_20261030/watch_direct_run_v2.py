"""Observe one exact Guard owner and analyze only its completed episodes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from wait_then_qualify_v2 import identity

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.watcher.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--status', type=Path, required=True)
    ap.add_argument('--exec-session', type=int, required=True)
    args = ap.parse_args()
    assert not args.status.exists()
    run = args.run.resolve()
    cfg = read(run/'config.json')
    manifest = read(run/'launch_manifest.json')
    lease = read(run/'dashboard_guard/gpu_registry_lease.json')
    expected = lease['owner_pid'], lease['owner_start_ticks']
    previous = None
    deadline = time.monotonic()+12*3600
    while time.monotonic() < deadline:
        owner = identity(expected[0])
        live = bool(owner and owner['start_ticks'] == expected[1] and owner['state'] != 'Z')
        progress = read(run/'task/progress.json')
        count = len(list(run.glob('results/*/*/*/completed.r*.json')))
        signature = count, live
        if count and signature != previous:
            for script in ('analyze_cluster.py', 'analyze_exposure.py', 'analyze_tradeoffs.py'):
                subprocess.run([sys.executable, str(ROOT/script), '--run', str(run)], check=True)
        previous = signature
        clean = read(run/'task/cleanup.json') if (run/'task/cleanup.json').exists() else {}
        posts = sorted((run/'dashboard_guard/checks').glob('*-post.json'))
        accepted = bool(not live and count == cfg['expected_points'] and progress['stage'] == 'complete'
                        and clean.get('passed') and clean.get('gpu_idle')
                        and posts and read(posts[-1])['passed'])
        stage = 'watching_direct_guard' if live else 'finished_direct_guard'
        observed = dict(pid=os.getpid(), stage=stage, updated_at=time.time(),
            run=run.name, engine_commit=manifest['engine_commit'],
            guard_pid=expected[0], guard_start_ticks=expected[1],
            run_stage=progress['stage'], completed_episodes=count,
            planned_episodes=cfg['expected_points'], run_accepted=accepted)
        save(args.status, observed)
        state = read(ROOT/'STATUS.json')
        state.update(updated_at=time.time(), goal_complete=False,
            source_commit=manifest['engine_commit'], active_run=run.name,
            active_stage=progress['stage'], active_protocol=cfg['protocol'],
            active_completed_episodes=count, direct_guard_exec_session=args.exec_session,
            matrix_queue_stage=stage, matrix_queue_exec_session=None,
            analysis_watcher_status=args.status.name, matrix_analysis_watcher_pid=os.getpid(),
            active_run_observation=observed,
            previous_goal_turn_classification='progress: bounded HiCache restore retry under independent GPU qualification')
        save(ROOT/'STATUS.json', state)
        if not live:
            return
        time.sleep(30)
    save(args.status, dict(pid=os.getpid(), stage='watcher_deadline', run=run.name))


if __name__ == '__main__':
    main()
