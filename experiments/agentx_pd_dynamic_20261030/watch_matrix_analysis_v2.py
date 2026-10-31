"""Refresh CPU-only analysis as immutable formal episode files appear."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from analyze_balanced_matrix import ROOT, RUNS


def main():
    status = ROOT/'matrix_analysis_watcher_v2.json'
    cancel = status.with_suffix('.cancel')
    last = None
    deadline = time.monotonic()+12*3600
    while time.monotonic()<deadline and not cancel.exists():
        handoff = json.loads((ROOT/'handoff_durable_matrix_e5.json').read_text())
        signature = tuple((str(p), p.stat().st_size) for n in RUNS
            for p in sorted((ROOT/'runs'/n).glob('results/*/*/*/completed.r*.json')))
        marker = signature, handoff['stage'], handoff.get('current')
        if marker != last:
            subprocess.run([sys.executable,str(ROOT/'analyze_balanced_matrix.py')],check=True)
            last = marker
        state_path = ROOT/'STATUS.json'
        state = json.loads(state_path.read_text())
        current = handoff.get('current')
        if current:
            run = Path(current)
            progress = json.loads((run/'task/progress.json').read_text())
            cfg = json.loads((run/'config.json').read_text())
            state.update(active_run=run.name,active_stage=progress['stage'],
                active_protocol=cfg['protocol'],active_completed_episodes=progress.get('valid_points',0))
        else:
            state.update(active_run=None,active_stage=handoff['stage'])
        state.update(updated_at=time.time(),matrix_queue_stage=handoff['stage'],
            matrix_analysis_watcher_pid=os.getpid(),goal_complete=False)
        temp = state_path.with_suffix('.matrix.tmp')
        temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
        status.write_text(json.dumps(dict(pid=os.getpid(),updated_at=time.time(),
            stage='watching',completed_episodes=len(signature),queue_stage=handoff['stage']),indent=2)+'\n')
        if handoff['stage'] in ('matrix_complete','stopped_on_failed_gate','cancelled_before_successor'):
            subprocess.run([sys.executable,str(ROOT/'analyze_balanced_matrix.py')],check=True)
            status.write_text(json.dumps(dict(pid=os.getpid(),updated_at=time.time(),
                stage='finished',completed_episodes=len(signature),queue_stage=handoff['stage']),indent=2)+'\n')
            return
        time.sleep(30)
    status.write_text(json.dumps(dict(pid=os.getpid(),updated_at=time.time(),stage='cancelled_or_deadline'),indent=2)+'\n')


if __name__=='__main__':
    main()
