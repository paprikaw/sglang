"""Track one already-launched exact Guard owner; never launch GPU work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from wait_then_qualify_v2 import identity
from analyze_balanced_matrix import ROOT,RUNS


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--status',type=Path,required=True)
    ap.add_argument('--exec-session',type=int,required=True)
    args=ap.parse_args();run=args.run.resolve()
    assert not args.status.exists()
    lease=json.loads((run/'dashboard_guard/gpu_registry_lease.json').read_text())
    expected=lease['owner_pid'],lease['owner_start_ticks']
    previous=None
    deadline=time.monotonic()+12*3600
    while time.monotonic()<deadline:
        owner=identity(expected[0])
        live=owner is not None and owner['start_ticks']==expected[1] and owner['state']!='Z'
        progress=json.loads((run/'task/progress.json').read_text())
        marker=tuple((str(p),p.stat().st_size) for name in RUNS
            for p in sorted((ROOT/'runs'/name).glob('results/*/*/*/completed.r*.json'))),live
        if marker!=previous:
            for script in ('analyze_balanced_matrix.py','compare_static_matrix.py'):
                subprocess.run([sys.executable,str(ROOT/script)],check=True)
            previous=marker
        stage='watching_direct_guard' if live else 'finished_direct_guard'
        state_path=ROOT/'STATUS.json'
        state=json.loads(state_path.read_text())
        cfg=json.loads((run/'config.json').read_text())
        state.update(updated_at=time.time(),active_run=run.name,active_stage=progress['stage'],
            active_protocol=cfg['protocol'],active_completed_episodes=progress.get('valid_points',0),
            direct_guard_exec_session=args.exec_session,matrix_queue_stage=stage,
            matrix_queue_exec_session=None,analysis_watcher_status=args.status.name,
            matrix_analysis_watcher_pid=os.getpid(),goal_complete=False,
            previous_goal_turn_classification='progress: PP2 accepted, strict-SLO overlap audited; PD C8 complete but C128 GDN-cache allocation failed; single baseline launched after verified cleanup')
        temp=state_path.with_suffix('.direct.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
        args.status.write_text(json.dumps(dict(pid=os.getpid(),stage=stage,
            updated_at=time.time(),run=run.name,guard_pid=expected[0],
            guard_start_ticks=expected[1],run_stage=progress['stage'],
            completed_episodes=len(marker[0])),indent=2)+'\n')
        if not live:
            return
        time.sleep(30)
    args.status.write_text(json.dumps(dict(pid=os.getpid(),stage='watcher_deadline',
        run=run.name,guard_pid=expected[0],guard_start_ticks=expected[1]),indent=2)+'\n')


if __name__=='__main__':
    main()
