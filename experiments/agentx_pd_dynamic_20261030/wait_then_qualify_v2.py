"""Chain one pre-authorized Guard qualification after a specific live run ends.

Never claims while the predecessor's original supervisor is alive. Failure,
missing cleanup proof, or an explicit cancellation file stops the chain.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def identity(pid):
    path = Path(f'/proc/{pid}')
    try:
        if path.stat().st_uid != os.getuid():
            return None
        tail = (path/'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'state': tail[0], 'start_ticks': int(tail[19])}
    except FileNotFoundError:
        return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--predecessor',type=Path,required=True)
    parser.add_argument('--successor',type=Path,required=True)
    parser.add_argument('--status',type=Path,required=True)
    args=parser.parse_args()
    pred=args.predecessor.resolve();succ=args.successor.resolve()
    lease=json.loads((pred/'dashboard_guard/gpu_registry_lease.json').read_text())
    expected=(lease['owner_pid'],lease['owner_start_ticks'])
    status={'pid':os.getpid(),'identity':identity(os.getpid()),'predecessor':str(pred),'successor':str(succ)}
    args.status.parent.mkdir(parents=True,exist_ok=True)
    with os.fdopen(os.open(args.status,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600),'w') as f:
        json.dump(status,f)
    def save(stage,**extra):
        status.update(stage=stage,timestamp=time.time(),**extra)
        temp=args.status.with_suffix('.tmp')
        with os.fdopen(os.open(temp,os.O_CREAT|os.O_TRUNC|os.O_WRONLY,0o600),'w') as f:
            json.dump(status,f,indent=2)
        temp.replace(args.status)
    cancel=args.status.with_suffix('.cancel')
    deadline=time.monotonic()+6*3600
    while True:
        if cancel.exists():
            save('cancelled_before_launch');return 0
        proc=identity(expected[0])
        if proc is None or proc['start_ticks'] != expected[1] or proc['state'] == 'Z':
            break
        if time.monotonic()>deadline:
            save('wait_deadline_exceeded');return 2
        save('waiting_for_exact_predecessor_supervisor',observed_predecessor=proc)
        time.sleep(10)
    progress=json.loads((pred/'task/progress.json').read_text())
    cleanup=json.loads((pred/'task/cleanup.json').read_text())
    if progress['stage']!='complete' or not cleanup['passed'] or not cleanup['gpu_idle']:
        save('predecessor_not_complete_and_clean',predecessor_stage=progress['stage']);return 2
    checks=sorted((pred/'dashboard_guard/checks').glob('*-post.json'))
    if not checks or not json.loads(checks[-1].read_text()).get('passed'):
        save('predecessor_postcheck_not_passed');return 2
    assert not (succ/'supervisor.log').exists(), 'Successor already has a launch attempt'
    cfg=json.loads((succ/'config.json').read_text())
    save('reconciling_background_capacity')
    subprocess.run([sys.executable,'/home/bxb1/.codex/skills/llm-experiment-guard/scripts/backfill_trigger.py'],check=True)
    if cancel.exists():
        save('cancelled_before_launch');return 0
    command=[sys.executable,str(Path(cfg['guard_scripts'])/'dashboard_guard.py'),'run','--spec',str(succ/'dashboard_spec.json'),
             '--resource-choice','user','--node-limit','1','--requested-gpus','4','--exit-after-run','--',
             sys.executable,str(succ/'code/runner.py'),'--run',str(succ)]
    with (succ/'supervisor.log').open('xb') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=Path(__file__).resolve().parents[2])
        save('successor_guard_running',successor_supervisor_pid=child.pid,successor_identity=identity(child.pid))
        code=child.wait()
    save('successor_guard_finished',exit_code=code)
    return code


if __name__=='__main__':
    raise SystemExit(main())
