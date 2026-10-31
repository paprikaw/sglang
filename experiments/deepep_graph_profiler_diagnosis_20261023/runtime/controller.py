"""Guard direct CPU child: sequential fresh EP8 processes, no model or bank."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from common import ROOT,MODES,save,sha
from guard_authorization import (GUARD_KEYS,validate_guard,authorization_signature,
                                 close_remote_workers,visible_json)
from node_runner import clean_exit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    run=parser.parse_args().run.resolve();cfg=json.loads((run/'config.json').read_text())
    receipt=validate_guard(run,cfg,os.environ,run/'visibility/controller',require_parent=True)
    task=run/'task';cancelled={'signal':0}
    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(signum,lambda number,_:cancelled.update(signal=number))
    procs,logs,launched,outcomes,failure=[],[],[],{},None
    try:
        for mode in cfg['modes']:
            if cancelled['signal'] or (run/'STOP').exists():
                raise InterruptedError('Diagnostic cancelled')
            output=run/mode;output.mkdir(parents=True,exist_ok=False)
            save(task/'progress.json',dict(stage='running',mode=mode,timestamp=time.time(),
                diagnostic_cases_completed=len(outcomes),valid_points=sum(x['profile_valid'] for x in outcomes.values())))
            procs,logs=[],[]
            for node in cfg['nodes']:
                remote=shlex.join(['/usr/bin/python3',str(ROOT/'runtime/node_runner.py'),
                    '--run',str(run),'--mode',mode])
                log=(output/f'{node}_ssh.log').open('x');logs.append(log)
                proc=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none',
                    '-o','ConnectTimeout=5',node,remote],stdin=subprocess.PIPE,stdout=log,
                    stderr=subprocess.STDOUT,text=True,start_new_session=True)
                procs.append(proc);launched.append((mode,node))
                payload=dict(guard_env={key:os.environ[key] for key in GUARD_KEYS},node=node,
                    mode=mode,lease_id=receipt['lease_id'],config_sha256=sha(run/'config.json'))
                proc.stdin.write(json.dumps(dict(payload=payload,
                    signature=authorization_signature(payload,receipt['token'])))+'\n');proc.stdin.flush()
            deadline=time.monotonic()+cfg['worker_timeout_s']+45
            while any(p.poll() is None for p in procs):
                if cancelled['signal'] or (run/'STOP').exists():
                    raise InterruptedError('Guard/owner cancelled diagnosis')
                if time.monotonic()>deadline:
                    raise TimeoutError('Bounded two-node diagnostic deadline')
                # Node supervisors return zero after a clean native crash; a
                # nonzero here is lost ownership/cleanup and cannot continue.
                if any(p.poll() not in (None,0) for p in procs):
                    raise RuntimeError('Owned remote supervisor failed cleanup')
                time.sleep(.2)
            close_remote_workers(procs)
            for log in logs:log.close()
            exits=[visible_json(output/f'{node}_exit.json',run/'visibility/controller',30) for node in cfg['nodes']]
            if not all(clean_exit(x) and x['node']==node and x['mode']==mode for x,node in zip(exits,cfg['nodes'])):
                raise RuntimeError('Mode cannot continue without exact owned cleanup')
            ranks=[]
            for rank in range(8):
                path=output/f'result_rank{rank}.json'
                ranks.append(visible_json(path,run/'visibility/controller',30) if path.exists()
                    else dict(rank=rank,passed=False,profile_valid=False,result_missing=True))
            outcomes[mode]=dict(node_exits=exits,ranks=ranks,diagnostic_attempted=True,
                payload_valid=all(x.get('passed') is True for x in ranks),
                profile_valid=all(x.get('profile_valid') is True for x in ranks))
        save(task/'results.json',dict(passed=True,diagnostic_cases_completed=len(outcomes),
            valid_profile_modes=sum(x['profile_valid'] for x in outcomes.values()),modes=outcomes,
            note='Completion means modes attempted and cleaned. Missing/crashed traces are invalid and never substituted with estimates.'))
    except BaseException as error:
        failure=dict(type=type(error).__name__,message=str(error));save(task/'failure.json',failure)
    finally:
        close_remote_workers(procs)
        for log in logs:
            if not log.closed:log.close()
        proof=[];clean=bool(launched)
        for mode,node in launched:
            try:
                row=visible_json(run/mode/f'{node}_exit.json',run/'visibility/controller',30)
                proof.append(row);clean=clean and clean_exit(row) and row.get('node')==node and row.get('mode')==mode
            except Exception:clean=False
        save(task/'progress.json',dict(stage='complete' if failure is None else 'failed',timestamp=time.time(),
            diagnostic_cases_completed=len(outcomes),valid_points=sum(x['profile_valid'] for x in outcomes.values()),
            planned_points=len(cfg['modes'])))
        save(task/'cleanup.json',dict(passed=clean,gpu_idle=clean,gpu_resources_released=clean,
            model_processes_exited=clean,experiment_id=cfg['id'],task_id='lifecycle',timestamp=time.time(),evidence=proof))
    return 0 if failure is None and clean else 1


if __name__=='__main__':
    raise SystemExit(main())
