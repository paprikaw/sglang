"""Guard-authenticated worlds in existing Slurm steps; disjoint worlds parallel."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from common import ROOT,save,sha
from guard_authorization import (GUARD_KEYS,authorization_signature,close_remote_workers,
    process_ticks,validate_guard,visible_json)
from node import phase_for


def step_command(cfg,phase,run,node):
    job=cfg['booked_jobs'][node]
    assert job.isdigit() and node in phase['nodes']
    return ['srun','--jobid',job,'--overlap','--exact','--nodes=1','--ntasks=1',
        '--cpus-per-task=64','--gres=gpu:4','--nodelist',node,'--unbuffered',
        '/home/bxb1/conda/bin/python',str(ROOT/'node.py'),'--run',str(run),'--phase',phase['id'],'--controller-stdin']


def validate_exit(row,node,phase_id):
    assert row.get('node')==node and row.get('phase')==phase_id and row.get('returncode')==0
    assert all(row.get(k) is True for k in ('worker_exited','owned_processes_exited','gpu_idle'))
    assert row.get('gpu_query',{}).get('post_exit_only') is True
    assert row.get('gpu_query',{}).get('passed') is True
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run.resolve()
    cfg=json.loads((run/'config.json').read_text())
    task=run/'task'
    receipts=run/'artifact_visibility/controller'
    receipt=validate_guard(run,cfg,os.environ,receipts,require_parent=True)
    save(task/'guard_authorization.json',dict(lease_id=receipt['lease_id'],experiment_id=cfg['id'],
        nodes=cfg['nodes'],scope='communication_only_existing_slurm_steps'))
    cancelled={'signal':0}
    def interrupted(signum,_frame):
        cancelled['signal']=signum
    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(signum,interrupted)
    pid,ticks,started=os.getpid(),process_ticks(os.getpid()),time.time()
    completed,launched=[],[]
    procs,logs=[],[]
    valid,rc=False,1

    def progress(stage,phases,note):
        point_count=sum(len(list((run/p['id']).glob('point_*_rank0.json'))) for p in cfg['phases'])
        save(task/'progress.json',dict(timestamp=time.time(),controller_pid=pid,process_start_ticks=ticks,
            stage=stage,phase=','.join(phases),case=','.join(phases),valid_points=len(completed),
            completed_phases=len(completed),measured_workloads=point_count,total_workloads=624,
            elapsed_s=time.time()-started,note=note))

    try:
        for group in cfg['phase_groups']:
            phases=[phase_for(cfg,phase_id) for phase_id in group]
            used=[node for phase in phases for node in phase['nodes']]
            assert len(used)==len(set(used)), 'Concurrent worlds must have disjoint whole nodes'
            if cancelled['signal']:
                raise InterruptedError('Communication experiment cancelled')
            procs,logs=[],[]
            for phase in phases:
                out=run/phase['id']
                out.mkdir(exist_ok=False)
                for node in phase['nodes']:
                    payload=dict(guard_env={key:os.environ[key] for key in GUARD_KEYS},
                        lease_id=receipt['lease_id'],node=node,phase=phase['id'],config_sha256=sha(run/'config.json'))
                    command=step_command(cfg,phase,run,node)
                    log=(out/f'{node}.log').open('w')
                    logs.append(log)
                    proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,
                        text=True,start_new_session=True)
                    procs.append(proc)
                    launched.append((phase['id'],node))
                    proc.stdin.write(json.dumps(dict(payload=payload,signature=authorization_signature(payload,receipt['token'])))+'\n')
                    proc.stdin.flush()
            deadline=time.monotonic()+1600
            while any(proc.poll() is None for proc in procs):
                if cancelled['signal']:
                    raise InterruptedError('Communication experiment cancelled')
                if any(proc.poll() not in (None,0) for proc in procs):
                    raise RuntimeError('Communication world failed: '+','.join(group))
                if time.monotonic()>=deadline:
                    raise TimeoutError('Communication world deadline')
                progress('running',group,'实际 DeepEP Dispatch/Combine；初始化、正确性验证不计时')
                time.sleep(1)
            close_remote_workers(procs)
            for log in logs:
                log.close()
            if any(proc.returncode!=0 for proc in procs):
                raise RuntimeError('Nonzero Slurm step exit: '+','.join(group))
            for phase in phases:
                for node in phase['nodes']:
                    validate_exit(visible_json(run/phase['id']/f'{node}_exit.json',receipts,timeout=30),node,phase['id'])
                for rank in range(phase['world']):
                    result=visible_json(run/phase['id']/f'result_rank{rank}.json',receipts,timeout=30)
                    assert result.get('passed') is True and result.get('model_loaded') is False
                completed.append(phase['id'])
        subprocess.run(['/home/bxb1/conda/bin/python',str(ROOT/'analyze.py'),'--run',str(run)],check=True)
        valid=visible_json(run/'comparison.json',receipts).get('passed') is True
        rc=0 if valid else 1
    except BaseException as error:
        save(run/'controller_failure.json',dict(timestamp=time.time(),error_type=type(error).__name__,
            error=str(error),completed_phases=completed))
    finally:
        close_remote_workers(procs)
        for log in logs:
            if not log.closed:
                log.close()
        all_clean=bool(launched)
        proof=[]
        for phase_id,node in launched:
            try:
                row=visible_json(run/phase_id/f'{node}_exit.json',receipts,timeout=30)
                clean=all(row.get(k) is True for k in ('worker_exited','owned_processes_exited','gpu_idle'))
                clean=clean and row.get('phase')==phase_id and row.get('node')==node
                proof.append(row)
            except (OSError,ValueError,TimeoutError):
                clean=False
            all_clean=all_clean and clean
        progress('complete' if valid else 'failed',completed[-1:] if completed else ['init'],
            '通信网格与全部原始事件已保存' if valid else '保留失败及已测点，未将失败运行标记完成')
        save(task/'cleanup.json',dict(passed=all_clean,model_processes_exited=all_clean,gpu_idle=all_clean,
            experiment_id=cfg['id'],task_id='local',timestamp=time.time(),evidence=proof))
    return 0 if valid and all_clean else rc or 1


if __name__=='__main__':
    raise SystemExit(main())
