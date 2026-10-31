"""One-shot continuation of the already-authorized two-node experiment.

Wait for verified cleanup/release of the current uniform control, launch the
prepared owner-balanced control through Guard, and refresh the Markdown archive
at episode boundaries. This never books nodes or edits frozen run inputs.
Creating continuation.cancel prevents a queued launch; stopping an active model
still uses its owned controller/Guard cleanup, not this CPU monitor.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
ROOT=WORK/'experiments/agentx_pd_dynamic_20261030/runs'
PREVIOUS=ROOT/'two_p_six_d_3ep2_e8_v4_gdn256_20261031_control'
NEXT=ROOT/'two_p_six_d_ep4ep2_owner_balanced_e8_v4_gdn256_20261031_control'
GUARD=WORK/'experiments/h100_small_batch_layout_verify_20261029/guard_bundle_posix_v1/skills/llm-experiment-guard/scripts/dashboard_guard.py'
PYTHON='/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python'
CANCEL=HERE/'continuation.cancel'
STATE=HERE/'continuation.json'


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def state(stage, **kwargs):
    value=dict(stage=stage,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        owner_pid=os.getpid(),owner_start_ticks=Path('/proc/self/stat').read_text().split()[21],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        current_control=str(PREVIOUS),queued_control=str(NEXT),cancel_file=str(CANCEL),**kwargs)
    temporary=STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(STATE)


def refresh():
    with (HERE/'continuation_reports.log').open('a') as log:
        for script in (HERE/'analyze_pilot.py',HERE.parent/'agentx_workload_profile_20261031/archive.py'):
            subprocess.run([sys.executable,str(script)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)


def reconcile_backfill():
    trigger=Path('/home/bxb1/.codex/skills/llm-experiment-guard/scripts/backfill_trigger.py')
    result=subprocess.run([sys.executable,str(trigger)],capture_output=True,text=True,timeout=180)
    value=dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        invoked_after_verified_release=str(NEXT),returncode=result.returncode,
        stdout=result.stdout,stderr=result.stderr)
    (HERE/'backfill_reconciliation.json').write_text(json.dumps(value,indent=2)+'\n')


def released(run):
    cleanup=read(run/'task/cleanup.json')
    if not cleanup or not all(cleanup.get(k) for k in ('passed','gpu_idle','model_processes_exited')):
        return False
    checks=[read(p) for p in (run/'dashboard_guard/checks').glob('*-post.json')]
    if not any(x and x.get('cleanup_verified') is True for x in checks):return False
    log=run/'guard.supervisor.log'
    return log.exists() and '"gpu_registry_state": "released"' in log.read_text()


def wait_release(run, *, queued=False):
    previous=None
    while True:
        if queued and CANCEL.exists():
            state('cancelled_before_launch');refresh();return False
        progress=read(run/'task/progress.json') or {}
        key=(progress.get('stage'),tuple(sorted(str(p) for p in run.glob('results/*/*/*/completed.r*.json'))),
             tuple(sorted(str(p) for p in run.glob('results/*/*/*/partial.r*.json'))))
        if key!=previous:
            refresh();previous=key
        if released(run):return True
        time.sleep(10)


def main():
    assert not STATE.exists(), 'An existing continuation needs ownership review'
    assert (NEXT/'dashboard_spec.json').exists()
    state('waiting_for_current_control_cleanup')
    if not wait_release(PREVIOUS,queued=True):return
    if CANCEL.exists():
        state('cancelled_before_launch');return
    command=[sys.executable,str(GUARD),'run','--spec',str(NEXT/'dashboard_spec.json'),
        '--resource-choice','user','--node-limit','2','--requested-gpus','8',
        '--gpus-per-node','4','--preempt-conflicts','--',PYTHON,
        str(NEXT/'code/cluster_runner.py'),'--run',str(NEXT)]
    with (NEXT/'guard.supervisor.log').open('xb',buffering=0) as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state('running_owner_balanced_control',guard_pid=child.pid,dashboard_url='http://127.0.0.1:29836/')
        while not released(NEXT):
            if child.poll() is not None:
                refresh();state('guard_exited_before_verified_release',guard_exit=child.returncode);return
            progress=read(NEXT/'task/progress.json') or {}
            current=progress.get('stage')
            if current!=getattr(main,'previous_stage',None):
                refresh();main.previous_stage=current
            # Also refresh when an episode ends without a stage transition.
            markers=len(list(NEXT.glob('results/*/*/*/completed.r*.json')))
            if markers!=getattr(main,'previous_markers',-1):
                refresh();main.previous_markers=markers
            time.sleep(10)
        reconcile_backfill()
        refresh()
        analysis=read(HERE/'pilot_analysis.json')
        state('measurement_finished_dashboard_supervised',guard_pid=child.pid,
            accepted_primary_episodes=analysis['accepted_primary_episodes'],expected_primary_episodes=4,
            cleanup_verified=analysis['cleanup']['passed'],dashboard_url='http://127.0.0.1:29836/')
        refresh()
        # Guard deliberately retains the dashboard after releasing all GPUs.
        # Keep its parent supervised in the same foreground exec session.
        child.wait()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        state('continuation_error',error_type=type(error).__name__,error=str(error))
        raise
