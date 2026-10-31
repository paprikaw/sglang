"""Run the three already prepared matrix arms after an exact Guard owner exits.

Each transition requires completed scientific checks and verified GPU cleanup.
The cancellation file blocks subsequent launches; an active owned run is stopped
separately through its controller, preserving its cleanup path.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from wait_then_qualify_v2 import identity

ROOT = Path(__file__).resolve().parent


def completed_and_clean(run):
    progress = json.loads((run / 'task/progress.json').read_text())
    cleanup = json.loads((run / 'task/cleanup.json').read_text())
    checks = sorted((run / 'dashboard_guard/checks').glob('*-post.json'))
    assert progress['stage'] == 'complete', progress
    assert cleanup['passed'] and cleanup['gpu_idle'], cleanup
    assert checks and json.loads(checks[-1].read_text())['passed']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predecessor', type=Path, required=True)
    ap.add_argument('--successor', type=Path, action='append', required=True)
    ap.add_argument('--status', type=Path, required=True)
    args = ap.parse_args()
    current = args.predecessor.resolve()
    queue = [p.resolve() for p in args.successor]
    state = {'pid': os.getpid(), 'identity': identity(os.getpid()),
             'completed': [], 'queue': [str(p) for p in queue]}
    with os.fdopen(os.open(args.status, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), 'w') as f:
        json.dump(state, f)

    def save(stage, **extra):
        state.update(stage=stage, timestamp=time.time(), **extra)
        temp = args.status.with_suffix('.tmp')
        with os.fdopen(os.open(temp, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600), 'w') as f:
            json.dump(state, f, indent=2)
        temp.replace(args.status)

    cancel = args.status.with_suffix('.cancel')
    lease = json.loads((current / 'dashboard_guard/gpu_registry_lease.json').read_text())
    expected = (lease['owner_pid'], lease['owner_start_ticks'])
    deadline = time.monotonic() + 8 * 3600
    try:
        while True:
            if cancel.exists():
                save('cancelled_before_successor', current=str(current))
                return 0
            proc = identity(expected[0])
            if proc is None or proc['start_ticks'] != expected[1] or proc['state'] == 'Z':
                break
            assert time.monotonic() < deadline, 'Predecessor deadline exceeded'
            save('waiting_for_exact_predecessor', current=str(current), owner=proc)
            time.sleep(10)
        completed_and_clean(current)
        for script in ('analyze_cluster.py', 'analyze_exposure.py', 'analyze_tradeoffs.py'):
            with (current / (script.removesuffix('.py') + '.log')).open('w') as log:
                subprocess.run([sys.executable, str(ROOT / script), '--run', str(current)], check=True,
                               stdout=log, stderr=subprocess.STDOUT)
        state['completed'].append(str(current))
        for run in queue:
            if cancel.exists():
                save('cancelled_before_successor', current=None)
                return 0
            assert not (run / 'supervisor.log').exists(), 'Never overwrite a launch attempt'
            spec = json.loads((run / 'dashboard_spec.json').read_text())
            for source in spec['frozen_inputs']:
                assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256'], source['path']
            cfg = json.loads((run / 'config.json').read_text())
            first = cfg['node_configs'][cfg['nodes'][0]]
            save('reconciling_capacity', current=str(run))
            subprocess.run([sys.executable, '/home/bxb1/.codex/skills/llm-experiment-guard/scripts/backfill_trigger.py'], check=True)
            if cancel.exists():
                save('cancelled_before_successor', current=None)
                return 0
            command = [sys.executable, str(Path(first['guard_scripts']) / 'dashboard_guard.py'),
                       'run', '--spec', str(run / 'dashboard_spec.json'),
                       '--resource-choice', 'user', '--node-limit', '2', '--requested-gpus', '8',
                       '--exit-after-run', '--', sys.executable, str(run / 'code/cluster_runner.py'), '--run', str(run)]
            with (run / 'supervisor.log').open('xb') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT.parents[1])
                save('successor_guard_running', current=str(run), owner=identity(child.pid))
                code = child.wait()
            assert code == 0, f'Guard exited {code} for {run.name}'
            completed_and_clean(run)
            state['completed'].append(str(run))
            for script in ('analyze_cluster.py', 'analyze_exposure.py', 'analyze_tradeoffs.py'):
                with (run / (script.removesuffix('.py') + '.log')).open('w') as log:
                    subprocess.run([sys.executable, str(ROOT / script), '--run', str(run)], check=True,
                                   stdout=log, stderr=subprocess.STDOUT)
        save('matrix_complete', current=None)
        return 0
    except BaseException as exc:
        save('stopped_on_failed_gate', error=repr(exc))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
