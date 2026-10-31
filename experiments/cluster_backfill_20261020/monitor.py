"""Read-only view over shared reconciliation and exact benchmark workers."""
from pathlib import Path
import fcntl
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'tools/idle_gpu_benchmark'))
from prefill_monitor import save, query_gpu, fresh


def read(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, ValueError):
        return {}
    # A filesystem outage must not turn the last good snapshot into fresh empty
    # state. Other OSErrors reach poll_once, which retains the atomic old file.


def error_label(exc):
    label = type(exc).__name__
    if isinstance(exc, OSError) and exc.errno is not None:
        label += f' errno={exc.errno}'
    return label


def warn(message):
    try:
        print(f'monitor: {message}', file=sys.stderr, flush=True)
    except OSError:
        # The supervisor log may be on the same temporarily full filesystem.
        pass


def worker_snapshot(node, run, deployment, worker, control, now):
    """Keep an exact-run worker failure distinct from model readiness.

    A final controller record remains authoritative after the load generator
    exits, even if its last live JSON still says running. Preserve the last
    sample timestamp for age display without treating it as current telemetry.
    """
    matches = (bool(run) and worker.get('run') == run
               and worker.get('node', node) == node)
    current = matches and fresh(worker.get('timestamp'), now, 20)
    control_state = control.get('state') if control.get('node', node) == node else None
    if not run:
        status = 'not_applicable'
    elif control_state in ('failed', 'exited', 'stopped', 'cleanup_unverified'):
        status = control_state
        current = False
    elif current:
        status = worker.get('status') or 'unknown'
    elif deployment.get('status') == 'healthy':
        status = 'stale' if matches or control_state == 'running' else 'awaiting_worker'
    else:
        status = 'loading'
    error = control.get('error') if status == control_state else worker.get('error') if current else None
    if isinstance(error, dict):
        error = {key: str(error[key])[:256] for key in ('type', 'message') if key in error}
    elif error is not None:
        error = str(error)[:256]
    return dict(worker_updated_at=worker.get('timestamp') if matches else None,
                worker_status=status, worker_stale=not current,
                worker_error=error, worker_exit_code=control.get('exit_code'),
                deployment_status=deployment.get('status'))


def snapshot(root):
    state = read(root / 'state.json')
    services = read(root / 'services.json').get('services', {})
    rows = {}
    for node, status in state.get('nodes', {}).items():
        service = services.get(node, {})
        service_root = Path(service.get('root', '/tmp/nonexistent-backfill'))
        attempt = read(service_root / 'state.json').get('attempts', {}).get(node, {})
        run = attempt.get('run')
        deployment = read(Path(run) / 'deployment.json') if run else {}
        worker = read(service_root / 'prefill/nodes' / f'{node}.json')
        control = read(Path(run) / 'benchmark_worker.json') if run else {}
        telemetry = worker_snapshot(node, run, deployment, worker, control, time.time())
        row = dict(
            run=run,
            deployment_id=deployment.get('deployment_id'),
            model=Path(deployment.get('model_path', 'unknown')).name,
            state_reason=status.get('reason') if status.get('state') in
                ('waiting', 'unknown', 'background_unverified') else None)
        row.update(telemetry)
        if not telemetry['worker_stale'] and status.get('state') == 'benchmark':
            try:
                gpu = query_gpu(node)
                row.update(gpus=gpu['gpus'], gpu_stale=bool(gpu['error']),
                           gpu_error=gpu['error'])
            except (OSError, subprocess.SubprocessError) as exc:
                # Sampling is independent per node. Keep healthy samples while
                # explicitly invalidating this node, without reusing old data.
                reason = error_label(exc)
                warn(f'GPU sample failed for {node}: {reason}')
                row.update(gpus=None, gpu_stale=True, gpu_error=reason)
        rows[node] = row
    return dict(timestamp=time.time(), nodes=rows)


def poll_once(root):
    try:
        data = snapshot(root)
        save(root / 'all_nodes_state.json', data)
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        # save uses temp-file + replace. EDQUOT/ENOSPC or a transient I/O error
        # therefore leaves the previous snapshot and its timestamp untouched.
        warn(f'snapshot deferred: {error_label(exc)}')
        return False


def run_loop(root, interval=5):
    while True:
        poll_once(root)
        time.sleep(interval)


def main():
    with (ROOT / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_loop(ROOT)


if __name__ == '__main__':
    main()
