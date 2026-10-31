"""Authenticated no-model node supervisor with owned-process cleanup."""
import argparse
import hmac
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from common import ROOT, PYTHON, NATIVE_SO, DEEP_COMMIT, HOST_COMMIT, environment, save, sha
from guard_authorization import authorization_signature, process_ticks, validate_guard
from process_control import track_descendants, stop_owned


def phase_for(cfg, phase_id):
    matches = [p for p in cfg['phases'] if p['id'] == phase_id]
    if len(matches) != 1:
        raise ValueError('Unknown or duplicate communication phase')
    phase = matches[0]
    rows = phase['placement']
    if len(rows) != phase['world'] or sorted(r['rank'] for r in rows) != list(range(phase['world'])):
        raise ValueError('Noncontiguous or missing phase ranks')
    if len({(r['node'], r['gpu']) for r in rows}) != len(rows):
        raise ValueError('Duplicate phase GPU')
    if any(r['node'] not in phase['nodes'] or r['gpu'] not in (0, 1, 2, 3) for r in rows):
        raise ValueError('Invalid node/GPU placement')
    if phase['world'] not in (4, 8) or not set(phase['nodes']).issubset(cfg['nodes']):
        raise ValueError('Phase placement is outside the registered allocation')
    return phase


def command_for(cfg, phase, run, node, native_sha=None):
    if node not in phase['nodes']:
        raise ValueError('Node is not in this communication phase')
    if native_sha is None:
        built = json.loads((run / 'native12/build_result.json').read_text())
        if built.get('passed') is not True or (built['source_commit'] != HOST_COMMIT or built['cuda_override_commit'] != DEEP_COMMIT) or built['native_path'] != str(NATIVE_SO) or sha(NATIVE_SO) != built['native_sha256']:
            raise ValueError('Native build receipt does not match the frozen source/output')
        native_sha = built['native_sha256']
    return [PYTHON, str(ROOT / 'rank_launcher.py'), '--run', str(run), '--phase', phase['id']]



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--phase', required=True)
    p.add_argument('--controller-stdin', action='store_true', required=True)
    args = p.parse_args()
    run = args.run.resolve()
    cfg = json.loads((run / 'config.json').read_text())
    phase = phase_for(cfg, args.phase)
    node = os.uname().nodename.split('.')[0]
    packet = json.loads(sys.stdin.readline())
    payload = packet['payload']
    receipt = validate_guard(run, cfg, payload['guard_env'], run / 'artifact_visibility' / node)
    if not hmac.compare_digest(packet['signature'], authorization_signature(payload, receipt['token'])):
        raise ValueError('Controller authorization signature mismatch')
    if (payload['lease_id'] != receipt['lease_id'] or payload['node'] != node
            or payload['phase'] != phase['id'] or payload['config_sha256'] != sha(run / 'config.json')):
        raise ValueError('Controller packet identity/config mismatch')
    cmd = command_for(cfg, phase, run, node)
    topology = subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True, text=True, timeout=20)
    save(run / phase['id'] / f'{node}_topology.json', {'node':node, 'phase':phase['id'], 'returncode':topology.returncode, 'stdout':topology.stdout, 'stderr':topology.stderr, 'gpu_order':[r['gpu'] for r in phase['placement'] if r['node']==node]})
    save(run / phase['id'] / f'{node}_launch.json', {
        'timestamp': time.time(), 'node': node, 'phase': phase['id'], 'command': cmd,
        'lease_id': receipt['lease_id'], 'model_loaded': False,
    })
    cancelled = {'signal': 0}
    def interrupted(signum, _frame):
        cancelled['signal'] = signum
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    link = select.poll()
    link.register(sys.stdin.fileno(), select.POLLIN | select.POLLHUP | select.POLLERR)
    child, owned, rc, reason, exited = None, {}, 1, 'launch_error', False
    try:
        env = environment(tuple(r['gpu'] for r in phase['placement'] if r['node'] == node))
        env['DEEPEP_LL_NVLINK_PAIR_ONLY'] = str(int(phase['pair_only']))
        child = subprocess.Popen(cmd, env=env, start_new_session=True, stdin=subprocess.DEVNULL)
        owned[child.pid] = process_ticks(child.pid)
        deadline = time.monotonic() + 600
        while child.poll() is None:
            track_descendants(child.pid, owned)
            if cancelled['signal']:
                rc, reason = 128 + cancelled['signal'], 'signal'
                break
            if time.monotonic() >= deadline:
                rc, reason = 124, 'timeout'
                break
            if link.poll(100) and not os.read(sys.stdin.fileno(), 4096):
                rc, reason = 125, 'controller_stdin_closed'
                break
        else:
            rc, reason = child.returncode, 'worker_exit'
    finally:
        if child is not None:
            exited = stop_owned(child, owned)
        try:
            query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
            gpu = {'returncode': query.returncode, 'stdout': query.stdout, 'stderr': query.stderr,
                   'post_exit_only': True, 'passed': query.returncode == 0 and not query.stdout.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            gpu = {'passed': False, 'post_exit_only': True, 'error': str(error)}
        save(run / phase['id'] / f'{node}_exit.json', {
            'timestamp': time.time(), 'node': node, 'phase': phase['id'], 'returncode': rc,
            'reason': reason, 'worker_exited': child is None or child.poll() is not None,
            'owned_processes_exited': exited, 'owned_process_count': len(owned),
            'gpu_idle': gpu['passed'], 'gpu_query': gpu,
        })
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
