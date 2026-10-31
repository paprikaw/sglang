#!/usr/bin/env python3
"""Install the existing fleet's user services; never creates GPU bookings."""
import argparse
from pathlib import Path
import shutil
import socket
import subprocess

WORKSPACE = Path(__file__).resolve().parents[2]


def units(workspace=WORKSPACE, hostname=None):
    host = hostname or socket.gethostname()
    runtime = workspace / 'experiments/cluster_backfill_20261020'
    dashboard = workspace / 'experiments/idle_gpu_recovery_20261020/dashboard'
    shared = Path('/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/backfill/config.json')
    commands = {
        'gpu-resource-dashboard': ['/usr/bin/python3', '-u', str(dashboard / 'resource_overview.py'), '--config', str(dashboard / 'config.json')],
        'gpu-resource-monitor': ['/usr/bin/python3', '-u', str(runtime / 'monitor.py')],
        'gpu-idle-backfill': ['/usr/bin/python3', '-u', str(workspace / 'tools/idle_gpu_service/cluster_backfill.py'), '--config', str(shared)],
    }
    # Paths here are fixed local workspace paths, not shell commands. Reject
    # systemd expansion/line injection instead of invoking a shell.
    for command in commands.values():
        if any(any(c in arg for c in '\n\r%"\\') for arg in command):
            raise ValueError('Unsupported systemd command path')
    result = {}
    for name, command in commands.items():
        result[name + '.service'] = '\n'.join([
            '[Unit]', f'Description={name} (existing GPU fleet)',
            f'ConditionHost={host}', 'StartLimitIntervalSec=0', '',
            '[Service]', 'Type=simple',
            'ExecStart=' + ' '.join('"' + x + '"' for x in command),
            f'WorkingDirectory={workspace}', 'Restart=always', 'RestartSec=5',
            # A coordinator restart adopts existing verified managers. It must
            # not terminate its healthy background Guard children as a side effect.
            'KillMode=' + ('process' if name == 'gpu-idle-backfill' else 'control-group'),
            'TimeoutStopSec=20', 'UMask=0077',
            'Environment=PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1',
            'StandardOutput=journal', 'StandardError=journal', '',
            '[Install]', 'WantedBy=default.target', '',
        ])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install', action='store_true')
    args = p.parse_args()
    rendered = units()
    if not args.install:
        for name, value in rendered.items():
            print(name + '\n' + value)
        return
    target = Path.home() / '.config/systemd/user'
    target.mkdir(parents=True, exist_ok=True)
    for name, value in rendered.items():
        path = target / name
        if path.exists() and path.read_text() != value:
            shutil.copy2(path, path.with_suffix('.service.previous'))
        path.write_text(value)
    subprocess.run(['systemd-analyze', '--user', 'verify', *[str(target / n) for n in rendered]], check=True)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'enable', *rendered], check=True)
    print('Installed and enabled. Start services after retiring verified old standalone owners.')


if __name__ == '__main__':
    main()
