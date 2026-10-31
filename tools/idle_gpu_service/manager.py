#!/usr/bin/env python3
"""CPU-only service for preemptible Guard model deployments on existing bookings.

Never allocates Slurm resources or claims leases directly. Each independent node
attempt passes through ordinary Guard admission and authenticated model caches.
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid

MIN_DURATION_S = 604800
DEFAULT_REGISTRY = '/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/gpu_registry'


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix='.state-', dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path: Path) -> dict:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def process_start_ticks(pid: int) -> int | None:
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def same_process(pid: int, ticks: int | None) -> bool:
    if ticks is None or process_start_ticks(pid) != ticks:
        return False
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except (OSError, IndexError):
        return False


def remaining_seconds(value: str) -> float:
    value = value.strip()
    if value.upper() in {'UNLIMITED', 'NOT_SET', 'N/A', 'INVALID'}:
        raise ValueError(f'Unknown Slurm time remaining: {value}')
    days, clock = value.split('-', 1) if '-' in value else ('0', value)
    parts = [int(part) for part in clock.split(':')]
    if len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f'Invalid Slurm time remaining: {value}')
    if int(days) < 0 or min(hours, minutes, seconds) < 0 or minutes > 59 or seconds > 59:
        raise ValueError(f'Invalid Slurm time remaining: {value}')
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


@dataclasses.dataclass(frozen=True)
class Booking:
    node: str
    job_id: str
    end_at: float
    gpu_tres: str = ''


def discover_bookings(config: dict, now: float) -> dict[str, Booking]:
    command = [config.get('squeue', '/apps/slurm/latest/bin/squeue'),
               '--me', '-h', '-t', 'RUNNING', '-o', '%i|%N|%L|%b']
    result = subprocess.run(command, text=True, capture_output=True, timeout=20, check=True)
    bookings: dict[str, Booking] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        job_id, expression, remaining, gpu_tres = line.strip().split('|', 3)
        if not re.fullmatch(r'[0-9]+(?:_[0-9]+)?', job_id):
            raise ValueError(f'Unsupported job ID {job_id!r}')
        if 'gpu' not in gpu_tres.lower():
            continue
        end_at = now + remaining_seconds(remaining)
        if re.fullmatch(r'[A-Za-z0-9_.-]+', expression):
            nodes = [expression]
        else:
            expanded = subprocess.run(
                [config.get('scontrol', '/apps/slurm/latest/bin/scontrol'), 'show', 'hostnames', expression],
                text=True, capture_output=True, timeout=15, check=True)
            nodes = expanded.stdout.split()
        for node in nodes:
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', node):
                raise ValueError('Unsafe Slurm hostname')
            if node in bookings:
                raise ValueError(f'Multiple RUNNING bookings overlap on {node}')
            bookings[node] = Booking(node, job_id, end_at, gpu_tres)
    return bookings


def registry_blocks(path: Path, now: float) -> dict[str, str]:
    """Read atomic registry state; never mutate anybody's reservations."""
    state = read_json(path / 'registry.json')
    if state.get('schema_version') != 1 or not isinstance(state.get('leases'), list):
        raise ValueError('Malformed GPU registry; admission paused')
    blocked: dict[str, str] = {}
    for lease in state['leases']:
        if not isinstance(lease, dict) or not isinstance(lease.get('nodes'), list):
            raise ValueError('Malformed GPU registry lease')
        # Mirror Guard's expiry view without modifying the shared file. Every
        # candidate still passes Guard's atomic prune/claim before any launch.
        if lease.get('state') == 'active':
            if now - float(lease['heartbeat_at']) > float(lease['ttl_s']):
                continue
        elif lease.get('state') == 'quarantined':
            expires = float(lease.get('cleanup_grace_expires_at',
                                     float(lease.get('heartbeat_at', now)) + 90))
            if now >= expires:
                continue
        for node in lease['nodes']:
            blocked[node] = f"registry lease {lease.get('experiment_id', 'unknown')} ({lease.get('state', 'unknown')})"
    requests = state.get('preemption_requests', [])
    if not isinstance(requests, list):
        raise ValueError('Malformed GPU registry preemption requests')
    for request in requests:
        if float(request['expires_at']) <= now:
            continue
        nodes = request.get('target_nodes', request.get('nodes', []))
        if not nodes:
            nodes = [n for ns in request.get('target_allocation', request.get('allocation', {})).values() for n in ns]
        for node in nodes:
            blocked[node] = f"reserved by {request.get('requester_experiment_id', 'normal experiment')}"
    return blocked


def profile_reason(config: dict, booking: Booking) -> str | None:
    profiles = config.get('known_node_specs', {})
    if booking.node in profiles:
        profile = profiles[booking.node]
        if profile.get('disabled', False):
            return profile.get('reason', 'node profile disabled')
        return None
    allowed = config.get('allowed_gpu_types', [])
    if config.get('default_node_spec') and allowed:
        tokens = re.split(r'[:,=( )]+', booking.gpu_tres.lower())
        if any(str(kind).lower() in tokens for kind in allowed):
            return None
    return 'no compatible configured node profile'


def cache_ledger(config: dict) -> dict:
    root = Path(config.get('cpu_cache_registry_dir',
                str(Path(config.get('gpu_registry_dir', DEFAULT_REGISTRY)).parent / 'cpu_weight_daemons')))
    ledger = read_json(root / 'registry.json')
    if ledger.get('schema_version') != 1 or ledger.get('uid') != os.getuid() or not isinstance(ledger.get('entries'), list):
        raise ValueError('Unknown or malformed CPU daemon ledger')
    if any(not isinstance(row, dict) or not isinstance(row.get('node'), str) for row in ledger['entries']):
        raise ValueError('Malformed CPU daemon ledger entry')
    names = [row['node'] for row in ledger['entries']]
    if len(names) != len(set(names)):
        raise ValueError('Ambiguous duplicate CPU daemon node records')
    return ledger


def cached_weight_candidate(config: dict, booking: Booking, ledger: dict) -> tuple[str | None, dict | None]:
    """Advisory recorded-daemon selection; Guard validates the exact live model.

    Ledger shard keys are opaque and node local. An all-ready snapshot is a
    candidate, never proof of the requested model. Guard on_miss=fail is required
    and is the final authority; no daemon is created, resized, rebuilt or retired.
    """
    profile = config.get('known_node_specs', {}).get(booking.node, config.get('default_node_spec', {}))
    if 'cache_spec_path' in profile:
        desired = read_json(Path(profile['cache_spec_path']))
    else:
        desired = profile.get('cache_spec_template', {})
    if (desired.get('engine_root') != config.get('engine_root') or
            desired.get('source_commit') != config.get('engine_commit') or
            desired.get('model_path') != config.get('model_path')):
        return 'cache-only profile differs from the pinned serving model/source', None
    rows = [row for row in ledger['entries'] if row['node'] == booking.node]
    if len(rows) != 1:
        return 'cache-only: no recorded CPU weight daemon on this node', None
    row = rows[0]
    if row.get('state') != 'active':
        return 'cache-only: recorded CPU daemon is not active', None
    expected = {'source_commit': desired.get('source_commit'),
                'cache_scheme': desired.get('cache_scheme'),
                'expected_job_id': booking.job_id, 'uid': os.getuid()}
    mismatch = [key for key, value in expected.items() if value is None or row.get(key) != value]
    if mismatch:
        return 'cache-only: incompatible recorded daemon ' + ', '.join(mismatch), None
    # Preserve the existing daemon capacity and socket. The runner adopts these
    # admitted values rather than changing them to match its default template.
    if (row.get('cache_scheme') != 'canonical_full_weight' or
            type(row.get('max_bytes')) is not int or row['max_bytes'] <= 0 or
            type(row.get('headroom_bytes')) is not int or row['headroom_bytes'] < 32 * 2**30 or
            not isinstance(row.get('socket_path'), str) or not Path(row['socket_path']).is_absolute()):
        return 'cache-only: invalid recorded daemon capacity/socket', None
    status = row.get('status')
    manifest = row.get('manifest')
    if not isinstance(status, dict) or not isinstance(manifest, dict):
        return 'cache-only: no usable recorded daemon readiness snapshot', None
    expected_status = {'source_commit': row['source_commit'], 'max_bytes': row['max_bytes'],
                       'headroom_bytes': row['headroom_bytes'], 'uid': os.getuid(),
                       'pid': manifest.get('pid'), 'start_ticks': manifest.get('process_start_ticks')}
    if (status.get('ok') is not True or manifest.get('state') != 'running' or
            manifest.get('node') != booking.node or manifest.get('cache_id') != row.get('daemon_id') or
            any(value is None or status.get(key) != value for key, value in expected_status.items()) or
            'canonical_groups_v1' not in status.get('features', [])):
        return 'cache-only: recorded daemon readiness identity mismatch', None
    entries = status.get('entries')
    if not isinstance(entries, list) or not entries:
        return 'cache-only: recorded daemon contains no ready weights', None
    if any(not isinstance(entry, dict) or entry.get('state') != 'ready' or
           entry.get('evict_pending') or type(entry.get('size')) is not int or entry['size'] <= 0
           for entry in entries):
        return 'cache-only: recorded weights are building, incomplete or evicting', None
    if (status.get('reserved_bytes') != 0 or type(status.get('used_bytes')) is not int or
            status['used_bytes'] <= 0 or status.get('current_bytes') != status['used_bytes'] or
            status['used_bytes'] > row['max_bytes']):
        return 'cache-only: recorded weight accounting is incomplete', None
    evidence = {'selection': 'recorded_ready_candidate_requires_guard_live_model_hit',
                'daemon_id': row['daemon_id'], 'recorded_at': row.get('checked_at'),
                'max_bytes': row['max_bytes'], 'headroom_bytes': row['headroom_bytes'],
                'socket_path': row['socket_path'], 'ready_entries': len(entries),
                'model_cache_hit_proven': False}
    return None, evidence


@dataclasses.dataclass
class Attempt:
    booking: Booking
    run: Path
    process: subprocess.Popen
    phase: str
    started_at: float
    start_ticks: int | None
    log: object
    prepare_output: object | None = None
    stop_reason: str | None = None
    stop_requested_at: float | None = None


def benchmark_failure_reason(config: dict, attempt: Attempt, now: float) -> str | None:
    """Recover an owned idle deployment whose load generator has failed.

    A healthy serving process can outlive its benchmark child. The immutable,
    private attempt directory ties the worker receipt to this deployment; a
    terminal failure there remains actionable even when it happened hours ago.
    Progress is only a fallback and must be fresh, since it is a live snapshot.
    No remote process is inspected or signalled here: normal Guard teardown and
    admission still govern cleanup and any retry.
    """
    benchmark = config.get('benchmark')
    if (attempt.phase != 'guard' or attempt.stop_requested_at is not None
            or not isinstance(benchmark, dict) or not benchmark
            or benchmark.get('enabled') is False):
        return None

    def valid_time(value) -> bool:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and attempt.started_at <= value <= now + 5)

    try:
        worker = read_json(attempt.run / 'benchmark_worker.json')
    except (OSError, ValueError):
        worker = {}
    if (worker.get('state') == 'failed'
            and worker.get('node', attempt.booking.node) == attempt.booking.node
            and any(valid_time(worker.get(field)) for field in ('started_at', 'timestamp'))):
        code = worker.get('exit_code')
        suffix = f' (exit code {code})' if type(code) is int else ''
        return 'benchmark worker failed' + suffix

    try:
        progress = read_json(attempt.run / 'task/progress.json')
    except (OSError, ValueError):
        return None
    timestamp = progress.get('timestamp')
    freshness_s = max(30, 3 * float(config.get('poll_s', 10)))
    if (progress.get('benchmark_worker_state') == 'failed' and valid_time(timestamp)
            and now - timestamp <= freshness_s):
        return 'benchmark worker failed (fresh controller progress)'
    return None


class Launcher:
    def __init__(self, config: dict, config_path: Path):
        self.config, self.config_path = config, config_path

    def runner(self, subcommand: str, booking: Booking, run: Path) -> list[str]:
        return [self.config.get('python', sys.executable), self.config['node_runner'], subcommand,
                '--config', str(self.config_path), '--node', booking.node,
                '--job-id', booking.job_id, '--run', str(run)]

    def prepare(self, booking: Booking, run: Path, now: float) -> Attempt:
        log_path = Path(os.devnull) if self.config.get('artifact_retention') == 'none' else run / 'manager-child.log'
        log = log_path.open('ab', buffering=0)
        output = (run / 'prepare.json.out').open('w+b', buffering=0)
        try:
            process = subprocess.Popen(self.runner('prepare', booking, run), stdout=output,
                                       stderr=log, start_new_session=True)
        except BaseException:
            log.close()
            output.close()
            raise
        return Attempt(booking, run, process, 'preparing', now,
                       process_start_ticks(process.pid), log, output)

    def launch(self, attempt: Attempt, now: float) -> Attempt:
        attempt.prepare_output.seek(0)
        output = attempt.prepare_output.read(1024 * 1024 + 1)
        if len(output) > 1024 * 1024:
            raise ValueError('prepare output exceeded 1 MiB')
        result = json.loads(output)
        spec_path = Path(result['spec_path'])
        if not spec_path.is_absolute() or not spec_path.resolve().is_relative_to(attempt.run.resolve()):
            raise ValueError('prepare returned a spec outside the private attempt directory')
        spec = read_json(spec_path)
        if spec.get('resource_priority') != 'background' or spec.get('preemptible') is not True:
            raise ValueError('Idle deployment spec must be preemptible background priority')
        tasks = spec.get('tasks', [])
        if len(tasks) != 1 or tasks[0].get('nodes') != [attempt.booking.node] or any(
                name in tasks[0] for name in ('node_count', 'min_nodes', 'max_nodes')):
            raise ValueError('Idle deployment spec must declare exactly this booked node')
        if Path(spec.get('gpu_registry_dir', DEFAULT_REGISTRY)).resolve() != Path(self.config.get('gpu_registry_dir', DEFAULT_REGISTRY)).resolve():
            raise ValueError('Idle deployment must use the shared GPU registry')
        if float(spec.get('gpu_lease_ttl_s', 900)) < 15:
            raise ValueError('Guard lease TTL must be at least 15 seconds')
        if spec.get('model_cache', {}).get('policy') != 'required':
            raise ValueError('Idle model deployment requires the ordinary model-cache contract')
        if self.config.get('cached_weights_only') and spec['model_cache'].get('on_miss') != 'fail':
            raise ValueError('Cache-only idle deployment must fail on an exact live model cache miss')
        command = [self.config.get('python', sys.executable), self.config['guard_script'], 'run',
                   '--spec', str(spec_path), '--exit-after-run', '--']
        command.extend(self.runner('controller', attempt.booking, attempt.run))
        process = subprocess.Popen(command, stdout=attempt.log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        attempt.prepare_output.close()
        attempt.prepare_output = None
        attempt.process, attempt.phase = process, 'guard'
        attempt.started_at, attempt.start_ticks = now, process_start_ticks(process.pid)
        return attempt

    def stop(self, attempt: Attempt, reason: str, now: float) -> None:
        if attempt.stop_requested_at is not None:
            return
        attempt.stop_requested_at, attempt.stop_reason = now, reason
        if same_process(attempt.process.pid, attempt.start_ticks):
            # Guard signals its controller and records cleanup itself.
            attempt.process.send_signal(signal.SIGTERM)

    def kill(self, attempt: Attempt) -> None:
        if same_process(attempt.process.pid, attempt.start_ticks):
            os.killpg(attempt.process.pid, signal.SIGKILL)

    def close(self, attempt: Attempt) -> None:
        if attempt.prepare_output:
            attempt.prepare_output.close()
        attempt.log.close()


class Reconciler:
    """One independently owned attempt per node, with bounded retry backoff."""
    def __init__(self, config: dict, root: Path, launcher: Launcher, started_at: float):
        self.config, self.root, self.launcher = config, root, launcher
        self.started_at = started_at
        self.deadline = started_at + float(config.get('duration_s', MIN_DURATION_S))
        self.service_id = uuid.uuid4().hex
        self.attempts: dict[str, Attempt] = {}
        self.retries: dict[str, dict] = {}
        self.nodes: dict[str, dict] = {}
        self.stop_reason: str | None = None
        self.discovery_error: str | None = None
        self.cache_selection_error: str | None = None

    def request_stop(self, reason: str) -> None:
        self.stop_reason = reason

    def retry(self, node: str, reason: str, now: float) -> None:
        count = self.retries.get(node, {}).get('failures', 0) + 1
        delay = min(float(self.config.get('max_backoff_s', 600)),
                    float(self.config.get('retry_backoff_s', 30)) * 2 ** min(count - 1, 10))
        self.retries[node] = {'failures': count, 'retry_at': now + delay, 'reason': reason}

    def reconcile(self, bookings: dict[str, Booking], blocked: dict[str, str], now: float,
                  discovery_error: str | None = None) -> None:
        self.discovery_error = discovery_error
        if now >= self.deadline:
            self.request_stop('service deadline reached')
        margin = float(self.config.get('booking_margin_s', 60))
        for node, attempt in list(self.attempts.items()):
            booking = bookings.get(node)
            reason = self.stop_reason
            if discovery_error:
                reason = 'booking/registry discovery unavailable'
            elif booking is None or booking.job_id != attempt.booking.job_id:
                reason = reason or 'booking ended or changed'
            elif booking.end_at - now <= margin:
                reason = reason or 'booking expiry margin reached'
            elif attempt.phase == 'preparing' and node in blocked:
                reason = reason or blocked[node]
            elif attempt.phase == 'preparing' and now - attempt.started_at >= float(self.config.get('prepare_timeout_s', 60)):
                reason = reason or 'local profile preparation timed out'
            if reason is None:
                reason = benchmark_failure_reason(self.config, attempt, now)
            if reason:
                self.launcher.stop(attempt, reason, now)
            code = attempt.process.poll()
            if code is None:
                if attempt.stop_requested_at is not None and now - attempt.stop_requested_at > float(self.config.get('stop_grace_s', 150)):
                    self.launcher.kill(attempt)
                continue
            if attempt.phase == 'preparing' and not reason and attempt.stop_requested_at is None and code == 0:
                try:
                    self.launcher.launch(attempt, now)
                    continue
                except Exception as error:
                    reason = f'spec preparation rejected: {error}'
            self.retry(node, attempt.stop_reason or reason or f'{attempt.phase} exited {code}', now)
            self.launcher.close(attempt)
            del self.attempts[node]
        self.nodes = {}
        ledger = None
        self.cache_selection_error = None
        if self.config.get('cached_weights_only'):
            try:
                ledger = cache_ledger(self.config)
            except Exception as error:
                self.cache_selection_error = f'cache-only CPU ledger unavailable: {type(error).__name__}: {error}'
        for node, booking in sorted(bookings.items()):
            reason = (self.stop_reason or discovery_error or profile_reason(self.config, booking)
                      or ('booking expiry margin reached' if booking.end_at - now <= margin else None))
            if node in self.attempts:
                attempt = self.attempts[node]
                self.nodes[node] = {'state': 'stopping' if attempt.stop_reason else attempt.phase,
                                    'reason': attempt.stop_reason, 'run': str(attempt.run), 'job_id': booking.job_id}
                continue
            reason = reason or blocked.get(node)
            cache_evidence = None
            if not reason and self.config.get('cached_weights_only'):
                reason = self.cache_selection_error
                if not reason:
                    try:
                        reason, cache_evidence = cached_weight_candidate(self.config, booking, ledger)
                    except Exception as error:
                        reason = f'cache-only selection failed: {type(error).__name__}: {error}'
            retry = self.retries.get(node, {})
            if not reason and retry.get('retry_at', 0) > now:
                reason = f"retry backoff: {retry['reason']}"
            self.nodes[node] = {**retry, 'state': 'skipped' if reason else 'starting', 'reason': reason,
                                'job_id': booking.job_id, 'booking_end_at': booking.end_at, 'cache_evidence': cache_evidence}
            if reason:
                continue
            run = self.root / 'runs' / f'{int(now)}-{node}-{uuid.uuid4().hex[:10]}'
            run.mkdir(parents=True, mode=0o700)
            atomic_json(run / 'attempt.json', {'node': node, 'job_id': booking.job_id,
                        'service_id': self.service_id, 'deadline': self.deadline,
                        'booking_end_at': booking.end_at, 'stop_at': min(self.deadline, booking.end_at - margin),
                        'cache_selection_evidence': cache_evidence})
            try:
                self.attempts[node] = self.launcher.prepare(booking, run, now)
                self.nodes[node].update(state='preparing', run=str(run))
            except Exception as error:
                self.retry(node, f'prepare spawn failed: {error}', now)
                self.nodes[node].update(state='failed', reason=str(error))

    def snapshot(self, now: float) -> dict:
        attempts = {}
        for node, attempt in self.attempts.items():
            status = None
            for relative in ('remote_status.json', 'task/progress.json'):
                try:
                    status = read_json(attempt.run / relative)
                    break
                except (OSError, ValueError):
                    pass
            attempts[node] = {'pid': attempt.process.pid, 'start_ticks': attempt.start_ticks,
                              'phase': attempt.phase, 'run': str(attempt.run), 'job_id': attempt.booking.job_id,
                              'stop_reason': attempt.stop_reason, 'status': status}
        return {'schema_version': 1, 'service_id': self.service_id,
                'manager_pid': os.getpid(), 'manager_start_ticks': process_start_ticks(os.getpid()),
                'manager_host': socket.gethostname().split('.')[0],
                'started_at': self.started_at, 'deadline': self.deadline, 'updated_at': now,
                'state': 'stopping' if self.stop_reason and attempts else 'stopped' if self.stop_reason else 'running',
                'stop_reason': self.stop_reason, 'discovery_error': self.discovery_error,
                'cached_weights_only': bool(self.config.get('cached_weights_only')),
                'cache_selection_error': self.cache_selection_error,
                'nodes': self.nodes, 'attempts': attempts, 'retries': self.retries}


def load_config(path: Path) -> dict:
    if not path.is_absolute():
        raise ValueError('--config must be absolute')
    config = read_json(path)
    for name in ('runtime_root', 'guard_script', 'node_runner'):
        if not Path(config[name]).is_absolute():
            raise ValueError(f'{name} must be absolute')
    if not math.isfinite(float(config.get('duration_s', MIN_DURATION_S))) or float(config.get('duration_s', MIN_DURATION_S)) < MIN_DURATION_S:
        raise ValueError('duration_s must be at least 604800 seconds (seven days)')
    for name, default in (('poll_s', 10), ('booking_margin_s', 60), ('stop_grace_s', 150),
                          ('prepare_timeout_s', 60), ('retry_backoff_s', 30), ('max_backoff_s', 600)):
        if not math.isfinite(float(config.get(name, default))) or float(config.get(name, default)) <= 0:
            raise ValueError(f'{name} must be positive')
    if float(config.get('poll_s', 10)) > 30:
        raise ValueError('poll_s must not exceed 30 seconds')
    if not isinstance(config.get('known_node_specs', {}), dict) or any(
            not isinstance(profile, dict) for profile in config.get('known_node_specs', {}).values()):
        raise ValueError('known_node_specs must map node names to profile objects')
    if not isinstance(config.get('cached_weights_only', False), bool):
        raise ValueError('cached_weights_only must be boolean')
    if 'cpu_cache_registry_dir' in config and not Path(config['cpu_cache_registry_dir']).is_absolute():
        raise ValueError('cpu_cache_registry_dir must be absolute')
    return config


def secure_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink() or root.stat().st_uid != os.getuid():
        raise ValueError('runtime_root must be an owned non-symlink directory')
    os.chmod(root, 0o700)


def stop_previous(state: dict, timeout_s: float) -> None:
    """Recover only children whose immutable PID/start identity was saved."""
    if state.get('manager_host') != socket.gethostname().split('.')[0]:
        if state.get('attempts'):
            raise RuntimeError('Previous children belong to another host; refusing local PID signals')
        return
    victims = []
    for attempt in state.get('attempts', {}).values():
        pid, ticks = int(attempt['pid']), attempt.get('start_ticks')
        if same_process(pid, ticks):
            if os.getpgid(pid) != pid:
                raise RuntimeError('Previous child process-group identity changed')
            os.kill(pid, signal.SIGTERM)
            victims.append((pid, ticks))
    deadline = time.monotonic() + timeout_s
    while any(same_process(pid, ticks) for pid, ticks in victims):
        if time.monotonic() >= deadline:
            for pid, ticks in victims:
                if same_process(pid, ticks):
                    os.killpg(pid, signal.SIGKILL)
            break
        time.sleep(.2)


def run(config: dict, path: Path) -> int:
    root = Path(config['runtime_root'])
    secure_root(root)
    lock = (root / 'manager.lock').open('a+')
    os.chmod(root / 'manager.lock', 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('An idle GPU manager already owns this runtime_root')
    state_path = root / 'state.json'
    if state_path.exists():
        stop_previous(read_json(state_path), float(config.get('stop_grace_s', 150)))
    reconciler = Reconciler(config, root, Launcher(config, path), time.time())
    def interrupted(signum, _frame):
        reconciler.request_stop(f'signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    atomic_json(root / 'profile.json', config)
    try:
        while True:
            now = time.time()
            bookings, blocked, error = {}, {}, None
            if not reconciler.stop_reason and now < reconciler.deadline:
                try:
                    bookings = discover_bookings(config, now)
                    blocked = registry_blocks(Path(config.get('gpu_registry_dir', DEFAULT_REGISTRY)), now)
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
            reconciler.reconcile(bookings, blocked, now, error)
            atomic_json(state_path, reconciler.snapshot(time.time()))
            if reconciler.stop_reason and not reconciler.attempts:
                return 0
            time.sleep(min(float(config.get('poll_s', 10)), 1 if reconciler.stop_reason else 30))
    finally:
        reconciler.request_stop('manager exiting')
        for attempt in reconciler.attempts.values():
            reconciler.launcher.stop(attempt, 'manager exiting', time.time())
        while reconciler.attempts:
            reconciler.reconcile({}, {}, time.time())
            atomic_json(state_path, reconciler.snapshot(time.time()))
            time.sleep(.2)
        lock.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'status', 'stop'))
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    root = Path(config['runtime_root'])
    if args.command == 'run':
        return run(config, args.config)
    state = read_json(root / 'state.json')
    pid, ticks = int(state['manager_pid']), state.get('manager_start_ticks')
    state['manager_alive'] = state.get('manager_host') == socket.gethostname().split('.')[0] and same_process(pid, ticks)
    if args.command == 'stop':
        if state['manager_alive']:
            os.kill(pid, signal.SIGTERM)
            state['stop_requested'] = True
        else:
            with (root / 'manager.lock').open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                stop_previous(state, float(config.get('stop_grace_s', 150)))
            state['state'] = 'stopped'
            state['stop_reason'] = 'stop requested after manager exit'
            state['attempts'] = {}
            atomic_json(root / 'state.json', state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
