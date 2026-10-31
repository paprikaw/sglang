#!/usr/bin/env python3
"""CPU-only, read-only Guard/worker/GPU telemetry for idle prefill replicas.

This process sends no inference, creates no leases, and never changes bookings,
models, caches, or process lifetimes. Only its telemetry artifacts are written.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time

GPU_COMMAND = ('nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,'
               'memory.total,power.draw --format=csv,noheader,nounits')
GPU_KEYS = ('utilization_gpu_pct', 'memory_used_mib', 'memory_total_mib', 'power_draw_w')


def read(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def number(value):
    try:
        if isinstance(value, bool):
            return None
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def fresh(timestamp, now, max_age):
    value = number(timestamp)
    return value is not None and -5 <= now-value <= max_age


def parse_gpu_csv(text):
    """Parse documented nvidia-smi CSV; unavailable numbers remain null."""
    result, seen = [], set()
    for fields in csv.reader(io.StringIO(text)):
        if not fields or not any(value.strip() for value in fields):
            continue
        if len(fields) != 6:
            raise ValueError('invalid_gpu_csv_column_count')
        try:
            index = int(fields[0].strip())
        except ValueError:
            raise ValueError('invalid_gpu_index') from None
        if index < 0 or index in seen or not fields[1].strip():
            raise ValueError('invalid_or_duplicate_gpu_identity')
        seen.add(index)
        row = dict(index=index, name=fields[1].strip())
        row.update({key: number(value.strip()) for key, value in zip(GPU_KEYS, fields[2:])})
        if row['utilization_gpu_pct'] is not None and row['utilization_gpu_pct'] > 100:
            row['utilization_gpu_pct'] = None
        row['unavailable_fields'] = [key for key in GPU_KEYS if row[key] is None]
        result.append(row)
    if not result:
        raise ValueError('empty_gpu_csv')
    return result


def authorized_nodes(config, fleet, registry, deployments, now):
    """Pure admission check; return one decision for each fixed configured node.

    No private receipt/token is copied into telemetry. The exact current healthy
    deployment must match an active single-node background Guard lease. A normal
    preemption request denies sampling immediately, before background cleanup.
    """
    decisions = {}
    for node in config['nodes']:
        reason = None
        attempt = fleet.get('attempts', {}).get(node, {})
        status = attempt.get('status') or {}
        deployment = deployments.get(node) or {}
        if fleet.get('state') != 'running':
            reason = 'manager_not_running'
        elif not fresh(fleet.get('updated_at'), now, 30):
            reason = 'manager_state_stale'
        elif not attempt or not status.get('ready'):
            reason = 'instance_not_ready'
        elif not fresh(status.get('timestamp'), now, 20):
            reason = 'instance_status_stale'
        elif deployment.get('status') != 'healthy' or deployment.get('node') != node:
            reason = 'deployment_not_healthy'
        elif not deployment.get('deployment_id') or not attempt.get('run'):
            reason = 'deployment_identity_missing'
        else:
            candidates = [lease for lease in registry.get('leases', [])
                          if lease.get('nodes') == [node] and lease.get('state') == 'active']
            if len(candidates) != 1:
                reason = 'exact_active_lease_missing_or_ambiguous'
            else:
                lease = candidates[0]
                if lease.get('experiment_id') != deployment['deployment_id']:
                    reason = 'lease_deployment_mismatch'
                elif lease.get('resource_priority') != 'background':
                    reason = 'lease_not_background'
                elif lease.get('preemptible') is not True:
                    reason = 'lease_not_preemptible'
                elif lease.get('preemption_requested'):
                    reason = 'foreground_preemption_requested'
                elif (number(lease.get('ttl_s')) is None or number(lease['ttl_s']) <= 0
                      or not fresh(lease.get('heartbeat_at'), now, number(lease['ttl_s']))
                      or now-number(lease['heartbeat_at']) >= number(lease['ttl_s'])):
                    reason = 'lease_heartbeat_expired'
        decisions[node] = dict(authorized=reason is None, reason=reason,
            run=attempt.get('run'), deployment_id=deployment.get('deployment_id'))
    return decisions


def query_gpu(node):
    sample = dict(sampled_at=time.time(), gpus=None, error=None)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', node):
        sample['error'] = 'invalid_node_name'
        return sample
    try:
        completed = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=3',
            '-o', 'ServerAliveInterval=2', '-o', 'ServerAliveCountMax=1', node, GPU_COMMAND],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20, check=False)
        if completed.returncode:
            sample['error'] = 'ssh_exit_' + str(completed.returncode)
        else:
            sample['gpus'] = parse_gpu_csv(completed.stdout)
            if any(row['unavailable_fields'] for row in sample['gpus']):
                sample['error'] = 'gpu_fields_unavailable'
    except subprocess.TimeoutExpired:
        sample['error'] = 'ssh_timeout'
    except (OSError, UnicodeError):
        sample['error'] = 'ssh_unavailable'
    except ValueError:
        sample['error'] = 'invalid_gpu_csv'
    sample['finished_at'] = time.time()
    return sample


def safe_read(path):
    try:
        value = read(path)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def current_decisions(config, now):
    fleet = safe_read(Path(config['fleet_runtime'])/'state.json')
    registry = safe_read(Path(config['gpu_registry_dir'])/'registry.json')
    deployments = {}
    for node in config['nodes']:
        run = fleet.get('attempts', {}).get(node, {}).get('run')
        deployments[node] = safe_read(Path(run)/'deployment.json') if run else {}
    return fleet, authorized_nodes(config, fleet, registry, deployments, now)


def manager_is_live(fleet):
    if fleet.get('state') not in ('running', 'stopping'):
        return False
    if fleet.get('manager_host') != socket.gethostname().split('.')[0]:
        return fresh(fleet.get('updated_at'), time.time(), 30)
    try:
        # Read process identity without sending signals or publishing its PID.
        stat = Path('/proc')/str(int(fleet['manager_pid']))/'stat'
        fields = stat.read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and int(fields[19]) == int(fleet['manager_start_ticks'])
    except (OSError, ValueError, KeyError, IndexError):
        return False


def snapshot(config, executor):
    now = time.time()
    fleet, decisions = current_decisions(config, now)
    futures = {node: executor.submit(query_gpu, node) for node, decision in decisions.items()
               if decision['authorized']}
    sampled = {node: future.result() for node, future in futures.items()}
    observed_at = time.time()
    _, latest_decisions = current_decisions(config, observed_at)
    root = Path(config['runtime_root'])
    nodes, all_utils, rates = {}, [], []
    for node, decision in decisions.items():
        worker = safe_read(root/'nodes'/(node+'.json'))
        worker_stale = (worker.get('run') != decision['run'] or
                        not fresh(worker.get('timestamp'), observed_at, 15))
        sample = sampled.get(node, dict(sampled_at=None, gpus=None, error=decision['reason']))
        stale = (not decision['authorized'] or latest_decisions[node] != decision
                 or sample.get('gpus') is None
                 or not fresh(sample.get('finished_at'), time.time(), 10))
        if latest_decisions[node] != decision:
            sample = dict(sample, gpus=None, error='authorization_changed_during_sample')
        utils = [row['utilization_gpu_pct'] for row in sample.get('gpus') or []
                 if row['utilization_gpu_pct'] is not None]
        worker_current = (not worker_stale and decision['authorized']
                          and latest_decisions[node] == decision)
        rate = number(worker.get('input_tokens_per_s')) if worker_current else None
        cumulative = {key: number((worker.get('cumulative') or {}).get(key))
            for key in ('dispatched_requests', 'completed_requests', 'successful_requests', 'input_tokens')}
        nodes[node] = dict(authorized=decision['authorized'], reason=decision['reason'],
            deployment_id=decision['deployment_id'], worker_status=worker.get('status'),
            worker_updated_at=worker.get('timestamp'), worker_stale=worker_stale,
            cumulative=cumulative, observed_input_tokens_per_s=rate,
            completed_rps=number(worker.get('completed_rps')) if not worker_stale else None,
            concurrent_requests_per_instance=worker.get('concurrent_requests_per_instance'),
            gpu_sampled_at=sample.get('sampled_at'), gpu_stale=stale,
            gpu_error=sample.get('error'), gpus=sample.get('gpus'),
            gpu_utilization_avg_pct=sum(utils)/len(utils) if utils else None,
            valid_gpu_utilization_readings=len(utils))
        if not stale:
            all_utils.extend(utils)
        if rate is not None:
            rates.append(rate)
    return fleet, dict(schema_version=1, timestamp=time.time(), mode=config.get('mode', 'prefill'),
        state='running' if manager_is_live(fleet) else 'manager_stopped',
        queue_policy='observe_only', target_rps=None,
        total_observed_input_tokens_per_s=sum(rates) if rates else None,
        gpu_utilization_avg_pct=sum(all_utils)/len(all_utils) if all_utils else None,
        valid_gpu_utilization_readings=len(all_utils), nodes=nodes,
        throughput_interpretation='Sum of fresh current worker input_tokens_per_s, each averaged since its current deployment started.',
        gpu_utilization_interpretation='Physical nvidia-smi samples on currently authorized background instances; unavailable or stale values are null.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if not args.config.is_absolute():
        parser.error('--config must be an absolute path')
    config = read(args.config)
    root = Path(config['runtime_root'])
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    deadline = time.time() + config.get('duration_s', 604800)
    with (root/'monitor.lock').open('a') as lock, ThreadPoolExecutor(max_workers=max(1,len(config['nodes']))) as executor:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fd = os.open(root/'gpu_utilization.jsonl', os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'a', buffering=1) as history:
            while time.time() < deadline:
                started = time.monotonic()
                fleet, result = snapshot(config, executor)
                if fleet.get('deadline'):
                    deadline = min(deadline, fleet['deadline'])
                result['deadline'] = deadline
                save(root/'state.json', result)
                save(root/'gpu_latest.json', result)
                history.write(json.dumps(result, allow_nan=False)+'\n')
                history.flush()
                os.fsync(history.fileno())
                if not manager_is_live(fleet):
                    break
                time.sleep(max(0, config.get('monitor_poll_s', 5) - (time.monotonic()-started)))


if __name__ == '__main__':
    main()
