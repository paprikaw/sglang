#!/usr/bin/env python3
"""One CPU traffic worker, supervised by its exact background GPU Guard owner."""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'idle_gpu_service'))
from node_runner_benchmark import lease_check, read, save
from loadgen import run_stage


def own_membership(config, run, node, plan=None):
    """Fail closed on lease loss, preemption, replaced endpoints, or cohort change."""
    receipt, own = lease_check(os.environ['EXPERIMENT_GUARD_LEASE_RECEIPT'], run, node)
    state = read(Path(config['fleet_runtime']) / 'state.json')
    attempt = state.get('attempts', {}).get(node)
    if not attempt or Path(attempt['run']).resolve() != run.resolve():
        raise InterruptedError('Worker is no longer the current managed deployment')
    if plan is None:
        return own
    if not attempt.get('status', {}).get('ready'):
        raise InterruptedError('Own deployment is not ready')
    registry = read(Path(receipt['registry_dir']) / 'registry.json')
    leases = {row['lease_id']: row for row in registry['leases']}
    if node not in plan['members'] or plan['members'][node]['run'] != str(run):
        raise InterruptedError('Worker is not a member of this phase')
    for member_node, member in plan['members'].items():
        current = state.get('attempts', {}).get(member_node, {})
        status = current.get('status', {})
        lease = leases.get(member['lease_id'], {})
        if (current.get('run') != member['run'] or not status.get('ready')
                or time.time() - status.get('timestamp', 0) > 20
                or lease.get('state') != 'active' or lease.get('nodes') != [member_node]
                or lease.get('preemption_requested')
                or time.time() - lease.get('heartbeat_at', 0) >= lease.get('ttl_s', 0)):
            raise InterruptedError('Benchmark cohort lease/readiness changed')
    return own


def ephemeral_requests(plan, node):
    """Length replay: do not persist synthetic payloads or token IDs."""
    nodes = sorted(plan['members'])
    result = []
    for index, row in enumerate(plan['requests']):
        if nodes[index % len(nodes)] != node:
            continue
        seed = int(hashlib.sha256(f"{plan['id']}:{index}:{row['sample_id']}".encode()).hexdigest()[:16], 16)
        generator = random.Random(seed)
        # All IDs are ordinary vocabulary entries well below GLM's special IDs.
        payload = dict(sample_id=row['sample_id'], output_tokens=row['output_tokens'],
                       offset_s=row['offset_s'], input_ids=[generator.randrange(256, 8192)
                                                          for _ in range(row['prompt_tokens'])])
        result.append(payload)
    return result


async def work(args):
    config, run = read(args.config), args.run.resolve()
    node, cancelled = args.node, asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        loop.add_signal_handler(sig, cancelled.set)
    root = Path(config['runtime_root'])
    seen = set()
    while not cancelled.is_set():
        try:
            own_membership(config, run, node)
        except InterruptedError:
            return
        plan_path = root / 'phase.json'
        try:
            plan = read(plan_path)
        except (OSError, ValueError):
            await asyncio.sleep(.2)
            continue
        if plan.get('id') in seen or node not in plan.get('members', {}) or not plan.get('active'):
            await asyncio.sleep(.2)
            continue
        stage = root / 'stages' / plan['id'] / node
        seen.add(plan['id'])
        try:
            own_membership(config, run, node, plan)
            if time.time() >= plan['deadline']:
                continue
            requests = ephemeral_requests(plan, node)
            deployment = read(run / 'deployment.json')
            endpoint = dict(node=node, url=deployment['endpoint'], model_id=deployment['deployment_id'])
            async def stop_check():
                if cancelled.is_set() or time.time() >= plan['deadline']:
                    return True
                try:
                    current = read(plan_path)
                    if current.get('id') != plan['id'] or not current.get('active'):
                        return True
                    own_membership(config, run, node, plan)
                    return False
                except (OSError, ValueError, RuntimeError, InterruptedError):
                    return True
            save(run / 'benchmark_progress.json', dict(phase=plan['id'], stage='running',
                 target_node_rps=plan['total_rps']/len(plan['members']), stage_dir=str(stage), timestamp=time.time()))
            summary = await run_stage([endpoint], requests, plan['total_rps']/len(plan['members']),
                stage, stop_check, duration_s=plan['dispatch_window_s'], start_at=plan['start_at'],
                preserve_offsets=True, **config.get('loadgen', {}))
            save(run / 'benchmark_progress.json', dict(phase=plan['id'], stage=summary['status'],
                 successful_requests=summary['successful_requests'], timestamp=time.time()))
        except (OSError, ValueError, RuntimeError, InterruptedError) as exc:
            save(stage / 'worker_failure.json', dict(timestamp=time.time(), type=type(exc).__name__,
                 message=str(exc)))
            save(run / 'benchmark_progress.json', dict(phase=plan['id'], stage='failed',
                 error=type(exc).__name__, timestamp=time.time()))
        await asyncio.sleep(.1)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--node', required=True)
    asyncio.run(work(parser.parse_args()))

if __name__ == '__main__':
    main()
