#!/usr/bin/env python3
"""Guard-owned continuous prefill load on exactly one instance."""
from __future__ import annotations
import argparse
import asyncio
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import uuid
import aiohttp
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'idle_gpu_service'))
from node_runner_benchmark import read, save
from worker import own_membership
from loadgen import percentiles
from prefill_client import send_prefill
from prefill_trace import PrefillTrace


def current_instance(config, run, node):
    """Only this private exact-node receipt may authorize this endpoint."""
    own_membership(config, run, node)
    fleet = read(Path(config['fleet_runtime']) / 'state.json')
    attempt = fleet['attempts'][node]
    status = attempt.get('status', {})
    if not status.get('ready') or time.time() - status.get('timestamp', 0) > 20:
        return None
    deployment = read(run / 'deployment.json')
    if deployment.get('node') != node or deployment.get('status') != 'healthy':
        return None
    return dict(node=node, url=deployment['endpoint'], model_id=deployment['deployment_id'])


def synthetic_request(row, deployment_id):
    seed = hashlib.sha256(f"{deployment_id}:{row['sample_id']}:{row.get('source_cycle', 0)}".encode()).digest()
    rng = random.Random(int.from_bytes(seed[:16], 'big'))
    return dict(row, input_ids=[rng.randrange(256, 8192) for _ in range(row['prompt_tokens'])], output_tokens=1)


def batch_summary(rows, started, finished, offered, concurrency=1):
    good = [row for row in rows if row['status'] == 'ok']
    elapsed = max(.001, finished - started)
    return dict(schema_version=1, mode='continuous_prefill',
        concurrent_requests_per_instance=concurrency,
        queue_policy='observe_only', target_rps=None, started_at=started, finished_at=finished,
        offered_requests=offered, completed_requests=len(rows), successful_requests=len(good),
        valid=len(good) == offered, completed_rps=len(good) / elapsed,
        input_tokens=sum(row.get('prompt_tokens') or 0 for row in good),
        input_tokens_per_s=sum(row.get('prompt_tokens') or 0 for row in good) / elapsed,
        latency_s={key: percentiles([row[key+'_s'] for row in good if row.get(key+'_s') is not None])
                   for key in ('ttft', 'e2e', 'tpot', 'server_queue')},
        tpot_note='Not applicable: exactly one output token per prefill request.')


async def run_prefill_batch(session, endpoint, requests, stop_check, *, concurrency=1,
                            request_timeout_s=600, on_start=None, on_row=None):
    """Fixed completion-driven lanes sharing one unique request index.

    No RPS timer or queue admission policy participates. Once a failure or stop
    is observed, no lane allocates another request; active calls return partial
    cancelled rows through send_prefill's 50 ms stop watcher.
    """
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError('concurrent_requests_per_instance must be a positive integer')
    abort = asyncio.Event()
    next_index, rows = 0, []

    async def batch_stopped():
        if abort.is_set():
            return True
        try:
            should_stop = await stop_check()
        except asyncio.CancelledError:
            raise
        except Exception:
            should_stop = True
        if should_stop:
            abort.set()
        return bool(should_stop)

    async def lane():
        nonlocal next_index
        while not await batch_stopped():
            # No await between checking abort and reserving the unique index.
            if abort.is_set() or next_index >= len(requests):
                return
            metadata = requests[next_index]
            next_index += 1
            request = synthetic_request(metadata, endpoint['model_id'])
            if on_start is not None:
                on_start(metadata)
            row = await send_prefill(session, endpoint, request, batch_stopped,
                                      request_timeout_s=request_timeout_s)
            row.update({key: metadata.get(key) for key in
                        ('source_index', 'source_cycle', 'source_model', 'original_output_tokens')})
            if row['status'] != 'ok':
                abort.set()
            rows.append(row)
            if on_row is not None:
                on_row(row)
            if abort.is_set():
                return

    tasks = [asyncio.create_task(lane()) for _ in range(min(concurrency, len(requests)))]
    try:
        await asyncio.gather(*tasks)
    finally:
        # External cancellation or persistence failure also stops every lane.
        abort.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    return rows


async def work(args):
    config, run, node = read(args.config), args.run.resolve(), args.node
    if node not in config['nodes']:
        raise ValueError('Node is outside the configured prefill partition')
    concurrency = config.get('concurrent_requests_per_instance', 1)
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError('concurrent_requests_per_instance must be a positive integer')
    root = Path(config['runtime_root'])
    cancelled = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        loop.add_signal_handler(sig, cancelled.set)
    attempt_started = time.time()
    stop_at = min(read(run / 'attempt.json')['stop_at'], attempt_started + config.get('duration_s', 604800))
    cumulative = dict(dispatched_requests=0, completed_requests=0, successful_requests=0, input_tokens=0)
    node_state = root / 'nodes' / (node + '.json')
    stage, rows, offered, dispatched, stage_started, last_publish = None, [], 0, 0, None, 0.
    failure, stream, active_inflight, peak_inflight = None, None, 0, 0

    def publish(status, force=False, inflight=None):
        nonlocal last_publish
        now = time.time()
        if not force and now-last_publish < .5:
            return
        last_publish = now
        elapsed = max(.001, now-attempt_started)
        data = dict(schema_version=1, timestamp=now, status=status,
            mode='continuous_prefill',
            node=node, pid=os.getpid(), run=str(run), started_at=attempt_started, stop_at=stop_at,
            concurrent_requests_per_instance=concurrency, inflight=active_inflight,
            peak_inflight=peak_inflight, target_rps=None, queue_policy='observe_only',
            tpot_note='Not applicable: one output token', cumulative=dict(cumulative),
            completed_rps=cumulative['successful_requests']/elapsed,
            input_tokens_per_s=cumulative['input_tokens']/elapsed, error=failure,
            trace_cursor=str(root/'cursors'/(node+'.json')),
            stage_dir=str(stage) if stage else None)
        save(node_state, data)
        if stage:
            save(stage/'progress.json', dict(data, offered_requests=offered,
                dispatched_requests=dispatched, completed_requests=len(rows),
                successful_requests=sum(r['status']=='ok' for r in rows),
                inflight=active_inflight if inflight is None else inflight,
                queue_waiting={}, queue_sample_fresh={}, valid_so_far=all(r['status']=='ok' for r in rows),
                reasons=[r.get('error', r['status']) for r in rows if r['status']!='ok']))

    async def stopped():
        if cancelled.is_set() or time.time() >= stop_at:
            return True
        try:
            return current_instance(config, run, node) is None
        except (OSError, ValueError, KeyError, RuntimeError, InterruptedError):
            return True

    try:
        # The model probe can finish before the manager publishes readiness.
        while not cancelled.is_set() and time.time() < stop_at:
            own_membership(config, run, node)
            endpoint = current_instance(config, run, node)
            if endpoint:
                break
            publish('waiting_for_model')
            await asyncio.sleep(.1)
        else:
            return
        stream = PrefillTrace(**config['trace'], cursor_path=root/'cursors'/(node+'.json'),
                              node=node, nodes=config['nodes'])
        batch_index = 0
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)) as session:
            while not await stopped():
                batch = stream.peek_batch()
                requests = batch['requests']
                if not requests:
                    stream.commit()
                    continue
                batch_index += 1
                phase_id = f'{run.name}-prefill-{time.time_ns()}-{batch_index:06d}-{uuid.uuid4().hex[:8]}'
                stage = root/'stages'/phase_id/node
                stage.mkdir(parents=True, exist_ok=False, mode=0o700)
                rows, offered, dispatched, stage_started = [], len(requests), 0, time.time()
                active_inflight, peak_inflight = 0, 0
                save(stage/'workload.json', batch)
                save(run/'benchmark_progress.json', dict(phase=phase_id, stage='running', stage_dir=str(stage), timestamp=time.time()))
                with (stage/'requests.jsonl').open('a', buffering=1) as raw, (stage/'full_metrics.csv').open('w', newline='', buffering=1) as table:
                    fields=['node','sample_id','source_index','source_cycle','source_model','original_output_tokens',
                        'status','planned_prompt_tokens','prompt_tokens','completion_tokens',
                        'ttft_ms','e2e_ms','tpot_ms','server_queue_ms','dispatch_epoch','finished_epoch','error']
                    writer = csv.DictWriter(table, fieldnames=fields)
                    writer.writeheader()
                    def on_start(metadata):
                        nonlocal dispatched, active_inflight, peak_inflight
                        dispatched += 1
                        cumulative['dispatched_requests'] += 1
                        active_inflight += 1
                        peak_inflight = max(peak_inflight, active_inflight)
                        publish('running')

                    def on_row(row):
                        nonlocal failure, dispatched, active_inflight
                        active_inflight -= 1
                        # A stop discovered by the final check sent no POST.
                        if row['dispatch_epoch'] is None:
                            dispatched -= 1
                            cumulative['dispatched_requests'] -= 1
                        raw.write(json.dumps(row, allow_nan=False)+'\n')
                        converted = {key:row.get(key) for key in fields}
                        for key in ('ttft','e2e','tpot','server_queue'):
                            value = row.get(key+'_s')
                            converted[key+'_ms'] = value*1000 if value is not None else None
                        writer.writerow(converted)
                        rows.append(row)
                        cumulative['completed_requests'] += 1
                        if row['status']=='ok':
                            cumulative['successful_requests'] += 1
                            cumulative['input_tokens'] += row['prompt_tokens']
                        elif failure is None:
                            failure = row.get('error', row['status'])
                        publish('running')

                    batch_done = asyncio.Event()
                    async def heartbeat():
                        while not batch_done.is_set():
                            publish('running', force=True)
                            try:
                                await asyncio.wait_for(batch_done.wait(), .5)
                            except asyncio.TimeoutError:
                                pass
                    heartbeat_task = asyncio.create_task(heartbeat())
                    try:
                        await run_prefill_batch(session, endpoint, requests, stopped,
                            concurrency=concurrency,
                            request_timeout_s=config.get('request_timeout_s', 600),
                            on_start=on_start, on_row=on_row)
                    finally:
                        batch_done.set()
                        await heartbeat_task
                summary = batch_summary(rows, stage_started, time.time(), offered, concurrency=concurrency)
                summary['peak_inflight_requests'] = peak_inflight
                save(stage/'summary.json', summary)
                publish('running' if summary['valid'] else 'interrupted' if await stopped() else 'failed', force=True)
                if not summary['valid']:
                    if await stopped():
                        break
                    # Preserve partial records and retry the uncommitted batch.
                    retry_until = time.monotonic() + config.get('error_retry_s', 5)
                    while time.monotonic() < retry_until and not await stopped():
                        await asyncio.sleep(min(.05, max(0, retry_until-time.monotonic())))
                    failure = None
                    continue
                stream.commit()
                # No rate timer, arrival-time replay, queue gate, or inter-batch pause.
    except (OSError, ValueError, KeyError, RuntimeError, InterruptedError) as exc:
        failure = type(exc).__name__
    finally:
        if stream is not None:
            stream.close()
        publish('failed' if failure and not cancelled.is_set() else 'stopped', force=True)


def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--node', required=True)
    asyncio.run(work(parser.parse_args()))


if __name__=='__main__':
    main()
