#!/usr/bin/env python3
"""Guard-owned continuous BurstGPT prefill with no retained benchmark artifacts.

Only nodes/<node>.json is atomically replaced with bounded latest live counters.
Trace cursors, per-request timings, synthetic tokens and batch summaries live in
memory; every new deployment starts the trace at its beginning. Model weights,
KV checkpoints and foreground experiment artifacts are outside this worker.
"""
from __future__ import annotations
import sys
# Background workers must not spend project inodes on imported bytecode either.
sys.dont_write_bytecode = True
import argparse
import asyncio
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import tempfile
import time
import aiohttp
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'idle_gpu_service'))
from node_runner_benchmark import read
from worker import own_membership
from loadgen import percentiles
from prefill_client import send_prefill
from prefill_trace import PrefillTrace
from trace_stream import TraceStream


class MemoryTraceStream(TraceStream):
    """The frozen source reader/selection, with cursor state held only in RAM."""
    def __init__(self, dataset_path, batch_size=256, expected_source_sha256=None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        self.source = Path(dataset_path).expanduser().resolve(strict=True)
        self.batch_size = batch_size
        self.filters = dict(min_output_tokens=1, max_total_tokens=None)
        initial = self.source.stat()
        digest = hashlib.sha256()
        with self.source.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
        self.sha256 = digest.hexdigest()
        if expected_source_sha256 is not None and self.sha256 != expected_source_sha256:
            raise ValueError('BurstGPT source SHA256 differs from the expected source')
        self._source_stat = (initial.st_ino, initial.st_size, initial.st_mtime_ns)
        self._verify_source()
        self._file = self._reader = self._pending = self._pending_state = None
        with self.source.open('r', encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(iter(stream.readline, ''))
            try:
                self.columns = next(reader)
            except StopIteration:
                raise ValueError('BurstGPT trace has no CSV header') from None
            missing = {'Timestamp', 'Model', 'Request tokens', 'Response tokens'}.difference(self.columns)
            if missing:
                raise ValueError('BurstGPT trace is missing required columns')
            self._header_offset = stream.tell()
        self.state = dict(schema_version=1, dataset_path=str(self.source), dataset_sha256=self.sha256,
            filters=self.filters, cycle=0, last_source_index=-1, source_offset=self._header_offset,
            last_source_timestamp=None, committed_requests=0, committed_batches=0,
            committed_source_rows=0, completed_cycles=0, exclusion_counts={})
        self._reset_reader(self.state)

    def commit(self):
        if self._pending is None:
            raise RuntimeError('There is no pending trace batch to commit')
        self._verify_source()
        self.state = self._pending_state
        self._pending = self._pending_state = None
        if self.state['last_source_index'] == -1:
            self._reset_reader(self.state)
        return copy.deepcopy(self.state)


class MemoryPrefillTrace(PrefillTrace):
    """The frozen fixed-node partition/context policy, without any cursor file."""
    def __init__(self, dataset_path, node, nodes, batch_size=64,
                 max_context_tokens=32768, expected_source_sha256=None):
        if (not isinstance(nodes, (list, tuple)) or not nodes
                or any(not isinstance(item, str) or not item for item in nodes)
                or len(set(nodes)) != len(nodes)):
            raise ValueError('nodes must contain distinct nonempty node names')
        if node not in nodes:
            raise ValueError('node must belong to the fixed partition nodes')
        for key, value in (('batch_size', batch_size), ('max_context_tokens', max_context_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(key + ' must be a positive integer')
        if max_context_tokens < 2:
            raise ValueError('max_context_tokens must fit a prompt and one token')
        self.node, self.nodes = node, sorted(nodes)
        self.slot = self.nodes.index(node)
        self.batch_size, self.max_context_tokens = batch_size, max_context_tokens
        self._pending = self._pending_counts = None
        self._commit_failed = False
        self._stream = MemoryTraceStream(dataset_path, batch_size=batch_size * len(nodes),
            expected_source_sha256=expected_source_sha256)
        self.state = dict(schema_version=1, kind='burstgpt_prefill_partition_cursor',
            partition=dict(method='source_index_modulo_fixed_nodes', node=node,
                nodes=self.nodes, slot=self.slot, max_context_tokens=max_context_tokens, output_tokens=1),
            source_cursor=copy.deepcopy(self._stream.state), committed_requests=0,
            committed_batches=0, excluded_for_context=0, other_partition_rows=0)
        self.sha256, self.source = self._stream.sha256, self._stream.source

    def close(self):
        self._stream.close()

    def commit(self):
        if self._pending is None:
            raise RuntimeError('There is no pending prefill trace batch to commit')
        next_state = copy.deepcopy(self.state)
        next_state['committed_requests'] += len(self._pending['requests'])
        next_state['committed_batches'] += 1
        for key, increment in self._pending_counts.items():
            next_state[key] += increment
        next_state['source_cursor'] = self._stream.commit()
        self.state = next_state
        self._pending = self._pending_counts = None
        return copy.deepcopy(self.state)


def save_live(path, value):
    """Replace one bounded current-state file; never append or leave temp files."""
    payload = (json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n').encode()
    if len(payload) > 32768:
        raise ValueError('Live benchmark state exceeds 32 KiB')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def current_instance(config, run, node):
    own_membership(config, run, node)
    fleet = read(Path(config['fleet_runtime']) / 'state.json')
    status = fleet['attempts'][node].get('status', {})
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
    return dict(schema_version=1, mode='continuous_prefill', concurrent_requests_per_instance=concurrency,
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
    """Frozen completion-driven lanes; stop/failure prevents any further POST."""
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
    cancelled = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        loop.add_signal_handler(sig, cancelled.set)
    attempt_started = time.time()
    stop_at = min(read(run/'attempt.json')['stop_at'], attempt_started + config.get('duration_s', 604800))
    cumulative = dict(dispatched_requests=0, completed_requests=0, successful_requests=0, input_tokens=0)
    node_state = Path(config['runtime_root'])/'nodes'/(node + '.json')
    rows, offered, dispatched, phase_id, failure, stream = [], 0, 0, None, None, None
    last_publish, active_inflight, peak_inflight, batch_index = 0., 0, 0, 0
    latest_summary = None

    def publish(status, force=False):
        nonlocal last_publish
        now = time.time()
        if not force and now - last_publish < .5:
            return
        last_publish = now
        elapsed = max(.001, now - attempt_started)
        data = dict(schema_version=1, timestamp=now, status=status, mode='continuous_prefill',
            storage_layout='ephemeral_live_v1', retention='none', phase=phase_id,
            storage_phase=run.name, node=node, pid=os.getpid(), run=str(run),
            started_at=attempt_started, stop_at=stop_at,
            concurrent_requests_per_instance=concurrency, inflight=active_inflight,
            peak_inflight=peak_inflight, target_rps=None, queue_policy='observe_only',
            tpot_note='Not applicable: one output token', cumulative=dict(cumulative),
            completed_rps=cumulative['successful_requests']/elapsed,
            input_tokens_per_s=cumulative['input_tokens']/elapsed,
            offered_requests=offered, dispatched_requests=dispatched,
            completed_requests=len(rows), successful_requests=sum(r['status']=='ok' for r in rows),
            valid_so_far=all(r['status']=='ok' for r in rows), error=failure,
            trace_cursor=None, stage_dir=None, latest_batch=latest_summary)
        try:
            save_live(node_state, data)
        except OSError:
            # Live telemetry must not abandon the GPU just because storage is full.
            pass

    async def stopped():
        if cancelled.is_set() or time.time() >= stop_at:
            return True
        try:
            return current_instance(config, run, node) is None
        except (OSError, ValueError, KeyError, RuntimeError, InterruptedError):
            return True

    try:
        while not cancelled.is_set() and time.time() < stop_at:
            own_membership(config, run, node)
            endpoint = current_instance(config, run, node)
            if endpoint:
                break
            publish('waiting_for_model')
            await asyncio.sleep(.1)
        else:
            return
        stream = MemoryPrefillTrace(**config['trace'], node=node, nodes=config['nodes'])
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)) as session:
            while not await stopped():
                batch = stream.peek_batch()
                requests = batch['requests']
                if not requests:
                    stream.commit()
                    await asyncio.sleep(0)
                    continue
                batch_index += 1
                phase_id = f'{run.name}-prefill-{batch_index}'
                rows, offered, dispatched, stage_started = [], len(requests), 0, time.time()
                active_inflight, peak_inflight = 0, 0
                latest_summary = None
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
                    if row['dispatch_epoch'] is None:
                        dispatched -= 1
                        cumulative['dispatched_requests'] -= 1
                    rows.append(row)
                    cumulative['completed_requests'] += 1
                    if row['status'] == 'ok':
                        cumulative['successful_requests'] += 1
                        cumulative['input_tokens'] += row['prompt_tokens']
                    elif failure is None:
                        # Transport errors are already safe codes, still bound their size.
                        failure = str(row.get('error', row['status']))[:128]
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
                        concurrency=concurrency, request_timeout_s=config.get('request_timeout_s', 600),
                        on_start=on_start, on_row=on_row)
                finally:
                    batch_done.set()
                    await heartbeat_task
                latest_summary = batch_summary(rows, stage_started, time.time(), offered, concurrency=concurrency)
                latest_summary['peak_inflight_requests'] = peak_inflight
                publish('running' if latest_summary['valid'] else 'interrupted' if await stopped() else 'failed', force=True)
                if not latest_summary['valid']:
                    if await stopped():
                        break
                    retry_until = time.monotonic() + config.get('error_retry_s', 5)
                    while time.monotonic() < retry_until and not await stopped():
                        await asyncio.sleep(min(.05, max(0, retry_until-time.monotonic())))
                    failure = None
                    continue
                stream.commit()
    except (OSError, ValueError, KeyError, RuntimeError, InterruptedError) as exc:
        failure = type(exc).__name__
    finally:
        if stream is not None:
            stream.close()
        publish('failed' if failure and not cancelled.is_set() else 'stopped', force=True)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--node', required=True)
    asyncio.run(work(parser.parse_args()))


if __name__ == '__main__':
    main()
