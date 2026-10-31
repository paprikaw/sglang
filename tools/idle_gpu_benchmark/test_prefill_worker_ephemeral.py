"""CPU-only no-retention, frozen trace, preemption and atomic live-state checks."""
import sys
sys.dont_write_bytecode = True
import asyncio
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import prefill_worker_ephemeral as worker
from prefill_trace import PrefillTrace
from test_prefill_worker import GuardFixture


def result(request, status='ok'):
    return dict(node='a', sample_id=request['sample_id'], status=status,
        planned_prompt_tokens=len(request['input_ids']), prompt_tokens=len(request['input_ids']),
        completion_tokens=1, ttft_s=.01, e2e_s=.02, tpot_s=None,
        server_queue_s=.03, dispatch_epoch=1., finished_epoch=2., error=None)


class MemoryTraceTests(unittest.TestCase):
    def test_frozen_trace_partition_context_wrap_and_retry_match_without_cursor_files(self):
        with tempfile.TemporaryDirectory(prefix='ephemeral-trace-', dir='/tmp') as name:
            root = Path(name)
            source = root/'trace.csv'
            with source.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['Timestamp', 'Model', 'Request tokens', 'Response tokens'])
                writer.writerows([(0,'GPT-4',3,500), (1,'ChatGPT',5,200),
                    (2,'GPT-4',32,800), (3,'GPT-4',4,0), (4,'GPT-4',8,1),
                    (5,'ChatGPT',1,100), (6,'ChatGPT',0,1), (7,'GPT-4',7,9)])
            metadata = dict(dataset_path=source, node='a', nodes=['b','a'],
                            batch_size=2, max_context_tokens=16)
            memory = worker.MemoryPrefillTrace(**metadata)
            durable = PrefillTrace(**metadata, cursor_path=root/'old/cursor.json')
            self.addCleanup(memory.close)
            self.addCleanup(durable.close)
            original_mkstemp = tempfile.mkstemp
            with mock.patch.object(worker.tempfile, 'mkstemp', side_effect=AssertionError('trace wrote a file')):
                for _ in range(15):
                    actual, expected = memory.peek_batch(), durable.peek_batch()
                    self.assertEqual(actual, expected)
                    actual['requests'].append(dict(sample_id='caller mutation'))
                    self.assertEqual(memory.peek_batch(), expected)
                    # Compare states, permitting only the legacy cursor write outside patch.
                    memory_state = memory.commit()
                    with mock.patch.object(worker.tempfile, 'mkstemp', wraps=original_mkstemp):
                        durable_state = durable.commit()
                    self.assertEqual(memory_state, durable_state)
            self.assertGreater(memory.state['source_cursor']['completed_cycles'], 1)
            self.assertFalse(hasattr(memory, 'cursor_path'))
            self.assertFalse(hasattr(memory, '_temporary'))
            self.assertEqual({p.name for p in root.iterdir()}, {'trace.csv', 'old'})

    def test_expected_source_hash_and_modified_source_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix='ephemeral-source-', dir='/tmp') as name:
            source = Path(name)/'trace.csv'
            source.write_text('Timestamp,Model,Request tokens,Response tokens\n0,GPT-4,3,4\n')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                worker.MemoryPrefillTrace(source, node='a', nodes=['a'], expected_source_sha256='bad')
            with worker.MemoryPrefillTrace(source, node='a', nodes=['a']) as trace:
                source.write_text(source.read_text() + '1,GPT-4,5,2\n')
                with self.assertRaisesRegex(RuntimeError, 'source changed'):
                    trace.peek_batch()
            self.assertEqual([p.name for p in Path(name).iterdir()], ['trace.csv'])


class LiveStateTests(unittest.TestCase):
    def test_repeated_replace_is_one_bounded_current_file_without_history(self):
        with tempfile.TemporaryDirectory(prefix='ephemeral-live-', dir='/tmp') as name:
            root = Path(name)
            path = root/'nodes/a.json'
            for index in range(100):
                worker.save_live(path, dict(status='running', phase=str(index), retention='none',
                    completed_requests=index, storage_layout='ephemeral_live_v1'))
                self.assertEqual(json.loads(path.read_text())['phase'], str(index))
                self.assertEqual([p for p in root.rglob('*') if p.is_file()], [path])
                self.assertLess(path.stat().st_size, 32768)
            with self.assertRaisesRegex(ValueError, '32 KiB'):
                worker.save_live(path, dict(error='x'*32768))
            self.assertEqual(json.loads(path.read_text())['phase'], '99')

    def test_replace_failure_preserves_previous_json_and_removes_temp(self):
        with tempfile.TemporaryDirectory(prefix='ephemeral-atomic-', dir='/tmp') as name:
            path = Path(name)/'a.json'
            worker.save_live(path, dict(phase='previous'))
            original = path.read_bytes()
            def inspect_then_fail(source, destination):
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(json.loads(Path(source).read_text()), dict(phase='next'))
                raise OSError('simulated quota failure')
            with mock.patch.object(worker.os, 'replace', side_effect=inspect_then_fail):
                with self.assertRaises(OSError):
                    worker.save_live(path, dict(phase='next'))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(name).iterdir()), [path])


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_many_batches_save_no_requests_workloads_summaries_cursors_or_payloads(self):
        with tempfile.TemporaryDirectory(prefix='ephemeral-worker-', dir='/tmp') as name:
            fixture = GuardFixture(Path(name))
            fixture.config['concurrent_requests_per_instance'] = 4
            fixture.config['duration_s'] = 60
            fixture.sync()
            baseline = {p for p in fixture.root.rglob('*') if p.is_file()}
            source_digest = hashlib.sha256(fixture.source.read_bytes()).hexdigest()
            sent, snapshots = [], []
            async def send(session, endpoint, request, stop_check, **kwargs):
                self.assertFalse(await stop_check())
                sent.append(dict(sample_id=request['sample_id'], cycle=request['source_cycle']))
                self.assertEqual(request['output_tokens'], 1)
                self.assertEqual(len(request['input_ids']), request['prompt_tokens'])
                await asyncio.sleep(0)
                return result(request)
            save_actual = worker.save_live
            def publish(path, value):
                save_actual(path, value)
                snapshots.append(json.loads(Path(path).read_text()))
                if value['cumulative']['successful_requests'] >= 80:
                    fixture.lease_a['preemption_requested'] = dict(request_id='foreground')
                    fixture.sync()
            loop = asyncio.get_running_loop()
            with mock.patch.dict(os.environ, EXPERIMENT_GUARD_LEASE_RECEIPT=str(fixture.receipt)), \
                 mock.patch.object(loop, 'add_signal_handler'), \
                 mock.patch.object(worker, 'send_prefill', side_effect=send), \
                 mock.patch.object(worker, 'save_live', side_effect=publish):
                await asyncio.wait_for(worker.work(fixture.args()), 4)
            self.assertEqual(len(sent), 80)
            self.assertGreater(sent[-1]['cycle'], 1)
            path = fixture.runtime/'nodes/a.json'
            self.assertEqual({p for p in fixture.root.rglob('*') if p.is_file()} - baseline, {path})
            data = json.loads(path.read_text())
            self.assertEqual(data['status'], 'stopped')
            self.assertEqual(data['retention'], 'none')
            self.assertEqual(data['storage_layout'], 'ephemeral_live_v1')
            self.assertEqual(data['cumulative']['successful_requests'], 80)
            self.assertEqual(data['inflight'], 0)
            self.assertIsNone(data['trace_cursor'])
            self.assertIsNone(data['stage_dir'])
            self.assertLess(path.stat().st_size, 32768)
            self.assertEqual(source_digest, hashlib.sha256(fixture.source.read_bytes()).hexdigest())
            for state in snapshots:
                text = json.dumps(state)
                for forbidden in ('input_ids', 'sample_id', 'source_model', 'original_output_tokens', 'requests.jsonl'):
                    self.assertNotIn(forbidden, text)
            self.assertTrue(any(s['latest_batch'] and s['latest_batch']['valid'] for s in snapshots))
            self.assertTrue(any(s['peak_inflight'] == 4 for s in snapshots))

    async def test_progress_storage_failure_does_not_stop_workload_or_create_artifacts(self):
        with tempfile.TemporaryDirectory(prefix='ephemeral-unwritable-', dir='/tmp') as name:
            fixture = GuardFixture(Path(name))
            sent = 0
            async def send(session, endpoint, request, stop_check, **kwargs):
                nonlocal sent
                sent += 1
                if sent == 12:
                    fixture.lease_a['preemption_requested'] = dict(request_id='foreground')
                    fixture.sync()
                return result(request)
            with mock.patch.dict(os.environ, EXPERIMENT_GUARD_LEASE_RECEIPT=str(fixture.receipt)), \
                 mock.patch.object(asyncio.get_running_loop(), 'add_signal_handler'), \
                 mock.patch.object(worker, 'send_prefill', side_effect=send), \
                 mock.patch.object(worker, 'save_live', side_effect=OSError('disk quota')):
                await asyncio.wait_for(worker.work(fixture.args()), 3)
            self.assertEqual(sent, 12)
            self.assertFalse(fixture.runtime.exists())

    async def test_preemption_inflight_cancels_every_lane_and_prevents_further_requests(self):
        arrived, stopped = asyncio.Event(), asyncio.Event()
        sent = []
        async def send(session, endpoint, request, stop_check, **kwargs):
            sent.append(request['sample_id'])
            if len(sent) == 4:
                arrived.set()
            while not await stop_check():
                await asyncio.sleep(.01)
            return result(request, 'cancelled')
        async def stop_check():
            return stopped.is_set()
        requests = [dict(sample_id=str(i), prompt_tokens=4, source_index=i, source_cycle=0) for i in range(40)]
        with mock.patch.object(worker, 'send_prefill', side_effect=send):
            pending = asyncio.create_task(worker.run_prefill_batch(None, dict(node='a', model_id='m'),
                requests, stop_check, concurrency=4))
            await asyncio.wait_for(arrived.wait(), 1)
            started = time.monotonic()
            stopped.set()
            rows = await asyncio.wait_for(pending, .2)
        self.assertLess(time.monotonic() - started, .15)
        self.assertEqual(len(sent), 4)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r['status'] == 'cancelled' for r in rows))


if __name__ == '__main__':
    unittest.main()
