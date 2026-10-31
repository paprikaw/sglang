"""CPU-only fixed-concurrency, failure, preemption, and heartbeat integration."""
import asyncio
import csv
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import aiohttp
from aiohttp import web
import prefill_worker as worker
from test_prefill_worker import GuardFixture


class ConcurrentPrefill(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.active = self.peak = 0
        self.payloads = []
        self.arrived = asyncio.Event()
        self.behavior = 'ok'
        app = web.Application()
        app.router.add_post('/generate', self.generate)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await self.site.start()
        self.endpoint = dict(node='a', model_id='model-a',
            url='http://127.0.0.1:' + str(self.site._server.sockets[0].getsockname()[1]))
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=8))

    async def asyncTearDown(self):
        await self.session.close()
        await self.runner.cleanup()

    async def generate(self, request):
        body = await request.json()
        index = len(self.payloads)
        self.payloads.append(body)
        self.active += 1
        self.peak = max(self.peak, self.active)
        if len(self.payloads) >= 8:
            self.arrived.set()
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        try:
            await asyncio.sleep(.01 if self.behavior == 'failure' and index == 0 else
                                .8 if self.behavior == 'long' else
                                .3 if self.behavior == 'failure' else .02)
            meta = dict(prompt_tokens=len(body['input_ids']), completion_tokens=1,
                        queue_time=500, num_retractions=5)
            if self.behavior == 'failure' and index == 0:
                meta['prompt_tokens'] += 1
            await response.write(('data: ' + json.dumps(dict(meta_info=meta)) + '\n\ndata: [DONE]\n\n').encode())
            await response.write_eof()
        except ConnectionResetError:
            pass
        finally:
            self.active -= 1
        return response

    def requests(self, count):
        return [dict(sample_id=str(i), prompt_tokens=i+3, source_index=i,
                     source_cycle=0, original_output_tokens=100) for i in range(count)]

    async def never_stop(self):
        return False

    async def test_eight_lanes_fill_capacity_and_each_sample_sent_once(self):
        rows = await worker.run_prefill_batch(self.session, self.endpoint,
            self.requests(32), self.never_stop, concurrency=8)
        self.assertEqual(self.peak, 8)
        self.assertEqual(len(self.payloads), 32)
        self.assertEqual(len(rows), 32)
        self.assertEqual({row['sample_id'] for row in rows}, {str(i) for i in range(32)})
        self.assertTrue(all(row['status'] == 'ok' for row in rows))
        self.assertTrue(all(row['server_queue_s'] == 500 for row in rows))
        summary = worker.batch_summary(rows, 10, 12, 32, concurrency=8)
        self.assertTrue(summary['valid'])
        self.assertEqual(summary['mode'], 'continuous_prefill')
        self.assertEqual(summary['concurrent_requests_per_instance'], 8)
        self.assertIsNone(summary['target_rps'])

    async def test_first_failure_cancels_peers_without_new_posts(self):
        self.behavior = 'failure'
        started = time.monotonic()
        rows = await worker.run_prefill_batch(self.session, self.endpoint,
            self.requests(32), self.never_stop, concurrency=8)
        self.assertLess(time.monotonic()-started, .2)
        self.assertEqual(len(self.payloads), 8)
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(row['status'] == 'error' for row in rows), 1)
        self.assertEqual(sum(row['status'] == 'cancelled' for row in rows), 7)
        self.assertEqual(len({row['sample_id'] for row in rows}), 8)

    async def test_own_preempt_cancels_all_lanes_promptly(self):
        self.behavior = 'long'
        stop = asyncio.Event()
        async def stopped():
            return stop.is_set()
        pending = asyncio.create_task(worker.run_prefill_batch(self.session, self.endpoint,
            self.requests(32), stopped, concurrency=8))
        await asyncio.wait_for(self.arrived.wait(), 1)
        started = time.monotonic()
        stop.set()
        rows = await asyncio.wait_for(pending, .2)
        self.assertLess(time.monotonic()-started, .15)
        self.assertEqual(len(self.payloads), 8)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row['status'] == 'cancelled' for row in rows))

    async def test_invalid_concurrency_does_not_send(self):
        for concurrency in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                await worker.run_prefill_batch(self.session, self.endpoint,
                    self.requests(3), self.never_stop, concurrency=concurrency)
        self.assertEqual(self.payloads, [])

    async def test_guard_worker_heartbeat_while_eight_requests_are_inflight(self):
        self.behavior = 'long'
        with tempfile.TemporaryDirectory() as temporary:
            fixture = GuardFixture(Path(temporary))
            fixture.config['concurrent_requests_per_instance'] = 8
            fixture.config['trace']['batch_size'] = 8
            with fixture.source.open('a', newline='') as handle:
                csv.writer(handle).writerows([(i, 'GPT-4', i+3, 500) for i in range(8, 32)])
            fixture.deployment['endpoint'] = self.endpoint['url']
            fixture.sync()
            with mock.patch.dict(os.environ, EXPERIMENT_GUARD_LEASE_RECEIPT=str(fixture.receipt)), \
                 mock.patch.object(asyncio.get_running_loop(), 'add_signal_handler'):
                pending = asyncio.create_task(worker.work(fixture.args()))
                await asyncio.wait_for(self.arrived.wait(), 1)
                await asyncio.sleep(.6)
                node_state = json.loads((fixture.runtime/'nodes'/'a.json').read_text())
                self.assertEqual(node_state['concurrent_requests_per_instance'], 8)
                self.assertEqual(node_state['inflight'], 8)
                self.assertEqual(node_state['peak_inflight'], 8)
                self.assertLess(time.time()-node_state['timestamp'], .3)
                fixture.lease_a['preemption_requested'] = dict(request_id='foreground')
                fixture.sync()
                await asyncio.wait_for(pending, .3)
                self.assertEqual(len(self.payloads), 8)
                self.assertFalse((fixture.runtime/'cursors'/'a.json').exists())
                stage_rows = [json.loads(line) for path in fixture.runtime.glob('stages/*/a/requests.jsonl')
                              for line in path.read_text().splitlines()]
                self.assertEqual(len(stage_rows), 8)
                self.assertTrue(all(row['status'] == 'cancelled' for row in stage_rows))
                final = json.loads((fixture.runtime/'nodes'/'a.json').read_text())
                self.assertEqual(final['inflight'], 0)


if __name__ == '__main__':
    unittest.main()
