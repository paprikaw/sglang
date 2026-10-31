"""Guard receipt and real HTTP integration for independent sequential workers."""
from __future__ import annotations
import argparse
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

import prefill_worker as worker
from node_runner_benchmark import save, sha
try:
    from aiohttp import web
except ImportError:
    web = None


class GuardFixture:
    def __init__(self, root):
        self.root = root
        self.run = root/'run-a'
        self.run.mkdir()
        self.registry, self.fleet = root/'registry', root/'fleet'
        self.runtime = root/'benchmark'
        self.config_path = root/'config.json'
        self.source = root/'trace.csv'
        with self.source.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['Timestamp','Model','Request tokens','Response tokens'])
            writer.writerows([(i,'GPT-4' if i % 3 else 'ChatGPT',3+i,100000+i) for i in range(8)])
        self.config = dict(fleet_runtime=str(self.fleet), runtime_root=str(self.runtime),
            nodes=['a','b'], duration_s=604800, error_retry_s=.01, request_timeout_s=1,
            trace=dict(dataset_path=str(self.source),batch_size=4,max_context_tokens=32768))
        save(self.run/'dashboard_spec.json',dict(frozen_inputs=[]))
        digest = sha(self.run/'dashboard_spec.json')
        self.receipt = self.run/'receipt.json'
        save(self.receipt,dict(registry_dir=str(self.registry),lease_id='lease-a',
            token='private-a',spec_sha256=digest))
        self.lease_a = dict(lease_id='lease-a',state='active',nodes=['a'],
            spec_sha256=digest,heartbeat_at=time.time(),ttl_s=15,
            token_sha256=hashlib.sha256(b'private-a').hexdigest())
        self.lease_b = dict(lease_id='lease-b',state='active',nodes=['b'],
            heartbeat_at=time.time(),ttl_s=15)
        self.fleet_state = dict(attempts={
            'a':dict(run=str(self.run),status=dict(ready=True,timestamp=time.time())),
            'b':dict(run=str(root/'run-b'),status=dict(ready=True,timestamp=time.time()))})
        self.deployment = dict(node='a',status='healthy',endpoint='http://127.0.0.1:1',deployment_id='model-a')
        save(self.run/'attempt.json',dict(stop_at=time.time()+3))
        self.sync()

    def sync(self):
        save(self.registry/'registry.json',dict(leases=[self.lease_a,self.lease_b]))
        save(self.fleet/'state.json',self.fleet_state)
        save(self.run/'deployment.json',self.deployment)
        save(self.config_path,self.config)

    def args(self):
        return argparse.Namespace(config=self.config_path,run=self.run,node='a')


class PrefillWorkerContract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = GuardFixture(Path(self.temporary.name))
        self.patch = mock.patch.dict(os.environ,EXPERIMENT_GUARD_LEASE_RECEIPT=str(self.fixture.receipt))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def current(self):
        fixture=self.fixture
        return worker.current_instance(fixture.config,fixture.run,'a')

    def test_private_receipt_authorizes_only_current_own_instance(self):
        self.assertEqual(self.current()['model_id'],'model-a')
        self.fixture.fleet_state['attempts']['a']['run']=str(self.fixture.root/'replacement')
        self.fixture.sync()
        with self.assertRaisesRegex(InterruptedError,'current managed deployment'):
            self.current()

    def test_other_node_preemption_absence_and_expiry_do_not_gate_own(self):
        self.fixture.lease_b.update(preemption_requested=dict(request_id='foreground'),heartbeat_at=0,state='released')
        self.fixture.fleet_state['attempts'].pop('b')
        self.fixture.sync()
        self.assertEqual(self.current()['node'],'a')

    def test_unready_or_stale_own_instance_waits_and_own_preempt_fails(self):
        self.fixture.fleet_state['attempts']['a']['status']['ready']=False
        self.fixture.sync()
        self.assertIsNone(self.current())
        self.fixture.fleet_state['attempts']['a']['status']=dict(ready=True,timestamp=0)
        self.fixture.sync()
        self.assertIsNone(self.current())
        self.fixture.lease_a['preemption_requested']=dict(request_id='foreground')
        self.fixture.sync()
        with self.assertRaisesRegex(InterruptedError,'Normal experiment'):
            self.current()

    def test_synthetic_payload_preserves_input_only_in_memory_and_forces_output1(self):
        row=dict(sample_id='source-4',source_cycle=2,prompt_tokens=32767,output_tokens=999999)
        first=worker.synthetic_request(row,'model-a')
        self.assertEqual(len(first['input_ids']),32767)
        self.assertEqual(first['output_tokens'],1)
        self.assertTrue(all(256<=token<8192 for token in first['input_ids']))
        self.assertEqual(first,worker.synthetic_request(row,'model-a'))
        self.assertNotEqual(first['input_ids'][:20],worker.synthetic_request(row,'model-b')['input_ids'][:20])
        self.assertNotIn('input_ids',row)

    def test_prefill_summary_has_no_tpot_and_observes_queues_without_rate_target(self):
        rows=[dict(status='ok',prompt_tokens=10,ttft_s=.02,e2e_s=.03,tpot_s=None,server_queue_s=100)]
        result=worker.batch_summary(rows,10,12,1)
        self.assertTrue(result['valid'])
        self.assertEqual(result['concurrent_requests_per_instance'],1)
        self.assertEqual(result['queue_policy'],'observe_only')
        self.assertIsNone(result['target_rps'])
        self.assertEqual(result['latency_s']['tpot']['count'],0)
        self.assertIsNone(result['latency_s']['tpot']['p50'])
        self.assertEqual(result['latency_s']['server_queue']['p50'],100)
        self.assertEqual(result['input_tokens_per_s'],5)
        self.assertFalse(worker.batch_summary(rows,10,12,2)['valid'])


@unittest.skipIf(web is None,'aiohttp required')
class PrefillWorkerHTTP(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.fixture=GuardFixture(Path(self.temporary.name))
        self.environment=mock.patch.dict(os.environ,EXPERIMENT_GUARD_LEASE_RECEIPT=str(self.fixture.receipt))
        self.environment.start()
        self.payloads=[]
        self.running=self.max_running=0
        self.finish_count=4
        app=web.Application()
        app.router.add_post('/generate',self.generate)
        self.runner=web.AppRunner(app)
        await self.runner.setup()
        self.site=web.TCPSite(self.runner,'127.0.0.1',0)
        await self.site.start()
        port=self.site._server.sockets[0].getsockname()[1]
        self.fixture.deployment['endpoint']=f'http://127.0.0.1:{port}'
        self.fixture.sync()
        loop=asyncio.get_running_loop()
        self.signals=mock.patch.object(loop,'add_signal_handler')
        self.signals.start()

    async def asyncTearDown(self):
        self.signals.stop()
        self.environment.stop()
        await self.runner.cleanup()
        self.temporary.cleanup()

    async def generate(self,request):
        body=await request.json()
        self.payloads.append(body)
        self.running+=1
        self.max_running=max(self.max_running,self.running)
        await asyncio.sleep(.015)
        meta=dict(prompt_tokens=len(body['input_ids']),completion_tokens=1,
            queue_time=99.5,num_retractions=5,finish_reason=dict(type='length'))
        response=web.Response(text='data: '+json.dumps(dict(text='SECRET OUTPUT TEXT',meta_info=meta))+'\n\ndata: [DONE]\n\n',
            content_type='text/event-stream')
        self.running-=1
        if len(self.payloads)>=self.finish_count:
            self.fixture.lease_a['preemption_requested']=dict(request_id='foreground')
            self.fixture.sync()
        return response

    async def run_worker(self):
        await asyncio.wait_for(worker.work(self.fixture.args()),2)

    async def test_sequential_continuous_loop_with_real_guard_and_no_payload_persistence(self):
        # A peer normal task must not stop this node's independent worker.
        self.fixture.lease_b['preemption_requested']=dict(request_id='other-foreground')
        self.fixture.fleet_state['attempts'].pop('b')
        self.fixture.sync()
        await self.run_worker()
        self.assertEqual(len(self.payloads),4)
        self.assertEqual(self.max_running,1)
        for payload in self.payloads:
            self.assertEqual(payload['sampling_params']['max_new_tokens'],1)
            self.assertTrue(payload['stream'])
        rows=[]
        for path in self.fixture.runtime.glob('stages/*/a/requests.jsonl'):
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
        self.assertEqual(len(rows),4)
        self.assertEqual([row['source_index'] for row in rows],[0,2,4,6])
        self.assertTrue(all(row['status']=='ok' for row in rows))
        self.assertTrue(all(row['tpot_s'] is None for row in rows))
        self.assertTrue(all(row['server_queue_s']==99.5 for row in rows))
        for path in self.fixture.runtime.rglob('*'):
            if path.is_file():
                text=path.read_text()
                self.assertNotIn('input_ids',text)
                self.assertNotIn('SECRET OUTPUT TEXT',text)
        summaries=[json.loads(path.read_text()) for path in self.fixture.runtime.glob('stages/*/a/summary.json')]
        self.assertEqual(summaries[0]['successful_requests'],4)
        self.assertIsNone(summaries[0]['latency_s']['tpot']['p50'])

    async def test_revoked_own_receipt_sends_no_post(self):
        self.fixture.lease_a['state']='released'
        self.fixture.sync()
        await self.run_worker()
        self.assertEqual(self.payloads,[])

    async def test_expired_own_receipt_sends_no_post(self):
        self.fixture.lease_a['heartbeat_at']=0
        self.fixture.sync()
        await self.run_worker()
        self.assertEqual(self.payloads,[])

    async def test_own_preemption_before_start_sends_no_post(self):
        self.fixture.lease_a['preemption_requested']=dict(request_id='foreground')
        self.fixture.sync()
        await self.run_worker()
        self.assertEqual(self.payloads,[])

    async def test_frozen_input_change_sends_no_post(self):
        frozen=self.fixture.root/'frozen.py'
        frozen.write_text('original')
        save(self.fixture.run/'dashboard_spec.json',dict(frozen_inputs=[dict(path=str(frozen),sha256=sha(frozen))]))
        digest=sha(self.fixture.run/'dashboard_spec.json')
        receipt=json.loads(self.fixture.receipt.read_text())
        receipt['spec_sha256']=digest
        save(self.fixture.receipt,receipt)
        self.fixture.lease_a['spec_sha256']=digest
        self.fixture.sync()
        frozen.write_text('changed')
        await self.run_worker()
        self.assertEqual(self.payloads,[])


if __name__=='__main__':
    unittest.main()
