"""Readiness sampling lag cannot replace the authenticated preemption authority."""
import asyncio
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock
import prefill_worker_ephemeral_lease as worker
from test_prefill_worker import GuardFixture
from test_prefill_worker_ephemeral import result

class LeaseWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_readiness_lag_keeps_sending_but_preemption_stops(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as name:
            fixture=GuardFixture(Path(name));fixture.config['concurrent_requests_per_instance']=4
            fixture.sync();sent=[]
            async def send(session, endpoint, request, stop_check, **kwargs):
                self.assertFalse(await stop_check())
                sent.append(request['sample_id'])
                # Simulate delayed manager readiness after the startup probe.
                fixture.fleet_state['attempts']['a']['status']['timestamp']=time.time()-120
                fixture.deployment['status']='unhealthy'
                fixture.sync()
                await asyncio.sleep(0)
                return result(request)
            actual=worker.save_live
            def publish(path,value):
                actual(path,value)
                if value['cumulative']['successful_requests']>=32:
                    fixture.lease_a['preemption_requested']={'request_id':'foreground'}
                    fixture.sync()
            with mock.patch.dict(os.environ,EXPERIMENT_GUARD_LEASE_RECEIPT=str(fixture.receipt)), \
                 mock.patch.object(asyncio.get_running_loop(),'add_signal_handler'), \
                 mock.patch.object(worker,'send_prefill',side_effect=send), \
                 mock.patch.object(worker,'save_live',side_effect=publish):
                await asyncio.wait_for(worker.work(fixture.args()),4)
            self.assertEqual(len(sent),32)
