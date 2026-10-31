"""CPU-only integration boundaries between the planner, workers, and Guard receipts."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import coordinator
import worker
from node_runner_benchmark import save, sha


class WorkerLeaseContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'run-a'
        self.run.mkdir()
        self.registry = self.root / 'registry'
        self.fleet = self.root / 'fleet'
        self.config = {'fleet_runtime': str(self.fleet)}
        save(self.run / 'dashboard_spec.json', {'frozen_inputs': []})
        digest = sha(self.run / 'dashboard_spec.json')
        self.receipt_path = self.run / 'receipt.json'
        save(self.receipt_path, {'registry_dir': str(self.registry), 'lease_id': 'lease-a',
             'token': 'test-private-token', 'spec_sha256': digest})
        self.lease_a = {'lease_id': 'lease-a', 'state': 'active', 'nodes': ['a'],
                        'spec_sha256': digest, 'heartbeat_at': time.time(), 'ttl_s': 15,
                        'token_sha256': hashlib.sha256(b'test-private-token').hexdigest()}
        self.lease_b = {'lease_id': 'lease-b', 'state': 'active', 'nodes': ['b'],
                        'heartbeat_at': time.time(), 'ttl_s': 15}
        self.state = {'attempts': {
            'a': {'run': str(self.run), 'status': {'ready': True, 'timestamp': time.time()}},
            'b': {'run': str(self.root / 'run-b'), 'status': {'ready': True, 'timestamp': time.time()}}}}
        self.plan = {'members': {'a': {'run': str(self.run), 'lease_id': 'lease-a'},
                                'b': {'run': str(self.root / 'run-b'), 'lease_id': 'lease-b'}}}
        self.env = mock.patch.dict(os.environ, {'EXPERIMENT_GUARD_LEASE_RECEIPT': str(self.receipt_path)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.sync()

    def sync(self):
        save(self.registry / 'registry.json', {'leases': [self.lease_a, self.lease_b]})
        save(self.fleet / 'state.json', self.state)

    def check(self, plan=True):
        return worker.own_membership(self.config, self.run, 'a', self.plan if plan else None)

    def test_exact_own_receipt_and_healthy_cohort_are_accepted(self):
        self.assertEqual(self.check()['lease_id'], 'lease-a')

    def test_worker_can_wait_for_first_plan_before_manager_ready_catches_up(self):
        self.state['attempts']['a']['status']['ready'] = False
        self.sync()
        self.assertEqual(self.check(plan=False)['lease_id'], 'lease-a')
        with self.assertRaises(InterruptedError):
            self.check()

    def test_a_receipt_for_a_cannot_authorize_requests_for_b(self):
        with self.assertRaisesRegex(RuntimeError, 'Exact active node'):
            worker.own_membership(self.config, self.run, 'b')

    def test_preemption_on_either_member_cancels_the_shared_phase(self):
        for key in ('a', 'b'):
            lease = self.lease_a if key == 'a' else self.lease_b
            lease['preemption_requested'] = {'request_id': 'foreground'}
            self.sync()
            with self.assertRaises(InterruptedError):
                self.check()
            del lease['preemption_requested']
        self.sync()

    def test_expired_or_replaced_peer_invalidates_cohort_before_new_send(self):
        self.lease_b['heartbeat_at'] = time.time() - 30
        self.sync()
        with self.assertRaises(InterruptedError):
            self.check()
        self.lease_b['heartbeat_at'] = time.time()
        self.state['attempts']['b']['run'] = str(self.root / 'replacement')
        self.sync()
        with self.assertRaises(InterruptedError):
            self.check()

    def test_own_replacement_never_sends_using_prior_run(self):
        self.state['attempts']['a']['run'] = str(self.root / 'replacement')
        self.sync()
        with self.assertRaisesRegex(InterruptedError, 'current managed deployment'):
            self.check(plan=False)


class CohortPartitionContract(unittest.TestCase):
    def test_distributed_workers_reconstruct_exact_global_burst_schedule(self):
        rows = [dict(sample_id=i, offset_s=offset, prompt_tokens=10+i, output_tokens=2+i)
                for i, offset in enumerate((4, 4, 5, 9, 9, 9, 13))]
        planned, window = coordinator.schedule(rows, 2, 'burstgpt')
        plan = {'id': 'phase', 'members': {'b': {}, 'a': {}}, 'requests': planned}
        a, b = worker.ephemeral_requests(plan, 'a'), worker.ephemeral_requests(plan, 'b')
        combined = sorted(a+b, key=lambda row: row['sample_id'])
        self.assertEqual([row['sample_id'] for row in combined], list(range(len(rows))))
        self.assertEqual([row['offset_s'] for row in combined], [row['offset_s'] for row in planned])
        self.assertLessEqual(abs(len(a)-len(b)), 1)
        self.assertEqual([len(row['input_ids']) for row in combined], [row['prompt_tokens'] for row in rows])
        self.assertEqual(window, len(rows)/2)
