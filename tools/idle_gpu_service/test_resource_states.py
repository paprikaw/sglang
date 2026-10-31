"""Dashboard state evidence must distinguish worker failure from model loading."""
import importlib.util
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[2] / 'experiments/idle_gpu_recovery_20261020/dashboard/resource_overview.py'
spec = importlib.util.spec_from_file_location('overview_states', SOURCE)
overview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overview)


class ResourceStatesTests(unittest.TestCase):
    def setUp(self):
        self.lease = dict(nodes=['node'], state='active', heartbeat_at=999,
                          ttl_s=15, resource_priority='background', experiment_id='current-run')

    def state(self, worker):
        return overview.node_state('node', [self.lease], worker, 1000)[0]

    def test_failed_worker_is_not_loading_even_with_old_last_sample(self):
        self.assertEqual(self.state(dict(deployment_id='current-run', worker_status='failed',
                                         worker_updated_at=100)), 'failed')

    def test_only_fresh_matching_running_worker_counts_as_benchmark(self):
        worker = dict(deployment_id='current-run', worker_status='running', worker_updated_at=999)
        self.assertEqual(self.state(worker), 'benchmark')
        self.assertEqual(self.state(dict(worker, worker_updated_at=900)), 'unknown')
        self.assertEqual(self.state(dict(worker, deployment_id='old-run')), 'unknown')

    def test_genuine_loading_is_separate_from_unavailable_or_stale_worker(self):
        self.assertEqual(self.state(dict(deployment_id='current-run', worker_status='loading')), 'loading')
        for status in ('stale', 'awaiting_worker'):
            self.assertEqual(self.state(dict(deployment_id='current-run', worker_status=status)), 'unknown')

    def test_foreground_and_preemption_take_precedence_over_failed_worker(self):
        worker = dict(deployment_id='current-run', worker_status='failed')
        self.lease['resource_priority'] = 'normal'
        self.assertEqual(self.state(worker), 'foreground')
        self.lease['preemption_requested'] = True
        self.assertEqual(self.state(worker), 'draining')


if __name__ == '__main__':
    unittest.main()
