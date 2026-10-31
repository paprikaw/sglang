"""CPU-only regression checks for the shared dashboard telemetry monitor."""
import contextlib
import errno
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

MONITOR = (Path(__file__).resolve().parents[2] /
           'experiments/cluster_backfill_20261020/monitor.py')
spec = importlib.util.spec_from_file_location('cluster_backfill_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class EndLoop(Exception):
    pass


class MonitorResilienceTests(unittest.TestCase):
    def test_failed_worker_wins_over_last_running_live_snapshot(self):
        row = monitor.worker_snapshot('a', '/run/current', dict(status='healthy'),
            dict(node='a', run='/run/current', timestamp=100, status='running'),
            dict(node='a', state='failed', exit_code=-9,
                 error=dict(type='WorkerExited', message='model remains serving')), 101)
        self.assertEqual(row['worker_status'], 'failed')
        self.assertEqual(row['deployment_status'], 'healthy')
        self.assertEqual(row['worker_updated_at'], 100)
        self.assertTrue(row['worker_stale'])
        self.assertEqual(row['worker_exit_code'], -9)
        self.assertEqual(row['worker_error']['type'], 'WorkerExited')

    def test_stale_worker_does_not_look_like_loading_model(self):
        row = monitor.worker_snapshot('a', '/run/current', dict(status='healthy'),
            dict(node='a', run='/run/current', timestamp=100, status='running'),
            dict(node='a', state='running'), 200)
        self.assertEqual(row['worker_status'], 'stale')
        self.assertEqual(row['worker_updated_at'], 100)
        self.assertTrue(row['worker_stale'])

    def test_missing_worker_after_model_readiness_is_explicit(self):
        row = monitor.worker_snapshot('a', '/run/current', dict(status='healthy'), {}, {}, 100)
        self.assertEqual(row['worker_status'], 'awaiting_worker')
        self.assertIsNone(row['worker_updated_at'])
        row = monitor.worker_snapshot('a', '/run/current', dict(status='queued'), {}, {}, 100)
        self.assertEqual(row['worker_status'], 'loading')

    def test_foreground_without_background_run_is_not_loading(self):
        row = monitor.worker_snapshot('a', None, {}, {}, {}, 100)
        self.assertEqual(row['worker_status'], 'not_applicable')

    def test_another_run_node_or_future_timestamp_cannot_be_current(self):
        for run, node, timestamp in (('/old/run', 'a', 100),
                                     ('/run/current', 'b', 100),
                                     ('/run/current', 'a', 1000)):
            with self.subTest(run=run, node=node, timestamp=timestamp):
                row = monitor.worker_snapshot('a', '/run/current', dict(status='healthy'),
                    dict(node=node, run=run, timestamp=timestamp, status='running'), {}, 101)
                self.assertNotEqual(row['worker_status'], 'running')
                self.assertTrue(row['worker_stale'])

    def test_failed_control_record_survives_into_monitor_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service, run = root / 'service', root / 'service/run'
            run.mkdir(parents=True)
            (root / 'state.json').write_text(json.dumps(dict(nodes={'a': dict(state='loading')})))
            (root / 'services.json').write_text(json.dumps(dict(services={'a': dict(root=str(service))})))
            (service / 'state.json').write_text(json.dumps(dict(attempts={'a': dict(run=str(run))})))
            (run / 'deployment.json').write_text(json.dumps(dict(status='healthy', model_path='/model')))
            (run / 'benchmark_worker.json').write_text(json.dumps(dict(node='a', state='failed', exit_code=-9)))
            with patch.object(monitor, 'query_gpu') as query:
                row = monitor.snapshot(root)['nodes']['a']
            self.assertEqual(row['worker_status'], 'failed')
            self.assertEqual(row['deployment_status'], 'healthy')
            self.assertEqual(row['worker_exit_code'], -9)
            query.assert_not_called()

    def test_atomic_quota_failure_preserves_old_snapshot_then_loop_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'all_nodes_state.json'
            target.write_text(json.dumps(dict(timestamp=1, nodes={'old': {}})))
            real_replace = monitor.save.__globals__['os'].replace
            attempts = []
            observations = []

            def replace(source, destination):
                attempts.append(destination)
                if len(attempts) == 1:
                    raise OSError(errno.EDQUOT, 'Disk quota exceeded')
                return real_replace(source, destination)

            def sleep(_interval):
                observations.append(json.loads(target.read_text()))
                if len(observations) == 2:
                    raise EndLoop()

            errors = io.StringIO()
            with patch.object(monitor, 'snapshot', side_effect=[
                    dict(timestamp=2, nodes={'new': {}}),
                    dict(timestamp=3, nodes={'new': {}})]), \
                 patch('prefill_monitor.os.replace', side_effect=replace), \
                 patch.object(monitor.time, 'sleep', side_effect=sleep), \
                 contextlib.redirect_stderr(errors), \
                 self.assertRaises(EndLoop):
                monitor.run_loop(root)
            self.assertEqual(observations[0], dict(timestamp=1, nodes={'old': {}}))
            self.assertEqual(observations[1], dict(timestamp=3, nodes={'new': {}}))
            self.assertEqual(list(root.glob('.all_nodes_state.json.*')), [])
            self.assertIn('snapshot deferred: OSError errno=122', errors.getvalue())

    def test_one_failed_gpu_sample_does_not_drop_healthy_node(self):
        for failure in (OSError(errno.EDQUOT, 'quota'),
                        subprocess.TimeoutExpired('fixture-no-ssh', 1)):
            with self.subTest(failure=type(failure).__name__), \
                 tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                services = {}
                for node in ('a', 'b'):
                    service = root / node
                    run = service / 'run'
                    (service / 'prefill/nodes').mkdir(parents=True)
                    run.mkdir()
                    (service / 'state.json').write_text(json.dumps(dict(
                        attempts={node: dict(run=str(run))})))
                    (run / 'deployment.json').write_text(json.dumps(dict(
                        deployment_id=node, model_path='/fixture/model')))
                    (service / f'prefill/nodes/{node}.json').write_text(json.dumps(
                        dict(run=str(run), timestamp=time.time(), status='running')))
                    services[node] = dict(root=str(service))
                (root / 'state.json').write_text(json.dumps(dict(nodes={
                    'a': dict(state='benchmark'), 'b': dict(state='benchmark')})))
                (root / 'services.json').write_text(json.dumps(dict(services=services)))
                healthy = [dict(index=0, utilization_gpu_pct=90)]
                with patch.object(monitor, 'query_gpu', side_effect=[
                        failure, dict(gpus=healthy, error=None)]) as query, \
                     contextlib.redirect_stderr(io.StringIO()):
                    self.assertTrue(monitor.poll_once(root))
                result = json.loads((root / 'all_nodes_state.json').read_text())
                self.assertEqual(query.call_count, 2)
                self.assertTrue(result['nodes']['a']['gpu_stale'])
                self.assertIsNone(result['nodes']['a']['gpus'])
                self.assertIn(type(failure).__name__, result['nodes']['a']['gpu_error'])
                self.assertFalse(result['nodes']['b']['gpu_stale'])
                self.assertEqual(result['nodes']['b']['gpus'], healthy)

    def test_full_supervisor_log_cannot_crash_recovery_loop(self):
        with patch.object(monitor, 'snapshot', side_effect=OSError(errno.EDQUOT, 'quota')), \
             patch.object(monitor.sys.stderr, 'write', side_effect=OSError(errno.EDQUOT, 'quota')):
            self.assertFalse(monitor.poll_once(Path('/fixture/no-live-output')))

    def test_source_io_error_does_not_publish_fresh_empty_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(monitor, 'read', side_effect=OSError(errno.EIO, 'I/O')), \
                 patch.object(monitor, 'save') as save, \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertFalse(monitor.poll_once(root))
            save.assert_not_called()


if __name__ == '__main__':
    unittest.main()
