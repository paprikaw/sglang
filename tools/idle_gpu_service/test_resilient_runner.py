"""CPU-only regression checks: reporting I/O errors cannot orphan owned work."""
import argparse
import contextlib
import errno
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import types
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('resilient_runner', ROOT / 'experiments/cluster_backfill_20261020/node_runner_qwen_resilient.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def quota():
    return OSError(errno.EDQUOT, 'Disk quota exceeded')


class CleanupResilienceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='runner-cleanup-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def spawn(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        def cleanup():
            if child.poll() is None:
                child.kill()
            child.wait()
        self.addCleanup(cleanup)
        return child

    def test_real_worker_stops_when_cleanup_report_hits_quota(self):
        child = self.spawn()
        owned = {child.pid: runner.record(child.pid)['ticks']}
        with mock.patch.object(runner, 'save', side_effect=quota()), contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertTrue(runner.stop_benchmark_worker(child, owned, {'state': 'running'}, self.root))
        self.assertIsNotNone(child.poll())
        self.assertIn('Disk quota exceeded', stderr.getvalue())

    def test_spawned_worker_is_stopped_if_initial_identity_report_fails(self):
        child = self.spawn()
        cfg = {'server_python': sys.executable, 'benchmark': {'worker_script': '/unused', 'config_path': '/unused'}}
        args = argparse.Namespace(run=self.root, node='node')
        with mock.patch.object(runner, 'lease_check'), mock.patch.object(runner, 'benchmark_frozen_inputs'), \
             mock.patch.object(runner.subprocess, 'Popen', return_value=child), \
             mock.patch.object(runner, 'save', side_effect=quota()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(OSError):
                runner.start_benchmark_worker(cfg, args, '/private/receipt')
        self.assertIsNotNone(child.poll())

    def test_worker_cleanup_failure_still_closes_model_lifetime_channel(self):
        remote = mock.Mock()
        remote.stdin.closed = False
        with mock.patch.object(runner, 'stop_benchmark_worker', side_effect=quota()), contextlib.redirect_stderr(io.StringIO()):
            self.assertFalse(runner.stop_controller_resources(None, {}, None, self.root, remote))
        remote.stdin.close.assert_called_once()
        remote.wait.assert_called_once_with(timeout=38)

    def test_report_failure_and_full_stderr_still_close_model_channel(self):
        remote = mock.Mock()
        remote.stdin.closed = False
        bad_log = mock.Mock()
        bad_log.write.side_effect = quota()
        with mock.patch.object(runner, 'stop_benchmark_worker', side_effect=quota()), mock.patch.object(runner.sys, 'stderr', bad_log):
            self.assertFalse(runner.stop_controller_resources(None, {}, None, self.root, remote))
        remote.stdin.close.assert_called_once()
        remote.wait.assert_called_once()

    def test_model_channel_teardown_escalates_after_timeout(self):
        remote = mock.Mock()
        remote.stdin.closed = False
        remote.wait.side_effect = [subprocess.TimeoutExpired('ssh', 38), subprocess.TimeoutExpired('ssh', 3), 0]
        with mock.patch.object(runner, 'stop_benchmark_worker', return_value=True):
            self.assertTrue(runner.stop_controller_resources(None, {}, None, self.root, remote))
        remote.terminate.assert_called_once()
        remote.kill.assert_called_once()
        self.assertEqual(remote.wait.call_count, 3)

    def test_strict_artifact_save_still_raises_and_does_not_leave_temp_inode(self):
        path = self.root / 'binding.json'
        path.write_text('{"original": true}')
        with mock.patch.object(runner.os, 'replace', side_effect=quota()):
            with self.assertRaises(OSError):
                runner.save(path, {'replacement': True})
        self.assertEqual(path.read_text(), '{"original": true}')
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_controller_quota_failure_runs_remote_cleanup_and_keeps_missing_proof_unsafe(self):
        cfg = dict(server_python=sys.executable, guard_script='/frozen/guard.py', api_port=1234)
        args = argparse.Namespace(run=self.root, config=self.root / 'profile.json', node='node', job_id='1')
        remote = mock.Mock()
        remote.stdin.closed = False
        remote.poll.return_value = None
        data = {'frozen_profile.json': cfg, 'deployment.json': {'deployment_id': 'test'},
                'attempt.json': {'stop_at': float('inf')},
                'remote_status.json': {'ready': True, 'timestamp': 10**15}}
        visibility = types.SimpleNamespace(wait_for_stable_json=mock.Mock(side_effect=OSError('missing cleanup proof')))
        writes = []
        def fail_after_remote_start(path, value):
            writes.append(path)
            if len(writes) > 1:
                raise quota()
        with mock.patch.dict(os.environ, {'EXPERIMENT_GUARD_LEASE_RECEIPT': '/private/receipt',
                                         'EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS': '/private/bindings'}), \
             mock.patch.dict(sys.modules, {'artifact_visibility': visibility}), \
             mock.patch.object(runner, 'read', side_effect=lambda p: data[Path(p).name]), \
             mock.patch.object(runner, 'record', return_value={'pid': 100, 'ticks': 10}), \
             mock.patch.object(runner, 'lease_check', return_value=({}, {'owner_pid': 100, 'owner_start_ticks': 10})), \
             mock.patch.object(runner.signal, 'signal'), \
             mock.patch.object(runner.subprocess, 'Popen', return_value=remote), \
             mock.patch.object(runner, 'save', side_effect=fail_after_remote_start), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runner.controller(args), 1)
        remote.stdin.close.assert_called_once()
        remote.wait.assert_called_once_with(timeout=38)
        visibility.wait_for_stable_json.assert_called_once()

    def test_fp8_copy_keeps_validated_entrypoint_and_reuse_contract(self):
        source = (ROOT / 'experiments/cluster_backfill_20261020/node_runner_fp8_resilient.py').read_text()
        self.assertIn("/experiments/idle_gpu_recovery_20261020/fp8_entry.py'", source)
        self.assertIn("reuse_only=True", source)
        self.assertNotIn("'-m', 'sglang.launch_server'", source)


if __name__ == '__main__':
    unittest.main()
