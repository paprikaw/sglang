"""CPU-only checks for benchmark lifetime and immutable Guard admission."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import node_runner_benchmark as runner
import test_node_runner as baseline


class FrozenBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.worker = self.root / 'worker.py'
        self.worker.write_text('print("worker")\n')
        self.config = self.root / 'workload.json'
        self.config.write_text('{"dataset":"existing"}\n')
        self.data = self.root / 'trace.csv'
        self.data.write_text('length\n128\n')
        self.cfg = dict(server_python=sys.executable, benchmark=dict(
            worker_script=str(self.worker), config_path=str(self.config),
            frozen_files={str(self.data): runner.sha(self.data)}))

    def test_freezes_worker_config_and_declared_workload(self):
        values = runner.benchmark_frozen_inputs(self.cfg)
        self.assertEqual({r['path']: r['sha256'] for r in values},
                         {str(p): runner.sha(p) for p in (self.worker, self.config, self.data)})

    def test_changed_dependency_rejected_before_admission(self):
        self.data.write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            runner.benchmark_frozen_inputs(self.cfg)

    def test_relative_missing_and_malformed_inputs_are_rejected(self):
        for changed in (dict(worker_script='relative.py'),
                        dict(config_path=str(self.root / 'missing.json')),
                        dict(frozen_files=[]), dict(frozen_files={str(self.data): 'bad'})):
            with self.subTest(changed=changed):
                cfg = dict(self.cfg, benchmark=dict(self.cfg['benchmark'], **changed))
                with self.assertRaises(ValueError):
                    runner.benchmark_frozen_inputs(cfg)

    def test_no_benchmark_preserves_normal_serving_profile(self):
        self.assertEqual(runner.benchmark_frozen_inputs({}), [])

    def test_worker_spawn_rechecks_exact_lease_and_inherits_guard_env(self):
        args = argparse.Namespace(run=self.root, node='n1')
        child = mock.Mock(pid=1234567)
        with mock.patch.object(runner, 'lease_check') as lease, \
             mock.patch.object(runner.subprocess, 'Popen', return_value=child) as spawn, \
             mock.patch.object(runner, 'record', return_value={'ticks': 42}), \
             mock.patch.dict(os.environ, {'EXPERIMENT_GUARD_LEASE_RECEIPT': '/private/receipt'}):
            actual, owned, status = runner.start_benchmark_worker(self.cfg, args, '/private/receipt')
        lease.assert_called_once_with('/private/receipt', self.root, 'n1')
        self.assertIs(actual, child)
        self.assertEqual(owned, {child.pid: 42})
        self.assertEqual(spawn.call_args.args[0], [sys.executable, str(self.worker), '--config',
                         str(self.config), '--run', str(self.root), '--node', 'n1'])
        self.assertEqual(spawn.call_args.kwargs['env']['EXPERIMENT_GUARD_LEASE_RECEIPT'], '/private/receipt')
        self.assertTrue(spawn.call_args.kwargs['start_new_session'])
        self.assertEqual(runner.read(self.root / 'benchmark_worker.json')['state'], 'running')

    def test_preemption_before_worker_spawn_starts_no_client(self):
        args = argparse.Namespace(run=self.root, node='n1')
        with mock.patch.object(runner, 'lease_check', side_effect=InterruptedError('preempted')), \
             mock.patch.object(runner.subprocess, 'Popen') as spawn:
            with self.assertRaises(InterruptedError):
                runner.start_benchmark_worker(self.cfg, args, '/private/receipt')
        spawn.assert_not_called()


class BenchmarkPreparationTests(baseline.PreparationTests):
    def prepare(self):
        def git(argv, **kwargs):
            return '' if 'status' in argv else self.commit + '\n'
        with mock.patch.object(runner.subprocess, 'check_output', side_effect=git), contextlib.redirect_stdout(io.StringIO()):
            runner.prepare(self.args)

    def test_benchmark_admission_freezes_all_inputs_and_manifest(self):
        worker = self.root / 'worker.py'
        worker.write_text('pass\n')
        workload = self.root / 'workload.json'
        workload.write_text('{}\n')
        data = self.root / 'data.csv'
        data.write_text('1,2\n')
        self.config['benchmark'] = dict(worker_script=str(worker), config_path=str(workload),
                                        frozen_files={str(data): runner.sha(data)})
        runner.save(self.config_path, self.config)
        self.prepare()
        spec = runner.read(self.args.run / 'dashboard_spec.json')
        files = {row['path']: row['sha256'] for row in spec['frozen_inputs']}
        for path in (worker, workload, data, Path(runner.__file__).resolve()):
            self.assertEqual(files[str(path)], runner.sha(path))
        self.assertEqual(spec['tasks'][0]['controller_script'], str(Path(runner.__file__).resolve()))
        deployment = runner.read(self.args.run / 'deployment.json')
        self.assertEqual(deployment['benchmark_workload']['config_path'], str(workload))
        self.assertEqual(deployment['benchmark_workload']['config_sha256'], runner.sha(workload))


class BenchmarkProbeTests(unittest.TestCase):
    def test_periodic_model_identity_check_has_no_generation_health_probe(self):
        with baseline.fake_api() as (cfg, calls):
            cfg['benchmark'] = {'worker_script': '/unused'}
            result = {}
            runner.probe(cfg, baseline.ProbeEvent(monitor_once=True), result, 'deployment-unique')
        self.assertTrue(result['ready'])
        self.assertEqual(result['periodic_check'], 'model_identity_no_inference')
        self.assertEqual(calls.count('/health'), 1)  # Startup only.
        self.assertEqual(calls.count('/v1/models'), 2)
        self.assertEqual(len([item for item in calls if isinstance(item, dict)]), 2)

    def test_periodic_check_rejects_foreign_model_identity(self):
        class Response:
            status = 200
            def __init__(self, value): self.value = io.BytesIO(json.dumps(value).encode())
            def read(self, *args): return self.value.read(*args)
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def __iter__(self): return iter([b'data: {"choices":[{}]}\n', b'data: [DONE]\n'])
        responses = [Response({}), Response({'data': [{'id': 'mine'}]}),
                     Response({'choices': [{}]}), Response({}), Response({'data': [{'id': 'foreign'}]})]
        with mock.patch.object(runner, 'urlopen', side_effect=responses):
            result = {}
            runner.probe({'api_port': 30000, 'benchmark': {'enabled': True}},
                         baseline.ProbeEvent(monitor_once=True), result, 'mine')
        self.assertFalse(result['ready'])


class WorkerCleanupTests(unittest.TestCase):
    def test_reused_pid_never_signals_group_or_process(self):
        child = mock.Mock(pid=1234567)
        child.poll.return_value = 0
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(runner, 'record', return_value={'ticks': 99, 'state': 'S'}), \
             mock.patch.object(runner, 'track_worker_owned'), \
             mock.patch.object(runner.os, 'killpg') as killpg, \
             mock.patch.object(runner.os, 'kill') as kill:
            self.assertTrue(runner.stop_benchmark_worker(child, {1234567: 42}, {'state': 'failed'}, directory))
            self.assertEqual(runner.read(Path(directory) / 'benchmark_worker.json')['state'], 'failed')
        killpg.assert_not_called()
        kill.assert_not_called()

    def test_worker_process_group_cleanup_preserves_unrelated_sibling(self):
        sibling = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
        worker = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], start_new_session=True)
        def cleanup():
            for child in (sibling, worker):
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=3)
        self.addCleanup(cleanup)
        ticks = runner.record(worker.pid)['ticks']
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(runner.stop_benchmark_worker(worker, {worker.pid: ticks}, {'state': 'running'}, directory))
            receipt = runner.read(Path(directory) / 'benchmark_worker.json')
            self.assertTrue(receipt['owned_processes_exited'])
            self.assertTrue(receipt['partial_results'])
        self.assertIsNone(sibling.poll())


class ControllerBenchmarkTests(unittest.TestCase):
    def run_controller(self, preempt=False, ready=True):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            cfg = dict(guard_script='/tmp/guard/dashboard_guard.py', server_python=sys.executable,
                       remote_python=sys.executable, api_port=30000, benchmark=dict(
                           worker_script='/tmp/worker.py', config_path='/tmp/workload.json'))
            runner.save(run / 'frozen_profile.json', cfg)
            runner.save(run / 'deployment.json', {'deployment_id': 'unique'})
            runner.save(run / 'attempt.json', {'stop_at': time.time() + 60})
            runner.save(run / 'remote_status.json', {'ready': ready, 'timestamp': time.time()})
            args = argparse.Namespace(run=run, config=run / 'config.json', node='n1', job_id='123')
            ssh = mock.Mock()
            ssh.stdin.closed = False
            ssh.poll.side_effect = [None, None, 0]
            worker = mock.Mock(pid=9876543)
            worker.poll.return_value = 17
            events = []
            ssh.stdin.close.side_effect = lambda: events.append('model_stdin_close')
            artifact = types.ModuleType('artifact_visibility')
            artifact.wait_for_stable_json = mock.Mock(return_value=(
                {'owned_processes_exited': True, 'gpu_idle': True}, {}))
            lease = {'owner_pid': 100, 'owner_start_ticks': 2}
            checks = [({}, lease), InterruptedError('normal demand')] if preempt else None
            def stop_worker(*args):
                events.append('worker_stop')
                return True
            with mock.patch.dict(sys.modules, {'artifact_visibility': artifact}), \
                 mock.patch.dict(os.environ, {'EXPERIMENT_GUARD_LEASE_RECEIPT': '/receipt',
                                              'EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS': '/bindings'}), \
                 mock.patch.object(runner, 'lease_check', side_effect=checks, return_value=({}, lease)), \
                 mock.patch.object(runner.os, 'getppid', return_value=100), \
                 mock.patch.object(runner, 'record', side_effect=lambda pid: {'pid': pid, 'ticks': 2}), \
                 mock.patch.object(runner.signal, 'signal'), \
                 mock.patch.object(runner.subprocess, 'Popen', return_value=ssh), \
                 mock.patch.object(runner, 'track_worker_owned'), \
                 mock.patch.object(runner, 'start_benchmark_worker', return_value=(
                     worker, {worker.pid: 2}, {'state': 'running'})) as start, \
                 mock.patch.object(runner, 'stop_benchmark_worker', side_effect=stop_worker):
                result = runner.controller(args)
            self.assertEqual(result, 0)
            self.assertLess(events.index('worker_stop'), events.index('model_stdin_close'))
            self.assertEqual(start.call_count, int(ready and not preempt))
            if ready and not preempt:
                status = runner.read(run / 'benchmark_worker.json')
                self.assertEqual(status['state'], 'failed')
                self.assertEqual(status['exit_code'], 17)
            return runner.read(run / 'task/cleanup.json')

    def test_worker_failure_is_recorded_without_retry_or_model_restart(self):
        self.run_controller()

    def test_preemption_stops_worker_before_model_lifetime_channel(self):
        cleanup = self.run_controller(preempt=True)
        self.assertTrue(cleanup['benchmark_worker_exited'])

    def test_worker_does_not_start_until_model_ready(self):
        self.run_controller(ready=False)


if __name__ == '__main__':
    unittest.main()


class BenchmarkCounterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        self.config = self.root / 'benchmark.json'
        runner.save(self.config, {'runtime_root': str(self.root / 'benchmark')})
        self.cfg = {'benchmark': {'config_path': str(self.config)}}
        self.stage = self.root / 'benchmark/stages/phase-a/n1'
        runner.save(self.run / 'benchmark_progress.json', {'phase': 'phase-a',
                    'stage': 'running', 'stage_dir': str(self.stage)})
        self.values = dict(timestamp=time.time(), offered_requests=8, dispatched_requests=5,
            completed_requests=3, successful_requests=2, inflight=2, target_rps=.5,
            completed_rps=.2, queue_waiting={'n1': 1, 'n2': 99},
            queue_sample_fresh={'n1': True}, queue_policy='no_backlog', valid_so_far=True,
            reasons=[], status='running')
        runner.save(self.stage / 'progress.json', self.values)

    def test_reports_own_real_counters_without_changing_readiness(self):
        actual = runner.benchmark_progress_snapshot(self.cfg, self.run, 'n1')
        self.assertEqual(actual['benchmark_requests'], 5)
        self.assertEqual(actual['completed_requests'], 3)
        self.assertEqual(actual['failed_requests'], 1)
        self.assertEqual(actual['inflight_requests'], 2)
        self.assertEqual(actual['benchmark_queue_waiting'], 1)
        self.assertEqual(actual['benchmark_target_node_rps'], .5)
        self.assertEqual(actual['progress_percent'], 37.5)
        self.assertEqual(actual['benchmark_progress_path'], str(self.stage / 'progress.json'))
        self.assertNotIn('valid_points', actual)
        self.assertNotIn('model_ready', actual)

    def test_foreign_node_or_path_escape_is_not_exported(self):
        for path in (self.root / 'benchmark/stages/phase-a/n2', self.root / 'outside'):
            runner.save(self.run / 'benchmark_progress.json', {'phase': 'phase-a', 'stage_dir': str(path)})
            self.assertEqual(runner.benchmark_progress_snapshot(self.cfg, self.run, 'n1'), {})
        runner.save(self.run / 'benchmark_progress.json', {'phase': '../../escape'})
        self.assertEqual(runner.benchmark_progress_snapshot(self.cfg, self.run, 'n1'), {})

    def test_completed_phase_retains_counter_path_without_worker_stage_dir(self):
        runner.save(self.run / 'benchmark_progress.json', {'phase': 'phase-a', 'stage': 'complete'})
        self.assertEqual(runner.benchmark_progress_snapshot(self.cfg, self.run, 'n1')['benchmark_requests'], 5)

    def test_missing_or_inconsistent_counts_are_not_exported(self):
        for change in ({'dispatched_requests': 1}, {'completed_requests': True}, {'inflight': -1}):
            runner.save(self.stage / 'progress.json', dict(self.values, **change))
            self.assertEqual(runner.benchmark_progress_snapshot(self.cfg, self.run, 'n1'), {})
