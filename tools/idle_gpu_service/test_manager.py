"""CPU-only tests of admission, lifecycle and retry safety."""
import dataclasses
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import manager


class FakeProcess:
    next_pid = 1000000

    def __init__(self):
        FakeProcess.next_pid += 1
        self.pid = FakeProcess.next_pid
        self.returncode = None

    def poll(self):
        return self.returncode


class FakeLauncher:
    def __init__(self):
        self.prepares = []
        self.launches = []
        self.stops = []
        self.kills = []

    def prepare(self, booking, run, now):
        attempt = manager.Attempt(booking, run, FakeProcess(), 'preparing', now, 123, io.BytesIO(), io.BytesIO())
        self.prepares.append(attempt)
        return attempt

    def launch(self, attempt, now):
        self.launches.append(attempt)
        attempt.process = FakeProcess()
        attempt.phase, attempt.started_at = 'guard', now
        return attempt

    def stop(self, attempt, reason, now):
        if attempt.stop_requested_at is None:
            attempt.stop_requested_at, attempt.stop_reason = now, reason
            self.stops.append(attempt)

    def kill(self, attempt):
        self.kills.append(attempt)
        attempt.process.returncode = -9

    def close(self, attempt):
        attempt.log.close()
        if attempt.prepare_output:
            attempt.prepare_output.close()


class ReconcileTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {'duration_s': manager.MIN_DURATION_S, 'known_node_specs': {'n1': {}, 'n2': {}},
                       'booking_margin_s': 60, 'retry_backoff_s': 30, 'max_backoff_s': 120}
        self.launcher = FakeLauncher()
        self.service = manager.Reconciler(self.config, self.root, self.launcher, 1000)
        self.bookings = {'n1': manager.Booking('n1', '10', 1000000, 'gres/gpu:A100:4'),
                         'n2': manager.Booking('n2', '11', 1000000, 'gres/gpu:A100:4')}

    def step(self, now=1000, blocked=None, bookings=None, error=None):
        self.service.reconcile(self.bookings if bookings is None else bookings,
                               blocked or {}, now, error)

    def test_one_attempt_per_free_booked_compatible_node(self):
        self.step()
        self.step(now=1001)
        self.assertEqual(len(self.launcher.prepares), 2)
        self.assertEqual(len(self.service.attempts), 2)
        for attempt in self.launcher.prepares:
            self.assertTrue((attempt.run / 'attempt.json').is_file())
            self.assertEqual(manager.read_json(attempt.run / 'attempt.json')['stop_at'], self.service.deadline)

    def test_normal_and_unknown_registry_owners_are_never_preempted(self):
        self.step(blocked={'n1': 'normal experiment', 'n2': 'legacy unknown owner'})
        self.assertEqual(self.launcher.prepares, [])
        self.assertEqual(self.service.nodes['n1']['reason'], 'normal experiment')
        self.assertEqual(self.launcher.stops, [])

    def test_unknown_hardware_skipped_disabled_reason_preserved(self):
        self.config['known_node_specs']['n1'] = {'disabled': True, 'reason': 'protected incompatible CPU cache'}
        del self.config['known_node_specs']['n2']
        self.step()
        self.assertEqual(self.launcher.prepares, [])
        self.assertEqual(self.service.nodes['n1']['reason'], 'protected incompatible CPU cache')
        self.assertIn('no compatible', self.service.nodes['n2']['reason'])

    def test_typed_hardware_dynamic_profile_exact_match(self):
        self.config['default_node_spec'] = {'cache_spec_template': {}}
        self.config['allowed_gpu_types'] = ['a100']
        del self.config['known_node_specs']['n2']
        self.step()
        self.assertIn('n2', self.service.attempts)
        self.assertIsNotNone(manager.profile_reason(self.config, manager.Booking('x', '1', 10000, 'gres/gpu:A1000:4')))

    def test_prepare_then_guard_only_while_still_free(self):
        self.step()
        self.service.attempts['n1'].process.returncode = 0
        self.service.attempts['n2'].process.returncode = 0
        self.step(now=1001, blocked={'n2': 'normal reservation arrived'})
        self.assertEqual(len(self.launcher.launches), 1)
        self.assertEqual(self.launcher.launches[0].booking.node, 'n1')
        self.assertNotIn('n2', self.service.attempts)

    def test_existing_guard_remains_owned_during_own_registry_lease(self):
        self.step()
        self.service.attempts['n1'].process.returncode = 0
        self.step(now=1001)
        self.step(now=1002, blocked={'n1': 'own background lease'})
        self.assertEqual(self.service.attempts['n1'].phase, 'guard')
        self.assertEqual(self.launcher.stops, [])

    def test_booking_end_and_changed_job_stop_only_our_children(self):
        self.step()
        changed = {'n1': dataclasses.replace(self.bookings['n1'], job_id='12')}
        self.step(now=1001, bookings=changed)
        self.assertEqual(len(self.launcher.stops), 2)
        self.assertTrue(all(a.stop_reason == 'booking ended or changed' for a in self.launcher.stops))

    def test_booking_margin_and_deadline_stop_without_relaunch(self):
        self.step()
        ending = {n: dataclasses.replace(b, end_at=1060) for n, b in self.bookings.items()}
        self.step(now=1001, bookings=ending)
        self.assertEqual(len(self.launcher.stops), 2)
        for attempt in self.service.attempts.values():
            attempt.process.returncode = -15
        self.step(now=self.service.deadline)
        self.assertEqual(self.service.attempts, {})
        self.assertEqual(len(self.launcher.prepares), 2)
        self.assertEqual(self.service.stop_reason, 'service deadline reached')

    def test_discovery_failure_stops_active_and_fails_closed(self):
        self.step()
        self.step(now=1001, error='registry unreadable')
        self.assertEqual(len(self.launcher.stops), 2)
        self.assertEqual(len(self.launcher.prepares), 2)

    def test_failed_guard_bounded_backoff_then_fresh_run(self):
        self.step()
        first = self.service.attempts['n1']
        first.process.returncode = 0
        self.step(now=1001)
        first.process.returncode = 2
        self.step(now=1002)
        self.assertEqual(self.service.retries['n1']['retry_at'], 1032)
        self.step(now=1031)
        self.assertNotIn('n1', self.service.attempts)
        self.step(now=1032)
        self.assertNotEqual(self.service.attempts['n1'].run, first.run)
        for n in range(10):
            self.service.retry('n1', 'preempted', 2000 + n)
        self.assertEqual(self.service.retries['n1']['retry_at'], 2129)

    def benchmark_guard(self):
        self.config['benchmark'] = {'worker_script': '/worker.py', 'config_path': '/benchmark.json'}
        self.step()
        attempt = self.service.attempts['n1']
        attempt.process.returncode = 0
        self.step(now=1001)
        return attempt

    def test_failed_worker_restarts_only_after_guard_cleanup_and_retry_backoff(self):
        attempt = self.benchmark_guard()
        manager.atomic_json(attempt.run / 'benchmark_worker.json',
                            {'state': 'failed', 'node': 'n1', 'started_at': 1001,
                             'timestamp': 1002, 'exit_code': -9})
        # Old terminal evidence is still authoritative for the exact owned run.
        self.step(now=5000)
        self.assertEqual(attempt.stop_reason, 'benchmark worker failed (exit code -9)')
        self.assertEqual(self.service.nodes['n1']['state'], 'stopping')
        self.assertEqual(self.launcher.stops.count(attempt), 1)
        self.assertIs(self.service.attempts['n1'], attempt)
        self.assertEqual(len(self.launcher.launches), 1)
        # A living Guard is allowed to complete ordinary cleanup first.
        self.step(now=5001)
        self.assertEqual(self.launcher.stops.count(attempt), 1)
        attempt.process.returncode = 0
        self.step(now=5002)
        self.assertNotIn('n1', self.service.attempts)
        self.assertEqual(self.service.retries['n1']['retry_at'], 5032)
        self.step(now=5031)
        self.assertNotIn('n1', self.service.attempts)
        # An arriving foreground reservation still prevents new admission.
        self.step(now=5032, blocked={'n1': 'normal experiment'})
        self.assertNotIn('n1', self.service.attempts)
        self.assertEqual(self.service.nodes['n1']['reason'], 'normal experiment')
        self.step(now=5033)
        self.assertNotEqual(self.service.attempts['n1'].run, attempt.run)
        self.assertEqual(self.service.attempts['n1'].phase, 'preparing')

    def test_fresh_failed_progress_recovers_when_worker_receipt_is_unavailable(self):
        attempt = self.benchmark_guard()
        manager.atomic_json(attempt.run / 'task/progress.json',
                            {'timestamp': 1002, 'model_ready': True,
                             'benchmark_worker_state': 'failed'})
        self.step(now=1003)
        self.assertIn('fresh controller progress', attempt.stop_reason)
        self.assertEqual(self.launcher.stops, [attempt])

    def test_failed_progress_ignores_stale_malformed_and_prior_attempt_evidence(self):
        attempt = self.benchmark_guard()
        cases = [
            {'timestamp': 1001, 'benchmark_worker_state': 'failed'},
            {'timestamp': 999, 'benchmark_worker_state': 'failed'},
            {'timestamp': 2000, 'benchmark_worker_state': 'failed'},
            {'timestamp': 'not a timestamp', 'benchmark_worker_state': 'failed'},
            {'timestamp': 1100, 'note': 'benchmark failed'},
        ]
        for progress in cases:
            with self.subTest(progress=progress):
                manager.atomic_json(attempt.run / 'task/progress.json', progress)
                self.assertIsNone(manager.benchmark_failure_reason(self.config, attempt, 1100))
        (attempt.run / 'task/progress.json').write_text('{invalid')
        self.assertIsNone(manager.benchmark_failure_reason(self.config, attempt, 1100))
        self.assertEqual(self.launcher.stops, [])

    def test_normal_loading_and_running_benchmark_do_not_trigger_recovery(self):
        attempt = self.benchmark_guard()
        self.step(now=1002)
        manager.atomic_json(attempt.run / 'benchmark_worker.json',
                            {'state': 'running', 'node': 'n1', 'started_at': 1002})
        manager.atomic_json(attempt.run / 'task/progress.json',
                            {'timestamp': 1003, 'model_ready': True,
                             'benchmark_worker_state': 'running'})
        self.step(now=1003)
        self.assertIsNone(attempt.stop_reason)
        self.assertEqual(self.launcher.stops, [])

    def test_disabled_benchmark_and_prepare_do_not_recover_failed_worker(self):
        attempt = self.benchmark_guard()
        manager.atomic_json(attempt.run / 'benchmark_worker.json',
                            {'state': 'failed', 'node': 'n1', 'timestamp': 1002})
        for disabled in (None, {}, {'enabled': False, 'worker_script': '/worker.py'}):
            with self.subTest(benchmark=disabled):
                self.config['benchmark'] = disabled
                self.step(now=1003)
                self.assertIsNone(attempt.stop_reason)
        self.config['benchmark'] = {'worker_script': '/worker.py'}
        attempt.phase = 'preparing'
        self.step(now=1003)
        self.assertIsNone(attempt.stop_reason)
        self.assertEqual(self.launcher.stops, [])

    def test_wrong_node_and_prelaunch_worker_receipts_are_ignored(self):
        attempt = self.benchmark_guard()
        for worker in (
            {'state': 'failed', 'node': 'another-node', 'timestamp': 1002},
            {'state': 'failed', 'node': 'n1', 'timestamp': 999, 'started_at': 998},
            {'state': 'failed', 'node': 'n1', 'timestamp': float('nan')},
        ):
            with self.subTest(worker=worker):
                manager.atomic_json(attempt.run / 'benchmark_worker.json', worker)
                self.step(now=1003)
                self.assertIsNone(attempt.stop_reason)
        self.assertEqual(self.launcher.stops, [])

    def test_foreground_owners_are_never_inspected_for_benchmark_failure(self):
        self.config['benchmark'] = {'worker_script': '/worker.py'}
        with mock.patch.object(manager, 'benchmark_failure_reason') as inspect:
            self.step(blocked={'n1': 'foreground experiment', 'n2': 'foreground lease'})
        inspect.assert_not_called()
        self.assertEqual(self.service.attempts, {})
        self.assertEqual(self.launcher.stops, [])

    def test_slow_prepare_cancelled_and_owned_child_killed_after_grace(self):
        self.step()
        self.step(now=1061)
        self.assertEqual(len(self.launcher.stops), 2)
        self.assertEqual(len(self.launcher.launches), 0)
        self.step(now=1212)
        self.assertEqual(len(self.launcher.kills), 2)
        self.step(now=1213)
        self.assertEqual(self.service.attempts, {})

    def test_snapshot_aggregates_readiness_without_inventing_measurements(self):
        self.step()
        attempt = self.service.attempts['n1']
        manager.atomic_json(attempt.run / 'remote_status.json', {'state': 'ready', 'benchmark_defined': False})
        state = self.service.snapshot(1001)
        self.assertEqual(state['attempts']['n1']['status']['state'], 'ready')
        self.assertFalse(state['attempts']['n1']['status']['benchmark_defined'])
        self.assertIsNone(state['attempts']['n2']['status'])


class InputTest(unittest.TestCase):
    def test_remaining_time(self):
        self.assertEqual(manager.remaining_seconds('7-01:02:03'), 608523)
        self.assertEqual(manager.remaining_seconds('02:03'), 123)
        for bad in ('UNLIMITED', 'INVALID', '1:99:01', '-1:00:00'):
            with self.assertRaises(ValueError):
                manager.remaining_seconds(bad)

    def test_reservations_and_unknown_lease_blocked_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'registry.json'
            state = {'schema_version': 1, 'leases': [{'nodes': ['n1'], 'experiment_id': 'old'}],
                     'preemption_requests': [{'expires_at': 200, 'target_nodes': ['n2'], 'requester_experiment_id': 'normal'},
                                             {'expires_at': 99, 'target_nodes': ['n3']}]}
            manager.atomic_json(path, state)
            before = path.read_bytes()
            blocks = manager.registry_blocks(root, 100)
            self.assertEqual(set(blocks), {'n1', 'n2'})
            self.assertEqual(path.read_bytes(), before)
            path.write_text('{"leases": []}')
            with self.assertRaises(ValueError):
                manager.registry_blocks(root, 100)

    def test_expired_registry_occupancy_becomes_candidate_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'registry.json'
            state = {'schema_version': 1, 'leases': [
                {'nodes': ['old'], 'state': 'active', 'heartbeat_at': 0, 'ttl_s': 15},
                {'nodes': ['cleanup'], 'state': 'quarantined', 'cleanup_grace_expires_at': 99},
                {'nodes': ['live'], 'state': 'active', 'heartbeat_at': 99, 'ttl_s': 15},
                {'nodes': ['quarantine'], 'state': 'quarantined', 'cleanup_grace_expires_at': 101}]}
            manager.atomic_json(path, state)
            before = path.read_bytes()
            self.assertEqual(set(manager.registry_blocks(root, 100)), {'live', 'quarantine'})
            self.assertEqual(path.read_bytes(), before)

    def test_booking_discovery_only_running_existing_allocations(self):
        queue = mock.Mock(stdout='123|n[1-2]|7-00:00:00|gres/gpu:A100:4\n124|cpu1|10:00:00|(null)\n')
        expansion = mock.Mock(stdout='n1\nn2\n')
        with mock.patch.object(manager.subprocess, 'run', side_effect=[queue, expansion]) as run:
            bookings = manager.discover_bookings({}, 100)
        self.assertEqual(set(bookings), {'n1', 'n2'})
        self.assertEqual(bookings['n1'].end_at, 604900)
        self.assertIn('RUNNING', run.call_args_list[0].args[0])
        self.assertFalse(any('salloc' in str(call) or 'sbatch' in str(call) for call in run.call_args_list))

    def test_process_identity_prevents_pid_reuse(self):
        with mock.patch.object(manager, 'process_start_ticks', return_value=9), mock.patch.object(Path, 'read_text', return_value='1 (test) S rest'):
            self.assertFalse(manager.same_process(1, 8))
            self.assertTrue(manager.same_process(1, 9))
            self.assertFalse(manager.same_process(1, None))

    def test_seven_day_minimum_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'runtime_root': str(root), 'node_runner': '/x/runner', 'guard_script': '/x/guard',
                      'duration_s': 604799}
            path = root / 'config.json'
            manager.atomic_json(path, config)
            with self.assertRaisesRegex(ValueError, 'seven days'):
                manager.load_config(path)
            config['duration_s'] = 604800
            manager.atomic_json(path, config)
            self.assertEqual(manager.load_config(path)['duration_s'], 604800)


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {'guard_script': '/guard.py', 'node_runner': '/runner.py',
                       'gpu_registry_dir': '/central', 'python': '/python'}
        self.launcher = manager.Launcher(self.config, Path('/profile.json'))
        self.spec = {'resource_priority': 'background', 'preemptible': True,
                     'gpu_registry_dir': '/central', 'gpu_lease_ttl_s': 15,
                     'tasks': [{'nodes': ['n1']}], 'model_cache': {'policy': 'required'}}
        self.spec_path = self.root / 'spec.json'
        self.log = io.BytesIO()
        self.output = io.BytesIO(json.dumps({'spec_path': str(self.spec_path)}).encode())
        self.attempt = manager.Attempt(manager.Booking('n1', '1', 1000000), self.root,
                                       FakeProcess(), 'preparing', 0, 1, self.log, self.output)

    def test_guard_is_the_only_launch_path_and_never_preempts(self):
        manager.atomic_json(self.spec_path, self.spec)
        process = FakeProcess()
        with mock.patch.object(manager.subprocess, 'Popen', return_value=process) as popen, mock.patch.object(manager, 'process_start_ticks', return_value=77):
            self.launcher.launch(self.attempt, 5)
        command = popen.call_args.args[0]
        self.assertEqual(command[:3], ['/python', '/guard.py', 'run'])
        self.assertIn('--exit-after-run', command)
        self.assertNotIn('--preempt-conflicts', command)
        self.assertEqual(command[command.index('--') + 1:command.index('--') + 4], ['/python', '/runner.py', 'controller'])
        self.assertEqual(self.attempt.phase, 'guard')
        self.assertEqual(self.attempt.start_ticks, 77)

    def test_normal_priority_extra_node_and_missing_cache_fail_closed(self):
        mutations = [dict(resource_priority='normal'), dict(preemptible=False),
                     dict(tasks=[{'nodes': ['n1']}, {'node_count': 1}]),
                     dict(model_cache={'policy': 'communication_only'}), dict(gpu_registry_dir='/other'),
                     dict(gpu_lease_ttl_s=14)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                manager.atomic_json(self.spec_path, {**self.spec, **mutation})
                self.output.seek(0)
                with mock.patch.object(manager.subprocess, 'Popen') as popen:
                    with self.assertRaises(ValueError):
                        self.launcher.launch(self.attempt, 5)
                    popen.assert_not_called()

    def test_stop_only_signals_matching_child_and_recovery_refuses_foreign_host(self):
        with mock.patch.object(manager, 'same_process', return_value=False), mock.patch.object(manager.os, 'kill') as kill:
            self.launcher.stop(self.attempt, 'stop', 5)
            kill.assert_not_called()
        foreign = {'manager_host': 'some-other-host', 'attempts': {'n1': {'pid': 1, 'start_ticks': 1}}}
        with mock.patch.object(manager.os, 'kill') as kill:
            with self.assertRaisesRegex(RuntimeError, 'another host'):
                manager.stop_previous(foreign, 1)
            kill.assert_not_called()


class CachedWeightTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {'cached_weights_only': True, 'cpu_cache_registry_dir': str(self.root / 'cache'),
                       'engine_root': '/engine', 'engine_commit': 'abc', 'model_path': '/model',
                       'known_node_specs': {'n1': {'cache_spec_template': {
                           'engine_root': '/engine', 'source_commit': 'abc', 'model_path': '/model',
                           'cache_scheme': 'canonical_full_weight', 'max_bytes': 999999999999,
                           'headroom_bytes': 128 * 2**30, 'socket_path': '/default.sock'}}}}
        self.booking = manager.Booking('n1', '10', 1000000, 'gres/gpu:A100:4')
        self.row = {'node': 'n1', 'state': 'active', 'uid': os.getuid(), 'source_commit': 'abc',
                    'cache_scheme': 'canonical_full_weight', 'expected_job_id': '10',
                    'max_bytes': 100, 'headroom_bytes': 32 * 2**30, 'socket_path': '/existing.sock',
                    'daemon_id': 'daemon', 'checked_at': 900,
                    'manifest': {'state': 'running', 'node': 'n1', 'cache_id': 'daemon',
                                 'pid': 123, 'process_start_ticks': 456},
                    'status': {'ok': True, 'source_commit': 'abc', 'max_bytes': 100,
                               'headroom_bytes': 32 * 2**30, 'uid': os.getuid(), 'pid': 123,
                               'start_ticks': 456, 'features': ['canonical_groups_v1'],
                               'used_bytes': 50, 'current_bytes': 50, 'reserved_bytes': 0,
                               'entries': [{'key': 'opaque-node-local-key', 'state': 'ready',
                                            'size': 50, 'evict_pending': False}]}}
        self.ledger = {'schema_version': 1, 'uid': os.getuid(), 'entries': [self.row]}

    def write_ledger(self):
        manager.atomic_json(self.root / 'cache/registry.json', self.ledger)

    def test_matching_existing_capacity_is_adopted_as_candidate_not_proven_hit(self):
        reason, evidence = manager.cached_weight_candidate(self.config, self.booking, self.ledger)
        self.assertIsNone(reason)
        self.assertEqual(evidence['max_bytes'], 100)
        self.assertEqual(evidence['headroom_bytes'], 32 * 2**30)
        self.assertEqual(evidence['socket_path'], '/existing.sock')
        self.assertFalse(evidence['model_cache_hit_proven'])

    def test_missing_or_different_source_scheme_job_skip_without_mutation(self):
        self.ledger['entries'] = []
        self.assertIn('no recorded', manager.cached_weight_candidate(self.config, self.booking, self.ledger)[0])
        self.ledger['entries'] = [self.row]
        for key in ('source_commit', 'cache_scheme', 'expected_job_id'):
            with self.subTest(key=key):
                original = self.row[key]
                self.row[key] = 'different foreground bank'
                before = json.dumps(self.ledger, sort_keys=True)
                self.assertIn(key, manager.cached_weight_candidate(self.config, self.booking, self.ledger)[0])
                self.assertEqual(json.dumps(self.ledger, sort_keys=True), before)
                self.row[key] = original

    def test_building_missing_and_untrusted_readiness_skip(self):
        for change, expected in ((lambda: self.row['status']['entries'][0].update(state='building'), 'building'),
                                 (lambda: self.row.update(status={}), 'identity'),
                                 (lambda: self.row.pop('status'), 'snapshot')):
            with self.subTest(expected=expected):
                saved = json.loads(json.dumps(self.row))
                change()
                self.assertIn(expected, manager.cached_weight_candidate(self.config, self.booking, self.ledger)[0])
                self.row.clear()
                self.row.update(saved)

    def test_unknown_registry_pauses_new_admission_but_keeps_active_attempt(self):
        self.write_ledger()
        launcher = FakeLauncher()
        service = manager.Reconciler(self.config, self.root, launcher, 1000)
        service.reconcile({'n1': self.booking}, {}, 1000)
        self.assertEqual(len(launcher.prepares), 1)
        # Simulate Guard holding a healthy active child before ledger damage.
        attempt = service.attempts['n1']
        attempt.phase = 'guard'
        (self.root / 'cache/registry.json').write_text('{bad')
        service.reconcile({'n1': self.booking}, {}, 1001)
        self.assertEqual(launcher.stops, [])
        self.assertIn('ledger unavailable', service.cache_selection_error)
        attempt.process.returncode = 2
        service.reconcile({'n1': self.booking}, {}, 1002)
        service.reconcile({'n1': self.booking}, {}, 2000)
        self.assertEqual(len(launcher.prepares), 1)
        self.assertIn('ledger unavailable', service.nodes['n1']['reason'])

    def test_metadata_missing_and_unknown_registry_fail_closed(self):
        with self.assertRaises(OSError):
            manager.cache_ledger(self.config)
        self.write_ledger()
        self.assertEqual(manager.cache_ledger(self.config)['entries'][0]['node'], 'n1')
        self.ledger['schema_version'] = 999
        self.write_ledger()
        with self.assertRaises(ValueError):
            manager.cache_ledger(self.config)

    def test_prepared_contract_requires_live_reuse_only(self):
        config = {**self.config, 'python': '/python', 'guard_script': '/guard', 'node_runner': '/runner'}
        spec_path = self.root / 'spec.json'
        spec = {'resource_priority': 'background', 'preemptible': True, 'tasks': [{'nodes': ['n1']}],
                'model_cache': {'policy': 'required', 'on_miss': 'build'}}
        manager.atomic_json(spec_path, spec)
        output = io.BytesIO(json.dumps({'spec_path': str(spec_path)}).encode())
        attempt = manager.Attempt(self.booking, self.root, FakeProcess(), 'preparing', 0, 1, io.BytesIO(), output)
        with mock.patch.object(manager.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(ValueError, 'exact live model cache miss'):
                manager.Launcher(config, Path('/profile')).launch(attempt, 1)
            popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
