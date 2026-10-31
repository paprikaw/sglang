"""GPU-free tests for authenticated idle deployment and owned-child cleanup."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import node_runner as runner


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'attempt'
        self.run.mkdir()
        self.registry = self.root / 'registry'
        self.registry.mkdir()
        self.profile = self.run / 'frozen_profile.json'
        runner.save(self.profile, {'model_path': '/model'})
        self.spec = {'id': 'idle-test', 'resource_priority': 'background',
                     'frozen_inputs': [{'path': str(self.profile), 'sha256': runner.sha(self.profile)}]}
        runner.save(self.run / 'dashboard_spec.json', self.spec)
        digest = runner.sha(self.run / 'dashboard_spec.json')
        self.receipt = {'registry_dir': str(self.registry), 'lease_id': 'lease',
                        'token': 'private-test-token', 'spec_sha256': digest}
        self.lease = {'lease_id': 'lease', 'state': 'active', 'nodes': ['n1'],
                      'resource_priority': 'background', 'spec_sha256': digest,
                      'heartbeat_at': time.time(), 'ttl_s': 15,
                      'token_sha256': hashlib.sha256(self.receipt['token'].encode()).hexdigest()}
        self.receipt_path = self.run / 'receipt.json'
        self.sync()

    def sync(self):
        runner.save(self.receipt_path, self.receipt)
        runner.save(self.registry / 'registry.json', {'leases': [self.lease]})

    def check(self):
        return runner.lease_check(self.receipt_path, self.run, 'n1')

    def test_private_exact_active_receipt_is_accepted(self):
        receipt, lease = self.check()
        self.assertEqual(receipt['lease_id'], 'lease')
        self.assertEqual(lease['nodes'], ['n1'])

    def test_wrong_token_is_rejected(self):
        self.receipt['token'] = 'wrong'
        self.sync()
        with self.assertRaisesRegex(RuntimeError, 'authentication'):
            self.check()

    def test_public_receipt_is_rejected(self):
        self.receipt_path.chmod(0o644)
        with self.assertRaisesRegex(RuntimeError, 'Private'):
            self.check()

    def test_extra_node_in_lease_is_rejected(self):
        self.lease['nodes'] = ['n1', 'n2']
        self.sync()
        with self.assertRaisesRegex(RuntimeError, 'Exact active node'):
            self.check()

    def test_expired_heartbeat_is_rejected(self):
        self.lease['heartbeat_at'] = time.time() - 16
        self.sync()
        with self.assertRaisesRegex(RuntimeError, 'expired'):
            self.check()

    def test_normal_demand_cancels_before_any_new_model_work(self):
        self.lease['preemption_requested'] = {'request_id': 'normal-request'}
        self.sync()
        with self.assertRaises(InterruptedError):
            self.check()

    def test_frozen_profile_tampering_is_rejected(self):
        runner.save(self.profile, {'model_path': '/different-model'})
        with self.assertRaisesRegex(RuntimeError, 'Frozen deployment input changed'):
            self.check()

    def test_dashboard_spec_tampering_is_rejected(self):
        runner.save(self.run / 'dashboard_spec.json', dict(self.spec, changed=True))
        with self.assertRaisesRegex(RuntimeError, 'spec changed'):
            self.check()


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config_path = self.root / 'config.json'
        self.commit = 'a' * 40
        self.config = {'engine_root': str(self.root / 'engine'), 'engine_commit': self.commit,
                       'source_tag': 'idle-test', 'gpu_registry_dir': str(self.root / 'registry'),
                       'model_path': '/model', 'model_revision': 'local-model', 'api_port': 30000,
                       'server_args': ['--model-path', '{model}', '--tp', '4'],
                       'default_node_spec': {'cache_spec_template': {
                           'engine_root': str(self.root / 'engine'), 'source_commit': self.commit,
                           'model_path': '/model', 'python': sys.executable}}}
        runner.save(self.config_path, self.config)
        self.args = argparse.Namespace(config=self.config_path, node='n1', job_id='123', run=self.root / 'run')

    def prepare(self):
        def git(argv, **kwargs):
            if 'status' in argv:
                return ''
            return self.commit + '\n'
        with mock.patch.object(runner.subprocess, 'check_output', side_effect=git), contextlib.redirect_stdout(io.StringIO()):
            runner.prepare(self.args)

    def test_profile_is_frozen_and_attempt_cannot_be_overwritten(self):
        self.prepare()
        original = runner.read(self.args.run / 'frozen_profile.json')
        self.config['api_port'] = 39999
        runner.save(self.config_path, self.config)
        self.assertEqual(runner.read(self.args.run / 'frozen_profile.json'), original)
        spec = runner.read(self.args.run / 'dashboard_spec.json')
        self.assertEqual(spec['resource_priority'], 'background')
        self.assertEqual(spec['tasks'][0]['nodes'], ['n1'])
        self.assertEqual(spec['model_cache']['hicache_reserve_bytes_per_node'], {'n1': 0})
        self.assertEqual(self.args.run.stat().st_mode & 0o777, 0o700)
        with self.assertRaisesRegex(RuntimeError, 'immutable'):
            self.prepare()

    def test_profile_for_another_booking_is_rejected(self):
        path = self.root / 'node.json'
        runner.save(path, {'node': 'n1', 'expected_job_id': '999'})
        self.config['known_node_specs'] = {'n1': {'cache_spec_path': str(path)}}
        runner.save(self.config_path, self.config)
        with self.assertRaisesRegex(RuntimeError, 'another node or booking'):
            self.prepare()
        self.assertFalse((self.args.run / 'dashboard_spec.json').exists())

    def test_cache_and_serving_source_must_match(self):
        self.config['default_node_spec']['cache_spec_template']['source_commit'] = 'b' * 40
        runner.save(self.config_path, self.config)
        with self.assertRaisesRegex(RuntimeError, 'serving profile differ'):
            self.prepare()
        self.assertFalse((self.args.run / 'dashboard_spec.json').exists())

    def test_hicache_requires_an_admitted_reserve(self):
        self.config['server_args'].append('--enable-hierarchical-cache')
        runner.save(self.config_path, self.config)
        with self.assertRaisesRegex(RuntimeError, 'zero HiCache reserve'):
            self.prepare()


class CachedOnlyPreparationTests(unittest.TestCase):
    """Existing weight ownership and budgets stay intact during selection."""
    prepare = PreparationTests.prepare

    def setUp(self):
        PreparationTests.setUp(self)
        self.config['cached_weights_only'] = True
        self.config['default_node_spec']['cache_spec_template'].update(
            max_bytes=256 * 2**30, headroom_bytes=32 * 2**30,
            socket_path='/tmp/default-daemon.sock', cache_scheme='canonical_full_weight')
        runner.save(self.config_path, self.config)
        self.ledger_path = Path(self.config['gpu_registry_dir']).parent / 'cpu_weight_daemons/registry.json'
        self.existing = {'node': 'n1', 'state': 'active', 'uid': os.getuid(),
                         'expected_job_id': '123', 'source_commit': self.commit,
                         'cache_scheme': 'canonical_full_weight', 'daemon_id': 'protected-existing',
                         'max_bytes': 512 * 2**30, 'headroom_bytes': 48 * 2**30,
                         'socket_path': '/tmp/protected-existing.sock',
                         'status': {'entries': [{'key': 'opaque-existing-group', 'state': 'ready'}]}}
        self.write_ledger([self.existing])

    def write_ledger(self, rows):
        runner.save(self.ledger_path, {'schema_version': 1, 'uid': os.getuid(), 'entries': rows})

    def test_no_daemon_or_missing_ledger_skips_without_creating_resources(self):
        self.write_ledger([])
        original = self.ledger_path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'No unique existing CPU weight daemon'):
            self.prepare()
        self.assertEqual(self.ledger_path.read_bytes(), original)
        self.assertFalse((self.args.run / 'dashboard_spec.json').exists())
        self.ledger_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse(self.ledger_path.exists())

    def test_incompatible_protected_daemon_is_skipped_without_mutation(self):
        for change in ({'source_commit': 'b' * 40}, {'expected_job_id': '999'},
                       {'cache_scheme': 'legacy-rank-snapshot'}, {'state': 'launching'}):
            with self.subTest(change=change):
                self.write_ledger([dict(self.existing, **change)])
                original = self.ledger_path.read_bytes()
                with self.assertRaisesRegex(RuntimeError, 'incompatible; preserve it and skip'):
                    self.prepare()
                self.assertEqual(self.ledger_path.read_bytes(), original)
                self.assertFalse((self.args.run / 'dashboard_spec.json').exists())

    def test_empty_or_incomplete_recorded_weight_bank_is_skipped(self):
        for entries in ([], [{'state': 'building'}], [{'state': 'ready'}, {'state': 'invalid'}]):
            with self.subTest(entries=entries):
                self.write_ledger([dict(self.existing, status={'entries': entries})])
                original = self.ledger_path.read_bytes()
                with self.assertRaisesRegex(RuntimeError, 'No complete recorded ready weight bank'):
                    self.prepare()
                self.assertEqual(self.ledger_path.read_bytes(), original)
                self.assertFalse((self.args.run / 'dashboard_spec.json').exists())

    def test_adopts_existing_admitted_budget_socket_and_fail_on_miss(self):
        original = self.ledger_path.read_bytes()
        self.prepare()
        node_spec = runner.read(self.args.run / 'cache_node_spec.json')
        for name in ('max_bytes', 'headroom_bytes', 'socket_path'):
            self.assertEqual(node_spec[name], self.existing[name])
        self.assertEqual(node_spec['node'], 'n1')
        self.assertEqual(node_spec['expected_job_id'], '123')
        spec = runner.read(self.args.run / 'dashboard_spec.json')
        self.assertEqual(spec['model_cache']['on_miss'], 'fail')
        self.assertEqual(spec['resource_priority'], 'background')
        self.assertEqual(spec['model_cache']['node_specs']['n1']['sha256'],
                         runner.sha(self.args.run / 'cache_node_spec.json'))
        self.assertEqual(self.ledger_path.read_bytes(), original)
        frozen = runner.read(self.args.run / 'frozen_profile.json')
        self.assertEqual(frozen['default_node_spec']['cache_spec_template']['max_bytes'], 256 * 2**30)

    def test_cached_only_does_not_include_dead_daemon_recovery(self):
        self.config['default_node_spec']['dead_daemon_recovery'] = {
            'daemon_id': self.existing['daemon_id'], 'manifest': '/missing/recovery-manifest'}
        runner.save(self.config_path, self.config)
        original = self.ledger_path.read_bytes()
        self.prepare()
        spec = runner.read(self.args.run / 'dashboard_spec.json')
        self.assertNotIn('dead_daemon_recovery', spec)
        self.assertFalse(any('dead_daemon_recovery.py' in row['path'] for row in spec['frozen_inputs']))
        self.assertEqual(self.ledger_path.read_bytes(), original)

    def test_ambiguous_duplicate_daemons_are_preserved_and_skipped(self):
        self.write_ledger([self.existing, dict(self.existing, daemon_id='other')])
        original = self.ledger_path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'No unique existing CPU weight daemon'):
            self.prepare()
        self.assertEqual(self.ledger_path.read_bytes(), original)


@contextlib.contextmanager
def fake_api(model='deployment-unique', valid_chat=True, done=True):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            body = {'data': [{'id': model}]} if self.path == '/v1/models' else {'status': 'ok'}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(body)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream' if body['stream'] else 'application/json')
            self.end_headers()
            if body['stream']:
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n')
                if done:
                    self.wfile.write(b'data: [DONE]\n\n')
            else:
                self.wfile.write(json.dumps({'choices': [{'message': {'content': 'OK'}}] if valid_chat else []}).encode())
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {'api_port': server.server_port, 'startup_timeout_s': 3}, calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


class ProbeEvent:
    def __init__(self, monitor_once=False):
        self.monitor_once = monitor_once
        self.monitor_checks = 0
    def is_set(self):
        return False
    def wait(self, seconds):
        if seconds == 10:
            self.monitor_checks += 1
            return self.monitor_checks > int(self.monitor_once)
        return False


class ProbeTests(unittest.TestCase):
    def test_foreign_endpoint_is_rejected_without_chat_requests(self):
        with fake_api(model='foreign-model') as (config, calls):
            result = {}
            runner.probe(config, ProbeEvent(), result, 'deployment-unique')
        self.assertFalse(result.get('ready', False))
        self.assertTrue(result['startup_failed'])
        self.assertFalse(any(isinstance(call, dict) for call in calls))

    def test_matching_endpoint_requires_nonstream_and_complete_stream(self):
        with fake_api() as (config, calls):
            result = {}
            runner.probe(config, ProbeEvent(), result, 'deployment-unique')
        self.assertTrue(result['ready'])
        self.assertTrue(result['validation']['non_stream_chat'])
        self.assertTrue(result['validation']['stream_chat'])
        self.assertFalse(result['validation']['performance_measurement'])
        self.assertEqual(len([call for call in calls if isinstance(call, dict)]), 2)

    def test_health_success_cannot_override_failed_startup_chat(self):
        for kwargs in ({'valid_chat': False}, {'done': False}):
            with self.subTest(**kwargs), fake_api(**kwargs) as (config, calls):
                result = {}
                runner.probe(config, ProbeEvent(monitor_once=True), result, 'deployment-unique')
            self.assertFalse(result.get('ready', False))
            self.assertTrue(result.get('startup_failed'))


class OwnedProcessTests(unittest.TestCase):
    def test_reused_pid_is_never_signalled(self):
        with mock.patch.object(runner, 'record', return_value={'ticks': 99, 'state': 'S'}), \
             mock.patch.object(runner, 'track_owned'), mock.patch.object(runner.os, 'kill') as kill:
            self.assertTrue(runner.stop_owned(None, {1234567: 42}))
        kill.assert_not_called()

    def test_model_tree_cleanup_does_not_signal_unrelated_sibling(self):
        sleeper = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
        def cleanup_sleeper():
            if sleeper.poll() is None:
                sleeper.kill()
            sleeper.wait(timeout=3)
        self.addCleanup(cleanup_sleeper)
        code = r'''
import ctypes,json,os,subprocess,sys,time
sys.path.insert(0,sys.argv[1])
import node_runner as r
assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0
child_code = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True);time.sleep(60)"
child=subprocess.Popen([sys.executable,'-c',child_code])
owned={child.pid:r.record(child.pid)['ticks']}
deadline=time.monotonic()+3
while len(owned)<2 and time.monotonic()<deadline:
 r.track_owned(owned);time.sleep(.02)
assert len(owned)>=2,owned
safe=r.stop_owned(child,owned)
print(json.dumps({'safe':safe,'identities':owned,'remaining':r.live_owned(owned)}))
'''
        result = subprocess.run([sys.executable, '-c', code, str(Path(runner.__file__).parent)],
                                capture_output=True, text=True, timeout=15, check=True)
        evidence = json.loads(result.stdout)
        self.assertTrue(evidence['safe'])
        self.assertEqual(evidence['remaining'], [])
        self.assertNotIn(str(sleeper.pid), evidence['identities'])
        self.assertIsNone(sleeper.poll())
        sleeper.terminate()
        sleeper.wait(timeout=3)


class RemoteLifecycleTests(unittest.TestCase):
    """Remote supervision stays cancellable while readiness is still pending."""
    def run_remote(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            cfg = {'guard_script': '/tmp/guard/dashboard_guard.py', 'engine_root': '/tmp/engine',
                   'server_python': sys.executable, 'server_env': {}, 'api_port': 30000,
                   'model_path': '/model', 'server_args': ['--model-path', '{model}', '--port', '{port}']}
            runner.save(run / 'frozen_profile.json', cfg)
            runner.save(run / 'deployment.json', {'deployment_id': 'unique-id'})
            runner.save(run / 'attempt.json', {'stop_at': time.time() + (-1 if mode == 'expired' else 60)})
            if mode == 'stop-file':
                (run / 'STOP').touch()
            args = argparse.Namespace(run=run, config=run / 'ignored.json', node='n1', job_id='123')
            receipt, lease = {'lease_id': 'lease'}, {'lease_id': 'lease'}
            checks = ([(receipt, lease), (receipt, lease), InterruptedError('normal demand')] if mode == 'preempt' else
                      [(receipt, lease), InterruptedError('normal demand')] if mode == 'preempt-before' else None)
            child = mock.Mock(pid=1234567)
            child.poll.return_value = None
            poller = mock.Mock()
            poller.poll.side_effect = lambda timeout: int(timeout != 0 or mode == 'eof-before')
            class Input(io.StringIO):
                def fileno(self):
                    return 123
            incoming = Input(json.dumps({'receipt_path': '/private/receipt', 'bindings_path': '/private/bindings'}) + '\n')
            module = types.ModuleType('cpu_model_cache_daemon')
            module._job_cgroup = mock.Mock(return_value='/job_123')
            with mock.patch.dict(sys.modules, {'cpu_model_cache_daemon': module}), \
                 mock.patch.object(runner.sys, 'stdin', incoming), \
                 mock.patch.object(runner.socket, 'gethostname', return_value='n1'), \
                 mock.patch.object(runner, 'lease_check', side_effect=checks, return_value=(receipt, lease)), \
                 mock.patch.object(runner.ctypes, 'CDLL') as libc, \
                 mock.patch.object(runner.signal, 'signal'), \
                 mock.patch.object(runner.select, 'poll', return_value=poller), \
                 mock.patch.object(runner.subprocess, 'Popen', return_value=child) as spawn, \
                 mock.patch.object(runner, 'record', return_value={'ticks': 42}), \
                 mock.patch.object(runner, 'track_owned'), \
                 mock.patch.object(runner, 'stop_owned', return_value=True) as stop, \
                 mock.patch.object(runner, 'gpu_cleanup', return_value={'gpu_idle': True}), \
                 mock.patch.object(runner.threading, 'Thread'), \
                 mock.patch.object(runner.os, 'read', return_value=b''):
                libc.return_value.prctl.return_value = 0
                code = runner.remote(args)
            self.assertEqual(code, 0)
            expected_spawn = mode in ('preempt', 'eof')
            stop.assert_called_once_with(child if expected_spawn else None, {child.pid: 42} if expected_spawn else {})
            evidence = runner.read(run / 'remote_cleanup.json')
            self.assertTrue(evidence['owned_processes_exited'])
            self.assertTrue(evidence['gpu_idle'])
            self.assertEqual(evidence['lease_id'], 'lease')
            if expected_spawn:
                argv = spawn.call_args.args[0]
                self.assertEqual(argv[-2:], ['--served-model-name', 'unique-id'])
            else:
                spawn.assert_not_called()
            return evidence

    def test_cancellation_or_deadline_before_spawn_starts_no_model(self):
        for mode in ('expired', 'stop-file', 'eof-before', 'preempt-before'):
            with self.subTest(mode=mode):
                evidence = self.run_remote(mode)
                self.assertEqual(evidence['reason'], 'preempted')
                self.assertEqual(evidence['error']['type'], 'InterruptedError')

    def test_preemption_during_startup_cleans_owned_child(self):
        evidence = self.run_remote('preempt')
        self.assertEqual(evidence['reason'], 'preempted')
        self.assertEqual(evidence['error']['type'], 'InterruptedError')

    def test_controller_eof_during_startup_cleans_owned_child(self):
        evidence = self.run_remote('eof')
        self.assertEqual(evidence['reason'], 'controller_eof')


if __name__ == '__main__':
    unittest.main()
