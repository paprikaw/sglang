"""CPU-only checks for discarded benchmark output and strict loader evidence."""
import argparse
import contextlib
import errno
import importlib.util
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
import manager


ROOT=Path(__file__).resolve().parents[2]


def load_runner(kind):
    path=ROOT/f'experiments/cluster_backfill_20261020/node_runner_{kind}_ephemeral.py'
    spec=importlib.util.spec_from_file_location('ephemeral_'+kind,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNERS=[load_runner(kind) for kind in ('qwen','fp8','glm','cache_model')]


class EphemeralRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='ephemeral-runner-',dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def spawn(self):
        child=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(60)'],
                               start_new_session=True)
        def cleanup():
            if child.poll() is None:
                child.kill()
            child.wait()
        self.addCleanup(cleanup)
        return child

    def config(self):
        return dict(artifact_retention='none',server_python=sys.executable,
            benchmark=dict(artifact_retention='none',worker_script='/unused',config_path='/unused'))

    def test_operational_logs_use_devnull_without_creating_paths(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__):
                path=self.root/'no-directory'/'benchmark-worker.log'
                with runner.transient_log(self.config(),path) as stream:
                    self.assertEqual(stream,subprocess.DEVNULL)
                self.assertFalse(path.parent.exists())

    def test_incompatible_retention_rejected_before_log_creation(self):
        for runner in RUNNERS:
            for change in (dict(artifact_retention='retain'),
                           dict(benchmark=dict(artifact_retention='retain'))):
                with self.subTest(runner=runner.__name__,change=change):
                    cfg=self.config();cfg.update(change)
                    with self.assertRaises(ValueError):
                        with runner.transient_log(cfg,self.root/'worker.log'):
                            self.fail('incompatible retention yielded a log')
                    self.assertFalse((self.root/'worker.log').exists())

    def test_worker_spawn_discards_output_and_cleanup_still_stops_owned_process(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__):
                child=self.spawn()
                args=argparse.Namespace(run=self.root,node='n1')
                with mock.patch.object(runner,'lease_check'), \
                     mock.patch.object(runner,'benchmark_frozen_inputs'), \
                     mock.patch.object(runner.subprocess,'Popen',return_value=child) as spawn:
                    owned_child,owned,status=runner.start_benchmark_worker(self.config(),args,'/receipt')
                self.assertIs(owned_child,child)
                self.assertEqual(spawn.call_args.kwargs['stdout'],subprocess.DEVNULL)
                self.assertEqual(spawn.call_args.kwargs['env']['PYTHONDONTWRITEBYTECODE'],'1')
                self.assertFalse((self.root/'benchmark-worker.log').exists())
                with mock.patch.object(runner,'save',side_effect=OSError(errno.EDQUOT,'Disk quota exceeded')), \
                     contextlib.redirect_stderr(io.StringIO()):
                    self.assertTrue(runner.stop_benchmark_worker(child,owned,status,self.root))
                self.assertIsNotNone(child.poll())

    def test_initial_identity_write_failure_cannot_orphan_worker(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__):
                child=self.spawn()
                with mock.patch.object(runner,'lease_check'), \
                     mock.patch.object(runner,'benchmark_frozen_inputs'), \
                     mock.patch.object(runner.subprocess,'Popen',return_value=child), \
                     mock.patch.object(runner,'save',side_effect=OSError(errno.EDQUOT,'Disk quota exceeded')), \
                     contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(OSError):
                        runner.start_benchmark_worker(self.config(),argparse.Namespace(run=self.root,node='n1'),'/receipt')
                self.assertIsNotNone(child.poll())

    def test_loader_lines_preserved_byte_for_byte_including_bad_and_duplicate_records(self):
        records=[b'CPU weight cache load: {"rank":"TP0","source":"cpu_cache"}\n',
                 b'prefix CPU weight cache load: {"rank":"TP1","source":"disk_fallback"}\n',
                 b'CPU weight cache load: {malformed json}\n',
                 b'CPU weight cache load: {"rank":"TP0","source":"cpu_cache"}\n']
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__):
                path=self.root/(runner.__name__+'.log')
                data=b'ordinary prefill stats\n'+b''.join(records)+b'ordinary runtime error\n'
                drain=runner.AdmissionLogDrain(io.BytesIO(data),path)
                drain.finish()
                self.assertEqual(path.read_bytes(),b''.join(records))

    def test_overlong_loader_line_cannot_be_silently_truncated(self):
        for runner in RUNNERS:
            for prefix in (b'',b'x'*65525):
                with self.subTest(runner=runner.__name__,prefix_length=len(prefix)):
                    path=self.root/(runner.__name__+'long-'+str(len(prefix))+'.log')
                    line=prefix+b'CPU weight cache load: '+b'x'*100000+b'\n'
                    drain=runner.AdmissionLogDrain(io.BytesIO(line),path)
                    try:
                        drain.finish()
                    except RuntimeError:
                        continue  # Rejecting an oversized physical line is safe.
                    self.assertEqual(path.read_bytes(),line)

    def test_loader_evidence_overflow_is_a_failure_not_partial_success(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__):
                path=self.root/(runner.__name__+'overflow.log')
                line=b'CPU weight cache load: '+b'x'*(1024*1024)+b'\n'
                drain=runner.AdmissionLogDrain(io.BytesIO(line),path)
                with self.assertRaises(RuntimeError):
                    drain.finish()
                self.assertLessEqual(path.stat().st_size,1024*1024)

    def test_loader_evidence_write_failure_is_visible(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__):
                path=self.root/'missing-parent'/runner.__name__
                drain=runner.AdmissionLogDrain(io.BytesIO(b'CPU weight cache load: {}\n'),path)
                with self.assertRaisesRegex(RuntimeError,'Cache evidence write failed'):
                    drain.finish()

    def test_existing_evidence_counts_toward_the_same_bound(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__):
                path=self.root/(runner.__name__+'existing.log')
                original=b'x'*(1024*1024-4)
                path.write_bytes(original)
                drain=runner.AdmissionLogDrain(io.BytesIO(b'CPU weight cache load: {}\n'),path)
                with self.assertRaises(RuntimeError):
                    drain.finish()
                self.assertEqual(path.read_bytes(),original)

    def test_serving_output_volume_does_not_grow_evidence_log(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__):
                path=self.root/(runner.__name__+'noise.log')
                drain=runner.AdmissionLogDrain(io.BytesIO(b'Prefill stats normal output\n'*100000),path)
                drain.finish()
                self.assertEqual(path.stat().st_size,0)

    def test_incomplete_cache_evidence_fails_run_even_with_verified_gpu_release(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__):
                cfg=dict(artifact_retention='none',guard_script='/frozen/guard.py',api_port=1234)
                args=argparse.Namespace(run=self.root,config=self.root/'profile.json',node='n1',job_id='1')
                data={'frozen_profile.json':cfg,'deployment.json':{'deployment_id':'test'},
                      'attempt.json':{'stop_at':0}}
                remote=mock.Mock()
                remote.stdin.closed=False
                remote.poll.return_value=0
                visibility=types.SimpleNamespace(wait_for_stable_json=mock.Mock(return_value=(
                    dict(owned_processes_exited=True,gpu_idle=True,cache_evidence_complete=False),{})))
                with mock.patch.dict(os.environ,{'EXPERIMENT_GUARD_LEASE_RECEIPT':'/receipt',
                                                'EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS':'/bindings'}), \
                     mock.patch.dict(sys.modules,{'artifact_visibility':visibility}), \
                     mock.patch.object(runner,'read',side_effect=lambda p:data[Path(p).name]), \
                     mock.patch.object(runner,'record',return_value={'pid':100,'ticks':10}), \
                     mock.patch.object(runner,'lease_check',return_value=({},dict(owner_pid=100,owner_start_ticks=10))), \
                     mock.patch.object(runner.signal,'signal'), \
                     mock.patch.object(runner.subprocess,'Popen',return_value=remote), \
                     mock.patch.object(runner,'stop_controller_resources',return_value=True):
                    self.assertEqual(runner.controller(args),1)
                cleanup=json.loads((self.root/'task/cleanup.json').read_text())
                self.assertTrue(cleanup['passed'])
                self.assertTrue(cleanup['gpu_idle'])
                manifest=json.loads((self.root/'deployment.json').read_text())
                self.assertFalse(manifest['cache_evidence_complete'])

    def test_manager_retains_prepare_protocol_but_discards_child_log(self):
        cfg=dict(artifact_retention='none',node_runner='/runner.py')
        launcher=manager.Launcher(cfg,self.root/'profile.json')
        with mock.patch.object(manager.subprocess,'Popen',return_value=types.SimpleNamespace(pid=123)) as spawn, \
             mock.patch.object(manager,'process_start_ticks',return_value=456):
            attempt=launcher.prepare(manager.Booking('n1','1',time.time()+3600),self.root,time.time())
        try:
            self.assertEqual(attempt.log.name,'/dev/null')
            self.assertEqual(spawn.call_args.kwargs['stderr'].name,'/dev/null')
            self.assertTrue((self.root/'prepare.json.out').is_file())
            self.assertFalse((self.root/'manager-child.log').exists())
        finally:
            launcher.close(attempt)


class EphemeralProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='ephemeral-progress-',dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.run=self.root/'attempt'
        self.path=self.root/'prefill/nodes/n1.json'
        self.path.parent.mkdir(parents=True)
        self.workload=self.root/'workload.json'
        self.workload.write_text(json.dumps(dict(artifact_retention='none',runtime_root=str(self.root/'prefill'))))
        self.cfg=dict(benchmark=dict(config_path=str(self.workload)))
        self.now=time.time()
        self.raw=dict(storage_layout='ephemeral_live_v1',retention='none',node='n1',
            run=str(self.run.resolve()),timestamp=self.now,phase='live-round',status='running',
            offered_requests=16,dispatched_requests=12,completed_requests=8,
            successful_requests=7,inflight=4,queue_policy='observe_only',
            valid_so_far=True,completed_rps=1.2)

    def snapshot(self,runner):
        self.path.write_text(json.dumps(self.raw))
        with mock.patch.object(runner.time,'time',return_value=self.now):
            return runner.benchmark_progress_snapshot(self.cfg,self.run,'n1')

    def test_fresh_exact_run_live_counters_are_exposed_without_stage_artifacts(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__):
                result=self.snapshot(runner)
                self.assertEqual(result['benchmark_progress_path'],str(self.path))
                self.assertEqual(result['benchmark_phase'],'live-round')
                self.assertEqual(result['dispatched_requests'],12)
                self.assertEqual(result['failed_requests'],1)
                self.assertEqual(result['inflight_requests'],4)
                self.assertFalse((self.run/'benchmark_progress.json').exists())
                self.assertFalse((self.root/'prefill/stages').exists())

    def test_stale_future_wrong_run_and_wrong_node_rejected(self):
        baseline=dict(self.raw)
        for runner in RUNNERS:
            for change in (dict(timestamp=self.now-21),dict(timestamp=self.now+1),
                           dict(timestamp='bad'),dict(run=str(self.root/'old-attempt')),
                           dict(node='other'),dict(storage_layout='append_per_run_v1'),
                           dict(retention='retain'),dict(phase='../escape')):
                with self.subTest(runner=runner.__name__,change=change):
                    self.raw=dict(baseline,**change)
                    self.assertEqual(self.snapshot(runner),{})

    def test_invalid_counter_types_and_inconsistent_counts_rejected(self):
        baseline=dict(self.raw)
        for runner in RUNNERS:
            for change in (dict(offered_requests=True),dict(inflight=-1),dict(completed_requests='8'),
                           dict(successful_requests=9),dict(completed_requests=13),
                           dict(dispatched_requests=17),dict(inflight=13)):
                with self.subTest(runner=runner.__name__,change=change):
                    self.raw=dict(baseline,**change)
                    self.assertEqual(self.snapshot(runner),{})

    def test_symlink_and_oversized_live_snapshot_rejected(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__):
                self.path.unlink(missing_ok=True)
                target=self.root/'outside.json'
                target.write_text(json.dumps(self.raw))
                self.path.symlink_to(target)
                self.assertEqual(runner.benchmark_progress_snapshot(self.cfg,self.run,'n1'),{})
                self.path.unlink()
                self.path.write_text(' '*32769)
                self.assertEqual(runner.benchmark_progress_snapshot(self.cfg,self.run,'n1'),{})


if __name__=='__main__':
    unittest.main()
