import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import cluster_backfill as c
from manager import Booking

class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.book=Booking('a','1',time.time()+3600,'gres/gpu:A100:4')
        self.recipe=dict(id='small',fallback=True,engine_commit='good',cache_keys=['x'],node_cache_keys={'a':['x']})
        self.bank=dict(state='active',expected_job_id='1',source_commit='good',cache_scheme='canonical_full_weight',status=dict(entries=[dict(key='x',state='ready')]))
    def test_empty_new_node_uses_explicit_fallback(self):
        self.assertEqual(c.select_recipe(self.book,None,[self.recipe])[0],self.recipe)
    def test_model_changes_choose_current_ready_keys(self):
        other=dict(self.recipe,id='other',node_cache_keys={'a':['y']})
        self.bank['status']['entries'][0]['key']='y'
        self.assertEqual(c.select_recipe(self.book,self.bank,[self.recipe,other])[0]['id'],'other')
    def test_prepared_partial_bank_never_loads_partial_model(self):
        self.bank['cache_scheme']='layer_window_weight'
        self.assertIn('complete_qualified_group',c.select_recipe(self.book,self.bank,[self.recipe])[1])
    def test_old_booking_not_retired_implicitly(self):
        self.bank['expected_job_id']='old'
        self.assertIn('retirement',c.select_recipe(self.book,self.bank,[self.recipe])[1])
    def test_unknown_source_not_inherited(self):
        self.bank['source_commit']='other'
        self.assertIsNone(c.select_recipe(self.book,self.bank,[self.recipe])[0])
    def test_uncached_key_incomplete_and_building_fail(self):
        for entries in [[dict(key='wrong',state='ready')],[dict(key='x',state='building')]]:
            self.bank['status']['entries']=entries
            self.assertIsNone(c.select_recipe(self.book,self.bank,[self.recipe])[0])
    def test_pid_reuse_never_signals(self):
        with patch.object(c,'live_manager',return_value=False),patch.object(c.os,'kill')as kill:
            c.stop_manager(dict(root='/tmp/absent'));kill.assert_not_called()
    def test_foreground_priority_over_old_background_worker(self):
        lease=dict(state='active',nodes=['a'],heartbeat_at=time.time(),ttl_s=15,resource_priority='normal')
        self.assertEqual(c.lease_state('a',dict(leases=[lease]),{},time.time())[0],'foreground')
    def test_background_heartbeat_is_not_proof_of_benchmark(self):
        lease=dict(state='active',nodes=['a'],heartbeat_at=time.time(),ttl_s=15,resource_priority='background')
        self.assertEqual(c.lease_state('a',dict(leases=[lease]),{},time.time())[0],'background_unverified')
    def test_expired_records_allow_recovery_audit(self):
        for state in ['active','quarantined']:
            lease=dict(state=state,nodes=['a'],heartbeat_at=0,ttl_s=15)
            self.assertEqual(c.lease_state('a',dict(leases=[lease]),{},time.time()),(None,None,None))
    def test_quarantine_within_cleanup_grace_blocks(self):
        lease=dict(state='quarantined',nodes=['a'],heartbeat_at=time.time(),ttl_s=15)
        self.assertEqual(c.lease_state('a',dict(leases=[lease]),{},time.time())[0],'draining')
    def test_expired_record_does_not_overlap_live_foreground(self):
        old=dict(state='active',nodes=['a'],heartbeat_at=0,ttl_s=15)
        live=dict(old,heartbeat_at=time.time(),resource_priority='normal')
        self.assertEqual(c.lease_state('a',dict(leases=[old,live]),{},time.time())[0],'foreground')
    def test_expired_lease_with_gpu_process_never_launches(self):
        old=dict(state='active',nodes=['a'],heartbeat_at=0,ttl_s=15)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(c,'discover_bookings',return_value={'a':self.book}), \
             patch.object(c,'registry_blocks',return_value={}), \
             patch.object(c,'read_json',side_effect=[dict(leases=[old]),dict(entries=[])]), \
             patch.object(c.subprocess,'check_output',return_value='Gres=gpu:A100:4'), \
             patch.object(c.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='12345\n')), \
             patch.object(c.subprocess,'Popen') as spawn:
            result=c.reconcile(dict(gpu_registry_dir='/tmp/test-gpu-registry'),Path(tmp),{},[self.recipe])
            self.assertEqual(result['nodes']['a']['reason'],'unregistered_gpu_work_or_inventory_unavailable')
            spawn.assert_not_called()
    def test_storage_quota_failure_pauses_new_admission_and_retries(self):
        cfg={'recipes':[]}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(c,'atomic_json',side_effect=[OSError(122,'Disk quota exceeded'),None,None]) as save, \
             patch.object(c,'reconcile',return_value={'timestamp':123}) as reconcile:
            with self.assertRaises(OSError):c.persist_reconciliation(cfg,Path(tmp),{})
            reconcile.assert_not_called()
            self.assertEqual(c.persist_reconciliation(cfg,Path(tmp),{}),{'timestamp':123})
            reconcile.assert_called_once()
    def test_low_inode_headroom_drains_only_owned_managers(self):
        services={'a':dict(root='/tmp/owned-manager')}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(c.os,'statvfs',return_value=SimpleNamespace(f_favail=512)), \
             patch.object(c,'atomic_json'), \
             patch.object(c,'stop_manager') as stop, \
             patch.object(c,'reconcile') as reconcile:
            result=c.persist_reconciliation(dict(recipes=[],min_free_inodes=1024),Path(tmp),services)
            self.assertEqual(result['state'],'storage_paused')
            stop.assert_called_once_with(services['a'])
            reconcile.assert_not_called()

    def test_new_node_inventory_uses_absolute_or_explicit_slurm_path(self):
        for configured in (None,'/custom/slurm/scontrol'):
            cfg=dict(gpu_registry_dir='/tmp/test-registry',node_runner='/runner',manager_script='/manager')
            if configured:cfg['scontrol']=configured
            with self.subTest(scontrol=configured),tempfile.TemporaryDirectory() as tmp, \
                 patch.dict(c.os.environ,{'PATH':'/usr/bin:/bin'}), \
                 patch.object(c,'discover_bookings',return_value={'a':self.book}), \
                 patch.object(c,'registry_blocks',return_value={}), \
                 patch.object(c,'read_json',side_effect=[dict(leases=[]),dict(entries=[])]), \
                 patch.object(c,'make_profile',return_value={'python':'/python'}), \
                 patch.object(c.subprocess,'check_output',return_value='Gres=gpu:A100:4') as inventory, \
                 patch.object(c.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='')), \
                 patch.object(c.subprocess,'Popen',return_value=SimpleNamespace(pid=123)):
                result=c.reconcile(cfg,Path(tmp),{},[self.recipe])
                self.assertEqual(result['nodes']['a']['state'],'starting')
                self.assertEqual(inventory.call_args.args[0][0],configured or '/apps/slurm/latest/bin/scontrol')

    def test_one_hardware_failure_keeps_other_live_benchmark_visible(self):
        other=Booking('b','2',time.time()+3600,'gres/gpu:A100:4')
        lease=dict(experiment_id='live-b')
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(c,'discover_bookings',return_value={'a':self.book,'b':other}), \
             patch.object(c,'registry_blocks',return_value={}), \
             patch.object(c,'read_json',side_effect=[dict(leases=[]),dict(entries=[])]), \
             patch.object(c,'lease_state',side_effect=[(None,None,None),('benchmark','verified_live_worker',lease)]), \
             patch.object(c.subprocess,'check_output',side_effect=FileNotFoundError('scontrol')), \
             patch.object(c.subprocess,'Popen') as spawn:
            result=c.reconcile(dict(gpu_registry_dir='/tmp/test-registry'),Path(tmp),{},[self.recipe])
            self.assertEqual(result['nodes']['a']['reason'],'hardware_inventory_unavailable')
            self.assertEqual(result['nodes']['b']['state'],'benchmark')
            spawn.assert_not_called()

    def test_one_cache_failure_keeps_other_live_benchmark_visible(self):
        other=Booking('b','2',time.time()+3600,'gres/gpu:A100:4')
        lease=dict(experiment_id='live-b')
        bank=dict(self.bank,node='a')
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(c,'discover_bookings',return_value={'a':self.book,'b':other}), \
             patch.object(c,'registry_blocks',return_value={}), \
             patch.object(c,'read_json',side_effect=[dict(leases=[]),dict(entries=[bank])]), \
             patch.object(c,'lease_state',side_effect=[(None,None,None),('benchmark','verified_live_worker',lease)]), \
             patch.object(c,'discover_recipe_keys',side_effect=TimeoutError('unreachable daemon')), \
             patch.object(c.subprocess,'Popen') as spawn:
            result=c.reconcile(dict(gpu_registry_dir='/tmp/test-registry'),Path(tmp),{},[self.recipe])
            self.assertEqual(result['nodes']['a']['reason'],'cache_inventory_unavailable')
            self.assertEqual(result['nodes']['b']['state'],'benchmark')
            spawn.assert_not_called()

    def test_one_gpu_probe_timeout_keeps_other_live_benchmark_visible(self):
        other=Booking('b','2',time.time()+3600,'gres/gpu:A100:4')
        lease=dict(experiment_id='live-b')
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(c,'discover_bookings',return_value={'a':self.book,'b':other}), \
             patch.object(c,'registry_blocks',return_value={}), \
             patch.object(c,'read_json',side_effect=[dict(leases=[]),dict(entries=[])]), \
             patch.object(c,'lease_state',side_effect=[(None,None,None),('benchmark','verified_live_worker',lease)]), \
             patch.object(c.subprocess,'check_output',return_value='Gres=gpu:A100:4'), \
             patch.object(c.subprocess,'run',side_effect=c.subprocess.TimeoutExpired('ssh',15)), \
             patch.object(c.subprocess,'Popen') as spawn:
            result=c.reconcile(dict(gpu_registry_dir='/tmp/test-registry'),Path(tmp),{},[self.recipe])
            self.assertEqual(result['nodes']['a']['reason'],'unregistered_gpu_work_or_inventory_unavailable')
            self.assertEqual(result['nodes']['b']['state'],'benchmark')
            spawn.assert_not_called()


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.workload=self.root/'source-benchmark.json'
        self.workload.write_text(json.dumps(dict(runtime_root='/old/prefill',
            storage_layout='append_per_run_v1',nodes=['old'])))
        self.template=self.root/'source-profile.json'
        self.template.write_text(json.dumps(dict(runtime_root='/old',
            benchmark=dict(config_path=str(self.workload),worker_script='/old/worker.py'))))
        self.recipe=dict(profile=str(self.template),profile_sha256=c.digest(self.template),
                         node_template={})
        self.book=Booking('a','1',time.time()+3600,'gres/gpu:A100:4')
        self.target=self.root/'new-service'
        self.cfg=dict(node_runner='/qualified/runner.py')

    def test_default_never_archives_background_results(self):
        original=self.template.read_bytes(),self.workload.read_bytes()
        profile=c.make_profile(self.cfg,self.recipe,self.book,None,self.target)
        workload=json.loads((self.target/'benchmark.json').read_text())
        self.assertEqual(profile['artifact_retention'],'none')
        self.assertEqual(profile['benchmark']['artifact_retention'],'none')
        self.assertEqual(profile['benchmark']['worker_script'],str(c.EPHEMERAL_WORKER))
        self.assertEqual(workload['artifact_retention'],'none')
        self.assertEqual(workload['storage_layout'],'ephemeral_live_v1')
        self.assertEqual(workload['runtime_root'],str(self.target/'prefill'))
        self.assertEqual(workload['nodes'],['a'])
        self.assertEqual(original,(self.template.read_bytes(),self.workload.read_bytes()))

    def test_retain_is_explicit_and_preserves_legacy_worker(self):
        self.cfg['artifact_retention']='retain'
        profile=c.make_profile(self.cfg,self.recipe,self.book,None,self.target)
        workload=json.loads((self.target/'benchmark.json').read_text())
        self.assertEqual(profile['benchmark']['worker_script'],'/old/worker.py')
        self.assertEqual(workload['artifact_retention'],'retain')
        self.assertEqual(workload['storage_layout'],'append_per_run_v1')

    def test_invalid_retention_fails_before_discovery_or_writes(self):
        for value in ('discard',None,True,0,[],{}):
            with self.subTest(value=value),patch.object(c,'discover_bookings') as discover:
                with self.assertRaisesRegex(ValueError,'artifact_retention'):
                    c.reconcile(dict(self.cfg,artifact_retention=value),self.target,{},[])
                discover.assert_not_called()
                self.assertFalse(self.target.exists())

    def test_append_worker_cannot_silently_ignore_no_retention(self):
        self.cfg['benchmark_worker_script']=str(c.EPHEMERAL_WORKER.with_name('prefill_worker_append.py'))
        with self.assertRaisesRegex(ValueError,'ephemeral prefill worker'):
            c.make_profile(self.cfg,self.recipe,self.book,None,self.target)
        self.assertFalse(self.target.exists())

    def test_manager_output_does_not_create_a_project_log_by_default(self):
        recipe=dict(id='small',fallback=True)
        cfg=dict(self.cfg,gpu_registry_dir='/tmp/test-registry',manager_script='/manager.py')
        with patch.object(c,'discover_bookings',return_value={'a':self.book}), \
             patch.object(c,'registry_blocks',return_value={}), \
             patch.object(c,'read_json',side_effect=[dict(leases=[]),dict(entries=[])]), \
             patch.object(c,'make_profile',return_value={'python':'/python'}), \
             patch.object(c.subprocess,'check_output',return_value='Gres=gpu:A100:4'), \
             patch.object(c.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='')), \
             patch.object(c.subprocess,'Popen',return_value=SimpleNamespace(pid=123)) as spawn:
            state=c.reconcile(cfg,self.root,{},[recipe])
        self.assertEqual(state['nodes']['a']['state'],'starting')
        self.assertEqual(spawn.call_args.kwargs['stdout'].name,'/dev/null')
        self.assertEqual(list((self.root/'services').glob('*/manager.log')),[])


class FailedWorkerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.service=self.root/'service'
        self.run=self.service/'runs/current'
        self.run.mkdir(parents=True)
        self.now=time.time()
        self.book=Booking('a','1',self.now+3600,'gres/gpu:A100:4')
        self.lease=dict(state='active',nodes=['a'],heartbeat_at=self.now,ttl_s=15,
            resource_priority='background',preemptible=True,experiment_id='idle-a-current')
        self.services={'a':dict(root=str(self.service),recipe_id='small')}
        self.write(self.service/'state.json',dict(attempts={'a':dict(run=str(self.run))}))
        self.write(self.run/'deployment.json',dict(node='a',deployment_id='idle-a-current',status='healthy'))
        self.write(self.service/'prefill/nodes/a.json',dict(node='a',run=str(self.run),timestamp=self.now,status='running'))

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def state(self):
        return c.lease_state('a',dict(leases=[self.lease]),self.services,self.now)

    def test_terminal_control_wins_over_fresh_running_snapshot(self):
        for state,code in [('failed',-9),('cleanup_unverified',None),('exited',1)]:
            with self.subTest(state=state):
                self.write(self.run/'benchmark_worker.json',dict(node='a',state=state,exit_code=code))
                result=self.state()
                self.assertEqual(result[0],'failed')
                self.assertIn('benchmark_worker_',result[1])

    def test_ready_model_with_stale_or_absent_worker_is_not_loading(self):
        for value in ({},dict(run=str(self.run),timestamp=self.now-120,status='running')):
            with self.subTest(worker=value):
                self.write(self.service/'prefill/nodes/a.json',value)
                self.assertEqual(self.state()[:2],('background_unverified','benchmark_worker_not_current'))

    def test_true_model_startup_stays_loading(self):
        self.write(self.run/'deployment.json',dict(node='a',deployment_id='idle-a-current',status='queued'))
        self.write(self.service/'prefill/nodes/a.json',{})
        self.assertEqual(self.state()[0],'loading')

    def test_foreground_priority_never_uses_old_failed_worker(self):
        self.lease['resource_priority']='normal'
        self.write(self.run/'benchmark_worker.json',dict(node='a',state='failed',exit_code=-9))
        self.assertEqual(self.state()[0],'foreground')

    def test_other_run_node_or_deployment_cannot_authorize_cleanup(self):
        for node,ident in [('b','idle-a-current'),('a','foreign')]:
            with self.subTest(node=node,ident=ident):
                self.write(self.run/'deployment.json',dict(node=node,deployment_id=ident,status='healthy'))
                self.assertEqual(self.state()[0],'background_unverified')
                self.assertEqual(c.matching_service('a',self.lease,self.services),(None,None,None))

    def reconcile(self):
        def load(path):
            if str(path)=='/tmp/test-registry/registry.json':return dict(leases=[self.lease])
            if str(path)=='/tmp/cpu_weight_daemons/registry.json':return dict(entries=[])
            return json.loads(Path(path).read_text())
        with patch.object(c,'discover_bookings',return_value={'a':self.book}), \
             patch.object(c,'registry_blocks',return_value={}), \
             patch.object(c,'read_json',side_effect=load), \
             patch.object(c,'stop_manager') as stop, \
             patch.object(c.subprocess,'Popen') as spawn:
            result=c.reconcile(dict(gpu_registry_dir='/tmp/test-registry'),self.root,self.services,[])
        spawn.assert_not_called()
        return result,stop

    def test_failed_owned_background_manager_is_stopped_without_new_spawn(self):
        self.write(self.run/'benchmark_worker.json',dict(node='a',state='failed',exit_code=-9))
        result,stop=self.reconcile()
        self.assertEqual(result['nodes']['a']['state'],'failed')
        stop.assert_called_once_with(self.services['a'])

    def test_foreground_or_nonpreemptible_manager_is_not_stopped(self):
        self.write(self.run/'benchmark_worker.json',dict(node='a',state='failed',exit_code=-9))
        for priority,preemptible in [('normal',True),('background',False)]:
            with self.subTest(priority=priority,preemptible=preemptible):
                self.lease.update(resource_priority=priority,preemptible=preemptible)
                _,stop=self.reconcile()
                stop.assert_not_called()

if __name__=='__main__':unittest.main()

class OldBookingRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.book=Booking('a','2',time.time()+3600,'gres/gpu:A100:4')
        self.bank=dict(node='a',uid=c.os.getuid(),state='active',expected_job_id='1',
                       cache_scheme='canonical_full_weight',daemon_id='a'*32,
                       socket_path='/tmp/cache.sock',manifest=dict(node='a',cache_id='a'*32,
                       pid=123,process_start_ticks=456))
        self.recipe=dict(id='small',fallback=True,node_runner=str(Path(c.__file__).resolve().parents[2]/'experiments/cluster_backfill_20261020/node_runner_qwen_ephemeral.py'))
    def probe(self,**changes):
        proof=dict(node='a',job='2',daemon_id='a'*32,dead=True,socket_absent=True)
        proof.update(changes)
        return patch.object(c.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(proof)))
    def test_dead_identity_returns_exact_leased_recovery_snapshot(self):
        with self.probe():
            recipe,why=c.old_booking_recovery(self.book,self.bank,[self.recipe])
        self.assertEqual(recipe['dead_daemon_recovery'],dict(node='a',daemon_id='a'*32,manifest=self.bank['manifest']))
        self.assertIn('-recover-2-',recipe['id'])
        self.assertEqual(why,'recover_verified_dead_booking_cache')
        self.assertNotIn('dead_daemon_recovery',self.recipe)
    def test_live_identity_or_new_socket_never_retires(self):
        for changes in [dict(dead=False),dict(socket_absent=False)]:
            with self.subTest(changes=changes),self.probe(**changes):
                self.assertIsNone(c.old_booking_recovery(self.book,self.bank,[self.recipe])[0])
    def test_partial_bank_and_unknown_runner_remain_protected(self):
        for bank,recipe in [(dict(self.bank,cache_scheme='partial'),self.recipe),
                            (self.bank,dict(self.recipe,node_runner='/unknown'))]:
            with patch.object(c.subprocess,'run') as probe:
                self.assertIsNone(c.old_booking_recovery(self.book,bank,[recipe])[0])
                probe.assert_not_called()
    def test_unreachable_and_mismatching_proofs_fail_closed(self):
        with patch.object(c.subprocess,'run',side_effect=c.subprocess.TimeoutExpired('ssh',20)):
            with self.assertRaises(c.subprocess.TimeoutExpired):
                c.old_booking_recovery(self.book,self.bank,[self.recipe])
        with self.probe(daemon_id='b'*32):
            with self.assertRaises(ValueError):c.old_booking_recovery(self.book,self.bank,[self.recipe])
    def test_recovery_profile_is_build_under_new_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);template=root/'template.json';bench=root/'bench.json'
            bench.write_text('{}')
            template.write_text(json.dumps(dict(benchmark=dict(config_path=str(bench)))))
            recipe=dict(self.recipe,profile=str(template),profile_sha256=c.digest(template),
                        node_template={},dead_daemon_recovery=dict(node='a',daemon_id='a'*32,manifest=self.bank['manifest']))
            profile=c.make_profile(dict(node_runner='/runner'),recipe,self.book,None,root)
            self.assertFalse(profile['cached_weights_only'])
            self.assertEqual(profile['model_cache_on_miss'],'build')
            ns=profile['known_node_specs']['a']
            self.assertEqual(ns['cache_spec_template']['expected_job_id'],'2')
            self.assertEqual(ns['dead_daemon_recovery'],recipe['dead_daemon_recovery'])
    def test_foreground_lease_precedes_old_bank_recovery(self):
        lease=dict(state='active',nodes=['a'],heartbeat_at=time.time(),ttl_s=15,
                   resource_priority='normal',experiment_id='foreground')
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(c,'discover_bookings',return_value={'a':self.book}), \
             patch.object(c,'registry_blocks',return_value={}), \
             patch.object(c,'read_json',side_effect=[dict(leases=[lease]),dict(entries=[self.bank])]), \
             patch.object(c,'old_booking_recovery') as recovery, \
             patch.object(c.subprocess,'Popen') as spawn:
            state=c.reconcile(dict(gpu_registry_dir='/tmp/registry'),Path(tmp),{},[self.recipe])
            self.assertEqual(state['nodes']['a']['state'],'foreground')
            recovery.assert_not_called();spawn.assert_not_called()
    def test_reconcile_never_adopts_old_bank_as_current(self):
        recovery=dict(self.recipe,id='small-recover-2',dead_daemon_recovery={})
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(c,'discover_bookings',return_value={'a':self.book}), \
             patch.object(c,'registry_blocks',return_value={}), \
             patch.object(c,'read_json',side_effect=[dict(leases=[]),dict(entries=[self.bank])]), \
             patch.object(c,'old_booking_recovery',return_value=(recovery,'recover_verified_dead_booking_cache')), \
             patch.object(c.subprocess,'check_output',return_value='gpu:A100:4'), \
             patch.object(c.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='')), \
             patch.object(c,'make_profile',return_value={'python':'/python'}) as profile, \
             patch.object(c.subprocess,'Popen',return_value=SimpleNamespace(pid=123)):
            state=c.reconcile(dict(gpu_registry_dir='/tmp/registry',manager_script='/manager'),Path(tmp),{},[self.recipe])
            self.assertEqual(state['nodes']['a']['state'],'starting')
            self.assertIsNone(profile.call_args.args[3])

class HardwareQualificationTests(unittest.TestCase):
    def test_h100_requires_explicit_loader_support(self):
        info='NodeName=x Gres=gpu:H100:4(S:0-1) CfgTRES=gres/gpu=4,gres/gpu:h100=4'
        self.assertIsNone(c.recipe_hardware({},info))
        hw=c.recipe_hardware(dict(qualified_gpu_types=['A100','H100']),info)
        self.assertEqual(hw['gpu_type'],'H100');self.assertEqual(hw['gpu_count'],4)
    def test_legacy_a100_profile_keeps_qualification(self):
        self.assertEqual(c.recipe_hardware({},'Gres=gpu:A100:4')['gpu_type'],'A100')
    def test_unknown_untyped_and_wrong_size_are_rejected(self):
        recipe=dict(qualified_gpu_types=['A100','H100'])
        for info in ['Gres=gpu:H200:4','Gres=gpu:4','Gres=gpu:H100:8','Gres=gpu:H100:40',
                     'Gres=gpu:H100:2,gpu:A100:2','NodeName=H100']:
            with self.subTest(info=info):self.assertIsNone(c.recipe_hardware(recipe,info))
    def test_invalid_qualification_rejected(self):
        with self.assertRaises(ValueError):c.recipe_hardware(dict(qualified_gpu_types=['H200']),'Gres=gpu:H200:4')
    def test_h100_reconcile_starts_using_qualified_fallback(self):
        book=Booking('h','2',time.time()+3600,'gres/gpu:4')
        recipe=dict(id='small',fallback=True,qualified_gpu_types=['A100','H100'])
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(c,'discover_bookings',return_value={'h':book}), \
             patch.object(c,'registry_blocks',return_value={}), \
             patch.object(c,'read_json',side_effect=[dict(leases=[]),dict(entries=[])]), \
             patch.object(c.subprocess,'check_output',return_value='Gres=gpu:H100:4'), \
             patch.object(c.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='')), \
             patch.object(c,'make_profile',return_value={'python':'/python'}) as profile, \
             patch.object(c.subprocess,'Popen',return_value=SimpleNamespace(pid=123)):
            state=c.reconcile(dict(gpu_registry_dir='/tmp/reg',manager_script='/manager'),Path(tmp),{},[recipe])
            self.assertEqual(state['nodes']['h']['state'],'starting')
            self.assertEqual(profile.call_args.args[1]['hardware_qualification']['gpu_type'],'H100')

class CachedHardwareSelectionTests(unittest.TestCase):
    def setUp(self):
        self.book=Booking('h','2',time.time()+3600,'gres/gpu:4')
        self.bank=dict(state='active',expected_job_id='2',cache_scheme='canonical_full_weight',
                       source_commit='good',status=dict(entries=[dict(key='fp8',state='ready'),dict(key='small',state='ready')]))
        self.large=dict(id='large',engine_commit='good',node_cache_keys={'h':['fp8']},discovered_model={})
        self.small=dict(id='small',engine_commit='good',node_cache_keys={'h':['small']},qualified_gpu_types=['A100','H100'],discovered_model={})
    def test_larger_a100_only_cache_does_not_block_qualified_h100_cache(self):
        chosen=c.select_hardware_recipe(self.book,self.bank,[self.large,self.small],self.large,'Gres=gpu:H100:4')
        self.assertEqual(chosen['id'],'small')
        self.assertEqual(chosen['hardware_qualification']['gpu_type'],'H100')
        self.assertNotIn('hardware_qualification',self.small)
    def test_explicit_preferred_model_never_substitutes_older_bank(self):
        self.assertIsNone(c.select_hardware_recipe(self.book,self.bank,[self.large],self.large,'Gres=gpu:H100:4'))
    def test_a100_keeps_original_largest_model_choice(self):
        chosen=c.select_hardware_recipe(self.book,self.bank,[self.large,self.small],self.large,'Gres=gpu:A100:4')
        self.assertEqual(chosen['id'],'large')
    def test_unsupported_bootstrap_never_rebuilds_another_model(self):
        self.assertIsNone(c.select_hardware_recipe(self.book,None,[self.large,self.small],self.large,'Gres=gpu:H100:4'))
    def test_compatible_bootstrap_keeps_exact_recovery_snapshot(self):
        recovery=dict(self.small,dead_daemon_recovery=dict(daemon_id='old'))
        chosen=c.select_hardware_recipe(self.book,None,[self.small],recovery,'Gres=gpu:H100:4')
        self.assertEqual(chosen['dead_daemon_recovery'],recovery['dead_daemon_recovery'])
    def test_reconciliation_does_not_stop_manager_for_rejected_first_candidate(self):
        large=dict(self.large,discovered_model=dict(model_root='/large'))
        small=dict(self.small,discovered_model=dict(model_root='/small'))
        bank=dict(self.bank,node='h')
        services={'h':dict(recipe_id='small',root='/existing-small')}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(c,'discover_bookings',return_value={'h':self.book}), \
             patch.object(c,'registry_blocks',return_value={}), \
             patch.object(c,'read_json',side_effect=[dict(leases=[]),dict(entries=[bank])]), \
             patch.object(c,'read',return_value={}), \
             patch.object(c,'discover_recipe_keys'), \
             patch.object(c,'live_manager',return_value=True), \
             patch.object(c,'stop_manager') as stop, \
             patch.object(c.subprocess,'check_output',return_value='Gres=gpu:H100:4'), \
             patch.object(c.subprocess,'Popen') as spawn:
            state=c.reconcile(dict(gpu_registry_dir='/tmp/reg'),Path(tmp),services,[large,small])
            self.assertEqual(state['nodes']['h']['state'],'recovering')
            stop.assert_not_called();spawn.assert_not_called()

class GuardOverrideTests(unittest.TestCase):
    def test_guard_dependencies_are_pinned_before_profile_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);guard=root/'dashboard_guard.py';dep=root/'gpu_registry.py'
            guard.write_text('# guard');dep.write_text('# POSIX locks')
            bench=root/'bench.json';bench.write_text('{}')
            template=root/'template.json';template.write_text(json.dumps(dict(benchmark=dict(config_path=str(bench)))))
            recipe=dict(profile=str(template),profile_sha256=c.digest(template),node_template={})
            override=dict(script=str(guard),sha256=c.digest(guard),dependency_sha256={'gpu_registry.py':c.digest(dep)})
            cfg=dict(node_runner='/runner',guard_override=override)
            book=Booking('h','2',time.time()+3600,'gres/gpu:4')
            profile=c.make_profile(cfg,recipe,book,None,root)
            self.assertEqual(profile['guard_script'],str(guard))
            dep.write_text('# changed')
            with self.assertRaisesRegex(ValueError,'dependency changed'):
                c.make_profile(cfg,recipe,book,None,root)
    def test_changed_guard_script_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);guard=root/'guard.py';guard.write_text('# changed')
            bench=root/'bench.json';bench.write_text('{}');template=root/'template.json'
            template.write_text(json.dumps(dict(benchmark=dict(config_path=str(bench)))))
            recipe=dict(profile=str(template),profile_sha256=c.digest(template),node_template={})
            cfg=dict(node_runner='/runner',guard_override=dict(script=str(guard),sha256='wrong',dependency_sha256={}))
            with self.assertRaisesRegex(ValueError,'override changed'):
                c.make_profile(cfg,recipe,Booking('h','2',time.time()+3600,'gres/gpu:4'),None,root)
