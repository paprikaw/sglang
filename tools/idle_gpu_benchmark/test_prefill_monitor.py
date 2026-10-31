"""CPU-only parser and exact background authorization checks."""
import copy
import unittest
from unittest import mock
import prefill_monitor as monitor


class GpuParserTests(unittest.TestCase):
    def test_physical_gpu_readings_and_average_inputs_are_numeric(self):
        rows=monitor.parse_gpu_csv('0, NVIDIA A100 80GB PCIe, 97, 62000, 81920, 245.67\n1, NVIDIA A100 80GB PCIe, 100, 61000, 81920, 250\n')
        self.assertEqual([row['index'] for row in rows],[0,1])
        self.assertEqual(rows[0]['utilization_gpu_pct'],97)
        self.assertEqual(rows[0]['memory_used_mib'],62000)
        self.assertEqual(rows[0]['unavailable_fields'],[])
        self.assertEqual(rows[1]['power_draw_w'],250)

    def test_unavailable_nonfinite_negative_and_bad_utilization_remain_null(self):
        rows=monitor.parse_gpu_csv('0, A100, N/A, -1, 81920, [Not Supported]\n1, A100, 101, nan, inf, 200\n')
        self.assertIsNone(rows[0]['utilization_gpu_pct'])
        self.assertIsNone(rows[0]['memory_used_mib'])
        self.assertIsNone(rows[0]['power_draw_w'])
        self.assertIsNone(rows[1]['utilization_gpu_pct'])
        self.assertIsNone(rows[1]['memory_total_mib'])
        self.assertEqual(len(rows[1]['unavailable_fields']),3)

    def test_empty_malformed_or_duplicate_gpu_identity_rejected(self):
        for text in ('', 'NVIDIA-SMI failed', 'zero,A100,10,20,30,40',
                     '0,A100,10,20,30,40\n0,A100,20,20,30,40'):
            with self.subTest(text=text),self.assertRaises(ValueError):
                monitor.parse_gpu_csv(text)

    def test_remote_query_uses_fixed_readonly_command_and_safe_timeout(self):
        completed=mock.Mock(returncode=0,stdout='0,A100,99,100,81920,250\n')
        with mock.patch.object(monitor.subprocess,'run',return_value=completed) as run:
            result=monitor.query_gpu('spartan-gpgpu069')
        self.assertIsNone(result['error'])
        args,kwargs=run.call_args
        self.assertEqual(args[0][-1],monitor.GPU_COMMAND)
        self.assertEqual(args[0][-2],'spartan-gpgpu069')
        self.assertEqual(kwargs['timeout'],20)
        self.assertNotIn('shell',kwargs)
        self.assertNotIn('pid',result)

    def test_invalid_node_never_spawns_shell_or_ssh(self):
        with mock.patch.object(monitor.subprocess,'run') as run:
            result=monitor.query_gpu('node; secret-command')
        run.assert_not_called()
        self.assertEqual(result['error'],'invalid_node_name')
        self.assertIsNone(result['gpus'])


class IndependentTelemetryTests(unittest.TestCase):
    def test_worker_freshness_uses_time_after_slow_parallel_ssh_collection(self):
        config=dict(nodes=['a'],runtime_root='/runtime')
        decision=dict(authorized=True,reason=None,run='/runs/a',deployment_id='model-a')
        fleet=dict(state='running')
        future=mock.Mock()
        future.result.return_value=dict(sampled_at=1000,finished_at=1012,gpus=[dict(utilization_gpu_pct=100)],error=None)
        executor=mock.Mock()
        executor.submit.return_value=future
        # The worker continues updating while SSH spends 12 seconds collecting.
        # Comparing timestamp1012 with the original start1000 would be incorrect.
        current_worker=dict(run='/runs/a',timestamp=1012,input_tokens_per_s=500,completed_rps=2,
                            cumulative=dict(input_tokens=1000),status='running')
        with mock.patch.object(monitor,'current_decisions',return_value=(fleet,{'a':decision})), \
             mock.patch.object(monitor,'safe_read',return_value=current_worker), \
             mock.patch.object(monitor,'manager_is_live',return_value=True), \
             mock.patch.object(monitor.time,'time',side_effect=[1000,1012,1012,1012]):
            _,snapshot=monitor.snapshot(config,executor)
        self.assertFalse(snapshot['nodes']['a']['worker_stale'])
        self.assertEqual(snapshot['total_observed_input_tokens_per_s'],500)
        self.assertEqual(snapshot['gpu_utilization_avg_pct'],100)

    def test_gpu_read_failure_keeps_fresh_worker_throughput_and_hides_process_fields(self):
        config=dict(nodes=['a'],runtime_root='/runtime')
        decision=dict(authorized=True,reason=None,run='/runs/a',deployment_id='model-a')
        fleet=dict(state='running')
        result=mock.Mock()
        result.result.return_value=dict(sampled_at=monitor.time.time(),finished_at=monitor.time.time(),gpus=None,error='ssh_timeout')
        executor=mock.Mock()
        executor.submit.return_value=result
        worker=dict(run='/runs/a',timestamp=monitor.time.time(),pid=1234,input_tokens_per_s=500,
                    completed_rps=2,cumulative=dict(input_tokens=1000),status='running')
        with mock.patch.object(monitor,'current_decisions',return_value=(fleet,{'a':decision})), \
             mock.patch.object(monitor,'safe_read',return_value=worker), \
             mock.patch.object(monitor,'manager_is_live',return_value=True):
            _,snapshot=monitor.snapshot(config,executor)
        self.assertEqual(snapshot['total_observed_input_tokens_per_s'],500)
        self.assertIsNone(snapshot['gpu_utilization_avg_pct'])
        self.assertTrue(snapshot['nodes']['a']['gpu_stale'])
        self.assertEqual(snapshot['nodes']['a']['gpu_error'],'ssh_timeout')
        self.assertNotIn('pid',snapshot['nodes']['a'])


class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.now=1000
        self.config=dict(nodes=['a','b'])
        self.fleet=dict(state='running',updated_at=999,attempts={
            'a':dict(run='/runs/a',status=dict(ready=True,timestamp=999)),
            'b':dict(run='/runs/b',status=dict(ready=True,timestamp=999))})
        self.deployments={node:dict(node=node,status='healthy',deployment_id='model-'+node) for node in ('a','b')}
        self.registry=dict(leases=[dict(nodes=[node],state='active',experiment_id='model-'+node,
            resource_priority='background',preemptible=True,heartbeat_at=999,ttl_s=15) for node in ('a','b')])

    def check(self):
        return monitor.authorized_nodes(self.config,self.fleet,self.registry,self.deployments,self.now)

    def test_ready_exact_active_background_nodes_authorized_without_mutation(self):
        before=copy.deepcopy((self.fleet,self.registry,self.deployments))
        result=self.check()
        self.assertTrue(result['a']['authorized'])
        self.assertTrue(result['b']['authorized'])
        self.assertEqual(before,(self.fleet,self.registry,self.deployments))
        self.assertNotIn('leases',result['a'])

    def test_foreground_preemption_denies_only_affected_node(self):
        self.registry['leases'][1]['preemption_requested']=dict(request_id='foreground')
        result=self.check()
        self.assertTrue(result['a']['authorized'])
        self.assertFalse(result['b']['authorized'])
        self.assertEqual(result['b']['reason'],'foreground_preemption_requested')

    def test_normal_owner_replaced_deployment_and_multi_node_lease_denied(self):
        for change in (dict(resource_priority='normal'),dict(experiment_id='different-model'),
                       dict(nodes=['a','b']),dict(preemptible=False),dict(heartbeat_at=1),dict(state='released')):
            original=copy.deepcopy(self.registry)
            self.registry['leases'][0].update(change)
            self.assertFalse(self.check()['a']['authorized'])
            self.registry=original

    def test_duplicate_active_ownership_rejected_even_if_one_matches(self):
        self.registry['leases'].append(copy.deepcopy(self.registry['leases'][0]))
        result=self.check()
        self.assertFalse(result['a']['authorized'])
        self.assertEqual(result['a']['reason'],'exact_active_lease_missing_or_ambiguous')

    def test_missing_or_malformed_heartbeat_and_timestamp_fail_closed(self):
        for field,value in (('heartbeat_at',None),('heartbeat_at','bad'),('ttl_s',None),('ttl_s','bad')):
            original=copy.deepcopy(self.registry)
            self.registry['leases'][0][field]=value
            self.assertFalse(self.check()['a']['authorized'])
            self.registry=original
        self.fleet['updated_at']=None
        self.assertFalse(self.check()['a']['authorized'])

    def test_stale_manager_stale_instance_missing_and_unhealthy_deployment_denied(self):
        cases=[lambda:self.fleet.update(updated_at=0),
               lambda:self.fleet['attempts']['a']['status'].update(timestamp=0),
               lambda:self.fleet['attempts']['a']['status'].update(ready=False),
               lambda:self.deployments['a'].update(status='unhealthy')]
        for change in cases:
            fleet,deployments=copy.deepcopy((self.fleet,self.deployments))
            change()
            self.assertFalse(self.check()['a']['authorized'])
            self.fleet,self.deployments=fleet,deployments


if __name__=='__main__':
    unittest.main()
