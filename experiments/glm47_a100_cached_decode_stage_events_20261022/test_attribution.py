"""CPU contract tests; no Torch import, model, GPU allocation, or serving."""
import ast
import inspect
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
import attribution_probe as probe

def attribution_plan():
    return dict(event_pool_size=24000,detailed_ordinals=list(range(9,13)),
                by_arm=dict(ep8=list(range(9,13)),pp2ep4=list(range(17,25))))


class AttributionTests(unittest.TestCase):
    def test_marker_matches_config_plan(self):
        plan = dict(attribution_plan(), run_token='test-owner-token')
        value = dict(schema_version=1, run_id='test_run', arm='ep8',
                     enabled=True, token=plan['run_token'],
                     detailed_forward_ordinals=plan['detailed_ordinals'])
        self.assertIs(probe.validate_marker(value, '/tmp/test_run', 'ep8', plan), value)
        self.assertEqual(plan['event_pool_size'], 24000)
        probe.validate_marker(dict(value, finished=True), '/tmp/test_run', 'ep8', plan, finish=True)
        for modified in (dict(value, run_id='old_run'),dict(value, token='wrong'),
                         dict(value, arm='pp2ep4'),dict(value, detailed_forward_ordinals=[1])):
            with self.assertRaises(ValueError):
                probe.validate_marker(modified, '/tmp/test_run', 'ep8', plan)

    def test_patch_preserves_descriptors_and_results(self):
        class Example:
            def ordinary(self,x): return x+1
            @staticmethod
            def static(x): return x+2
            @classmethod
            def classed(cls,x): return cls.__name__,x+3
        with tempfile.TemporaryDirectory() as directory:
            p = probe.Probe(directory,'ep8',dict(attribution=attribution_plan()))
            for name in ('ordinary','static','classed'):
                p.patch(Example,name,'test')
            self.assertEqual(Example().ordinary(2),3)
            self.assertEqual(Example.static(2),4)
            self.assertEqual(Example.classed(2),('Example',5))
            self.assertIsInstance(inspect.getattr_static(Example,'static'),staticmethod)
            self.assertIsInstance(inspect.getattr_static(Example,'classed'),classmethod)

    def test_exclusive_child_union_not_double_counted(self):
        rows=[dict(index=0,parent=None,category='A',start_stream=1,end_stream=1,
                   start_ms=0,end_ms=10,duration_ms=10),
              dict(index=1,parent=0,category='QKV',start_stream=1,end_stream=1,
                   start_ms=1,end_ms=6,duration_ms=5),
              dict(index=2,parent=0,category='core',start_stream=1,end_stream=1,
                   start_ms=4,end_ms=9,duration_ms=5)]
        result=probe.summarize(rows)
        self.assertEqual(result['stream1:A']['exclusive_ms'],2)
        self.assertEqual(probe.union([(0,2),(1,4),(5,6)]),5)
        with self.assertRaises(ValueError): probe.union([(2,1)])

    def test_transport_owner_counts_and_payload_semantics(self):
        routes=[rank*20 for rank in range(8)]
        ep8=probe.route_counts(routes+[ -1 ],0,8)
        ep4=probe.route_counts(routes,0,4)
        self.assertEqual(ep8['assignment_counts'],dict(self_gpu=1,nvlink_pair=1,same_node_rdma=2,cross_node_rdma=4))
        self.assertEqual(ep4['assignment_counts'],dict(self_gpu=2,nvlink_pair=2,same_node_rdma=4,cross_node_rdma=0))
        self.assertEqual(ep8['padding_assignments'],1)
        self.assertEqual(ep8['dispatch_payload_plus_token_record_bytes']['cross_node_rdma'],4*10256)
        self.assertEqual(ep8['combine_payload_bytes']['cross_node_rdma'],4*10240)
        with self.assertRaises(ValueError): probe.route_counts([160],0,8)

    def test_no_sync_inside_measured_wrapper(self):
        source=inspect.getsource(probe.Probe.call)
        self.assertNotIn('.synchronize(',source)
        self.assertNotIn('.cpu(',source)
        self.assertNotIn('torch.cuda.Event(',source)
        self.assertNotIn('.wait_stream(',source)
        install_source=inspect.getsource(probe.install)
        self.assertIn('SchedulerRequestReceiver.recv_requests = receive',install_source)
        self.assertIn('probe.finish()',install_source)
        self.assertIn('forward_batch.forward_mode.is_decode() and forward_batch.batch_size == 4',install_source)
        module=ast.parse(Path(probe.__file__).read_text())
        top_imports=[node for node in module.body if isinstance(node,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('torch' in ast.unparse(node) for node in top_imports))

    def test_light_mode_has_no_operator_trace_or_shape_walk(self):
        self.assertNotIn('torch.profiler.profile(', inspect.getsource(probe.Probe.start))
        source=inspect.getsource(probe.Probe.call)
        self.assertNotIn('record_function(',source)
        self.assertNotIn('shapes(args, kwargs)',source)

    def test_forward_api_partition_removes_nested_tp_from_attention(self):
        import summarize_events as analysis
        def span(index,parent,category,start,end):
            return dict(index=index,parent=parent,category=category,label='3',
                        forward_index=1,start_stream=7,end_stream=7,
                        start_ms=start,end_ms=end,duration_ms=end-start)
        rows=[span(0,None,'model_forward',0,10),span(1,0,'layer',0,10),
              span(2,1,'attention_full',1,5),span(3,2,'tp_collective',4,5),
              span(4,1,'routed_expert_compute',6,8),
              span(5,1,'deep_ep.dispatch_recv',8,9)]
        result=analysis.partition(dict(events=rows),rows[0])
        self.assertEqual(result['component_ms'],dict(attention=3,tp_communication=1,
                         expert_ffn=2,moe_communication=1,other=3))
        self.assertEqual(sum(result['component_ms'].values()),10)


if __name__ == '__main__': unittest.main()
