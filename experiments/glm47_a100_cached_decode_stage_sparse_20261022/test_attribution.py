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
        source=inspect.getsource(probe.Probe.call)+inspect.getsource(probe.Probe._call_impl)
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
        source=inspect.getsource(probe.Probe.call)+inspect.getsource(probe.Probe._call_impl)
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

    def test_sparse_layers_cover_full_model_and_controls_without_overlap(self):
        from collections import Counter
        cases=[('ep8',0,range(5,28),range(92),1),
               ('pp2ep4',0,range(9,55),range(46),2),
               ('pp2ep4',4,range(9,55),range(46,92),2)]
        for arm,rank,ordinals,layers,repeats in cases:
            detailed,control=Counter(),Counter()
            for ordinal in ordinals:
                d=probe.sampled_layers(arm,rank,ordinal)
                c=probe.envelope_control_layers(arm,rank,ordinal)
                self.assertFalse(set(d)&set(c))
                detailed.update(d);control.update(c)
            self.assertEqual(detailed,Counter({i:repeats for i in layers}))
            self.assertEqual(control,detailed)

    def test_sparse_estimator_sums_unique_layer_means_not_all_step_envelopes(self):
        # Fabricated CPU-only timing fixture; never experimental evidence.
        import json
        import summarize_sparse as analysis
        events=[];forwards=[]
        def event(parent,category,label,forward,start,end):
            row=dict(index=len(events),parent=parent,category=category,label=str(label),
                     forward_index=forward,extend_index=forward,start_stream=7,end_stream=7,
                     start_ms=start,end_ms=end,duration_ms=end-start)
            events.append(row);return row['index']
        for f in range(1,33):
            start=f*100
            parent=event(None,'model_forward','runner',f,start,start+46)
            forwards.append(dict(forward_index=f,extend_index=f,detailed=5<=f<=27,
                                 metadata=dict(forward_mode='2',batch_size=4)))
            for layer in probe.sampled_layers('ep8',0,f):
                low=start+layer*.5
                body=event(parent,'layer',layer,f,low,low+.5)
                event(body,'attention_full',layer,f,low,low+.3)
                event(body,'routed_expert_compute',layer,f,low+.3,low+.4)
                event(body,'deep_ep.dispatch_recv',layer,f,low+.4,low+.45)
            for layer in probe.envelope_control_layers('ep8',0,f):
                low=start+layer*.5
                event(parent,'layer_envelope_control',layer,f,low,low+.5)
        data=dict(passed=True,rank=0,arm='ep8',hostname='synthetic-fixture',events=events,
                  forwards=forwards,profiling_method='sparse_deferred_cuda_event_api_intervals')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.json';path.write_text(json.dumps(data))
            result=analysis.rank_summary(path)
        self.assertAlmostEqual(result['estimated_layer_body_ms'],46)
        self.assertAlmostEqual(result['estimated_component_ms']['attention'],27.6)
        self.assertAlmostEqual(result['estimated_component_ms']['expert_ffn'],9.2)
        self.assertAlmostEqual(result['local_component_scope_to_envelope_ratio'],1)


if __name__ == '__main__': unittest.main()
