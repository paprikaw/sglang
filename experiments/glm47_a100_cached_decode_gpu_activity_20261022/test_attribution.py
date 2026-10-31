"""CPU contract tests; no Torch import, model, GPU allocation, or serving."""
import ast
import inspect
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
import attribution_probe as probe
from analyze_gpu import partition,align_markers,correlate_markers

def attribution_plan():
    return dict(event_pool_size=24000,detailed_ordinals=list(range(9,13)),
                by_arm=dict(ep8=list(range(9,13)),pp2ep4=list(range(17,25))))


class AttributionTests(unittest.TestCase):
    def test_runtime_correlation_requires_real_launch_inside_same_forward(self):
        events=[];markers=[];calls=[]
        sample=dict(host_origin_ns=1000000,wall_origin_ns=2000000)
        for i in range(3):
            start=i*10000000
            events.append(dict(forward_index=i,start_ms=i*10,duration_ms=8,
                cpu_start_ns=1000000+start,cpu_end_ns=1000000+start+1000000))
            for j in range(2):
                correlation=i*2+j
                markers.append(dict(ts=50000+i*10000+j*7999,dur=1,args=dict(correlation=correlation)))
                calls.append(dict(ts=1000+i*10000+j*500,dur=1,args=dict(correlation=correlation)))
        result=correlate_markers(markers[1:],events,calls,sample,1000000)
        self.assertEqual(result['matched'],2)
        self.assertEqual(result['omitted_edge_forward_ids'],[0])
        with self.assertRaises(ValueError):correlate_markers(markers,events,calls[1:],sample,1000000)

    def test_gpu_markers_match_event_clock_with_only_edge_omission(self):
        events=[];markers=[];start=0
        for i,duration in enumerate([700,110,106,102]):
            events.append(dict(forward_index=i,start_ms=start,duration_ms=duration))
            markers += [dict(ts=1e6+start*1000,dur=1),dict(ts=1e6+(start+duration)*1000-1,dur=1)]
            start+=duration+5
        self.assertEqual(align_markers(markers,events)['matched'],4)
        edge=align_markers(markers[1:],events)
        self.assertEqual(edge['matched'],3)
        self.assertEqual(edge['omitted_model_edges'],[1,0])
        with self.assertRaises(ValueError):align_markers(markers[:3]+markers[4:],events)

    def test_gpu_timeline_idle_overlap_and_communication_union(self):
        rows=[dict(interval=(0,3000),phase='attention'),
              dict(interval=(2000,4000),phase='deep_ep.dispatch'),
              dict(interval=(3500,5000),phase='tp_communication'),
              dict(interval=(6000,7000),phase='expert_ffn')]
        result=partition(rows,(0,8000))
        self.assertEqual(result['component_ms'],dict(attention=2,overlap=1,
            communication=2,no_traced_gpu_activity=2,expert_ffn=1))
        self.assertEqual(sum(result['component_ms'].values()),8)
        self.assertEqual(result['phase_union_ms']['deep_ep.dispatch'],2)

    def test_unknown_kernels_are_not_forced_into_compute(self):
        result=partition([dict(interval=(1000,2000),phase='unclassified_gpu')],(0,3000))
        self.assertEqual(result['component_ms'],dict(no_traced_gpu_activity=2,unclassified_gpu=1))

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


if __name__ == '__main__': unittest.main()
