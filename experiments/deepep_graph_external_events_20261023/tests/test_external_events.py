import ast
import copy
import math
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from common import EVENT_NAMES,MODES,validate_event_sample,validate_freshness,expert_counts,routes
from worker import cleanup_graph_and_buffer
from node_runner import command,clean_exit


def sample(index=0):
    base=100+index*20
    return dict(sample=index,payload_phase=index+1,payload_valid=True,
        timestamps_from_origin_ms={name:base+offset for name,offset in zip(EVENT_NAMES,[0,.001,.101,.102,.302,.303])},
        dispatch_ms=.1,combine_ms=.2,graph_ms=.303)


class ExternalEventTests(unittest.TestCase):
    def test_realistic_positive_nested_event_spans(self):
        self.assertTrue(validate_event_sample(sample()))
        proof=validate_freshness([sample(i) for i in range(3)])
        self.assertTrue(proof['passed']);self.assertTrue(proof['not_capture_constant'])
        self.assertEqual(proof['elapsed_between_replay_starts_ms'],[20,20])

    def test_reject_missing_nonpositive_nan_inf_events_and_spans(self):
        for value in (0,-1,math.nan,math.inf,None):
            row=sample();row['timestamps_from_origin_ms']['dispatch_start']=value
            with self.assertRaises(ValueError):validate_event_sample(row)
            row=sample();row['dispatch_ms']=value
            with self.assertRaises(ValueError):validate_event_sample(row)
        row=sample();del row['timestamps_from_origin_ms']['graph_end']
        with self.assertRaises(ValueError):validate_event_sample(row)

    def test_reject_wrong_event_order_or_unbalanced_span(self):
        row=sample();row['timestamps_from_origin_ms']['combine_start']=99
        with self.assertRaises(ValueError):validate_event_sample(row)
        row=sample();row['dispatch_ms']=.9
        with self.assertRaises(ValueError):validate_event_sample(row)
        row=sample();row['graph_ms']=.1
        with self.assertRaises(ValueError):validate_event_sample(row)

    def test_reject_capture_constant_stale_and_unvalidated_payload(self):
        rows=[sample(i) for i in range(3)];rows[1]['timestamps_from_origin_ms']=dict(rows[0]['timestamps_from_origin_ms'])
        with self.assertRaises(ValueError):validate_freshness(rows)
        rows=[sample(i) for i in range(3)];rows[2]['payload_valid']=False
        with self.assertRaises(ValueError):validate_freshness(rows)
        rows=[sample(i) for i in range(3)];rows[2]['payload_phase']=2
        with self.assertRaises(ValueError):validate_freshness(rows)
        with self.assertRaises(ValueError):validate_freshness([sample()])

    def test_gpu_modules_only_import_inside_worker_main(self):
        for path in (ROOT/'runtime').glob('*.py'):
            for node in ast.parse(path.read_text()).body:
                if isinstance(node,ast.Import):self.assertFalse(any(x.name.startswith(('torch','deep_ep')) for x in node.names))
                elif isinstance(node,ast.ImportFrom):self.assertFalse((node.module or '').startswith(('torch','deep_ep')))

    def test_no_profiler_or_cupti_api_is_used(self):
        source=(ROOT/'runtime/worker.py').read_text()
        self.assertNotIn('torch.profiler',source)
        self.assertNotIn('cuptiGetVersion',source)
        self.assertNotIn('collector.start()',source)
        self.assertIn('collector_lifecycles=0',source)
        self.assertIn('cupti_api_calls=0',source)

    def test_external_timing_events_are_captured_and_retained(self):
        source=(ROOT/'runtime/worker.py').read_text()
        self.assertIn('torch.cuda.Event(enable_timing=True,external=True)',source)
        begin=source.index('with torch.cuda.graph(')
        end=source.index('# Graph, origin, all timing events',begin)
        body=source[begin:end]
        for name in EVENT_NAMES:self.assertIn(f"events['{name}'].record()",body)
        self.assertLess(body.index("events['dispatch_start'].record()"),body.index('buffer.low_latency_dispatch('))
        self.assertLess(body.index('hook()'),body.index("events['dispatch_end'].record()"))
        self.assertLess(body.index("events['combine_start'].record()"),body.index('buffer.low_latency_combine('))
        self.assertLess(body.index('hook2()'),body.index("events['combine_end'].record()"))
        self.assertIn('captured = dict(recv=recv,counts=counts,handle=handle,out=out',body)

    def test_every_sample_synchronizes_before_event_read_and_payload_check(self):
        source=(ROOT/'runtime/worker.py').read_text()
        body=source[source.index('for index in range(3):'):]
        replay=body.index('graph.replay()')
        completion=body.index('torch.cuda.synchronize()',replay)
        self.assertLess(replay,completion)
        self.assertLess(completion,body.index('origin.elapsed_time(events[name])'))
        self.assertIn('x.copy_(data(phase))',body)
        self.assertIn('if index:time.sleep(.02)',body)
        self.assertIn('payload=validate(captured,phase,already_synchronized=True)',body)
        self.assertIn('validate_freshness(samples)',body)

    def test_unchanged_balanced_real_verify_geometry(self):
        self.assertEqual(sum(expert_counts()),768)
        self.assertEqual(set(expert_counts()),{4,5})
        self.assertEqual([sum(expert_counts()[i*20:(i+1)*20]) for i in range(8)],[96]*8)
        self.assertEqual([e//20 for e in routes(0)[0]],list(range(8)))

    def test_single_fresh_mode_two_node_supervision(self):
        self.assertEqual(len(MODES),1)
        cmd=command(dict(python='/python',nodes=['spartan-gpgpu065','spartan-gpgpu167'],master_port=1),'/run',MODES[0],'spartan-gpgpu167')
        self.assertIn('--nnodes=2',cmd);self.assertIn('--nproc-per-node=4',cmd);self.assertIn('--max-restarts=0',cmd)
        self.assertTrue(clean_exit(dict(owned_processes_exited=True,gpu_cleanup={'passed':True})))
        self.assertFalse(clean_exit(dict(owned_processes_exited=True,gpu_cleanup={'passed':False})))

    def test_graph_reset_before_native_buffer_release(self):
        calls=[]
        class Graph:
            def reset(self):calls.append('graph')
        class Buffer:
            def destroy(self):calls.append('buffer')
        class Cuda:
            def synchronize(self):calls.append('sync')
        class Dist:
            def barrier(self):calls.append('barrier')
            def destroy_process_group(self):calls.append('group')
        cleanup_graph_and_buffer(Graph(),Buffer(),Cuda(),Dist())
        self.assertEqual(calls,['sync','graph','barrier','sync','buffer','group'])

    def test_terminal_cleanup_after_progress(self):
        source=(ROOT/'runtime/controller.py').read_text()
        self.assertLess(source.rindex("save(task/'progress.json'"),source.rindex("save(task/'cleanup.json'"))
        self.assertIn('event_valid=False,result_missing=True',source)
        self.assertIn('valid_event_modes=',source)


if __name__=='__main__':unittest.main()
