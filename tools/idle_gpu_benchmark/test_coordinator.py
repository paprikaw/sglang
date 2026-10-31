import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import coordinator as c
import worker as w

class ScheduleTests(unittest.TestCase):
    def test_burst_keeps_ties_and_rates(self):
        rows=[dict(sample_id=i,prompt_tokens=3,output_tokens=2,offset_s=x) for i,x in enumerate([5,5,9,15])]
        planned,window=c.schedule(rows,2,'burstgpt')
        for actual,expected in zip([r['offset_s'] for r in planned],[0,0,.6,1.5]):self.assertAlmostEqual(actual,expected)
        self.assertEqual(window,2)
    def test_smooth_explicitly_removes_original_bursts(self):
        rows=[dict(offset_s=x) for x in [0,0,4,9]]
        scheduled,window=c.schedule(rows,2,'smooth')
        self.assertEqual([r['offset_s'] for r in scheduled],[0,.5,1,1.5])
    def test_zero_trace_span_cannot_silently_smooth(self):
        with self.assertRaises(ValueError):c.schedule([dict(offset_s=0)]*4,1,'burstgpt')
    def test_partition_preserves_global_offsets_and_exact_lengths(self):
        plan=dict(id='phase-a',members={'node-b':{},'node-a':{}},requests=[
            dict(sample_id=i,offset_s=i*.2,prompt_tokens=6+i,output_tokens=2+i) for i in range(5)])
        a,b=w.ephemeral_requests(plan,'node-a'),w.ephemeral_requests(plan,'node-b')
        self.assertEqual([r['sample_id'] for r in a],[0,2,4]);self.assertEqual([r['sample_id'] for r in b],[1,3])
        self.assertEqual(a[1]['offset_s'],.4);self.assertEqual(len(a[1]['input_ids']),8)
        self.assertEqual(a,w.ephemeral_requests(plan,'node-a'))
        plan['id']='phase-b';self.assertNotEqual(a[0]['input_ids'],w.ephemeral_requests(plan,'node-a')[0]['input_ids'])
    def test_missing_queue_evidence_never_promotes_point(self):
        cfg=dict(runtime_root='/tmp/none',search_requests=2,max_search_stages=1)
        obj=c.Coordinator(cfg)
        with patch.object(obj,'stage',return_value=dict(valid=False,reasons=['node:telemetry_missing'])),patch.object(obj,'publish'):
            with self.assertRaisesRegex(RuntimeError,'not a server-capacity'):obj.tune({},dict(requests=[{},{}],path='a'),'x',1)
    def test_search_brackets_and_confirms(self):
        cfg=dict(runtime_root='/tmp/none',search_requests=2,max_search_stages=3)
        obj=c.Coordinator(cfg)
        def stage(_cohort,_rows,rate,label,_path):
            return dict(valid=rate<=1.5,reasons=[] if rate<=1.5 else ['node:sustained_backlog'])
        with patch.object(obj,'stage',side_effect=stage),patch.object(obj,'publish'),patch.object(obj,'wait'):
            result=obj.tune({},dict(requests=[{},{}],path='a'),'x',1)
        self.assertAlmostEqual(result,1.4)

if __name__=='__main__':unittest.main()
