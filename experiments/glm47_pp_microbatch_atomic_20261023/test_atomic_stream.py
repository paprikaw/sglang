import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
import atomic_stream
import measure
import spec_proof

class AtomicStreamTests(unittest.TestCase):
    def payload(self):
        return atomic_stream.make_payload([[i]*4096 for i in range(8)],lambda i,n,w:[100+i],'timed-decode-4096',3)

    def lines(self):
        payload=self.payload();lines=[]
        for count in (1,3):
            for index in reversed(range(8)):
                obj=dict(index=index,meta_info=dict(id=payload['rid'][index],completion_tokens=count),
                    output_ids=list(range(count)),text='answer')
                lines.append(b'data: '+json.dumps(obj).encode()+b'\n')
        return lines+[b'data: [DONE]\n']

    def test_exact_ordered_single_payload(self):
        payload=self.payload()
        self.assertEqual(len(payload['input_ids']),8)
        self.assertEqual(payload['rid'],[f'timed-decode-4096-{i}' for i in range(8)])
        self.assertTrue(payload['stream'])
        for i,ids in enumerate(payload['input_ids']):self.assertEqual(ids,([i]*4096+[100+i]))
        with self.assertRaises(ValueError):atomic_stream.make_payload([[1]],lambda *args:[2],'x',3)

    def test_out_of_order_sse_maps_by_index_and_shared_submit_clock(self):
        clock=iter(range(11,27))
        rows=atomic_stream.parse_sse(self.lines(),self.payload()['rid'],3,10,clock=lambda:next(clock))
        self.assertEqual([r['request_index'] for r in rows],list(range(8)))
        self.assertEqual(rows[7]['ttft_s'],1)
        self.assertEqual(rows[0]['ttft_s'],8)
        self.assertEqual(rows[7]['end_to_end_s'],9)
        self.assertEqual(rows[0]['tpot_s'],4)
        self.assertTrue(all(row['output_ids']==[0,1,2] for row in rows))

    def test_output_index_id_mismatch_and_missing_completion_rejected(self):
        for mutate in ('index','id','first_count','done','incomplete'):
            lines=self.lines()
            if mutate=='done':lines.pop()
            elif mutate=='incomplete':lines.pop(-2)
            else:
                obj=json.loads(lines[0][5:])
                if mutate=='index':obj['index']=True
                elif mutate=='id':obj['meta_info']['id']='wrong-id'
                else:obj['meta_info']['completion_tokens']=2;obj['output_ids']=[1,2]
                lines[0]=b'data: '+json.dumps(obj).encode()+b'\n'
            with self.subTest(mutate=mutate),self.assertRaises(RuntimeError):
                atomic_stream.parse_sse(lines,self.payload()['rid'],3,0,clock=lambda:1)

    def test_one_urlopen_for_all_eight(self):
        payload=self.payload();requests=[]
        def response(req,timeout):requests.append(json.loads(req.data));return io.BytesIO(b''.join(self.lines()))
        with patch.object(atomic_stream.urllib.request,'urlopen',side_effect=response):
            rows,wall=atomic_stream.generate_batch('http://unused',payload)
        self.assertEqual(requests,[payload]);self.assertEqual(len(rows),8)

    def test_strict_repeatability_and_index_matching(self):
        rows=[dict(request_index=i,output_ids=[i,2],output_text=str(i)) for i in range(8)]
        self.assertTrue(atomic_stream.output_parity(rows,list(reversed(rows)))['passed'])
        other=copy.deepcopy(rows);other[0]['output_ids'][0]=77
        self.assertFalse(atomic_stream.output_parity(rows,other)['passed'])
        with self.assertRaises(ValueError):atomic_stream.output_parity(rows,rows[:-1])

    def test_repeat_is_native_graph_proof_cohort(self):
        from types import SimpleNamespace
        batch=SimpleNamespace(reqs=[SimpleNamespace(rid=f'repeat-decode-4096-{i}') for i in range(4)])
        self.assertEqual(spec_proof.cohort(batch),'repeat-decode-4096')

    def test_profile_bootstrap_is_idle_command_not_inside_forward(self):
        import ast
        tree=ast.parse((Path(__file__).parent/'runtime/mtp_activity.py').read_text())
        arm=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='arm')
        self.assertTrue(any(isinstance(n,ast.Attribute) and n.attr=='synchronize' for n in ast.walk(arm)))
        forward=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='forward')
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr in ('synchronize','start') for n in ast.walk(forward)))

    def test_model_calls_use_atomic_batch_not_per_request_threads(self):
        import ast
        tree=ast.parse((Path(__file__).parent/'runtime/measure.py').read_text())
        for function in tree.body:
            if isinstance(function,ast.FunctionDef) and function.name in ('prepare_semantic_cohort','profile_once','measure_all'):
                self.assertFalse(any(isinstance(n,ast.Name) and n.id=='ThreadPoolExecutor' for n in ast.walk(function)))
                self.assertTrue(any(isinstance(n,ast.Name) and n.id=='batch_request' for n in ast.walk(function)))

if __name__=='__main__':unittest.main()
