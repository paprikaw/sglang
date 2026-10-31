import ast
from pathlib import Path
import unittest
from unittest.mock import MagicMock
from types import SimpleNamespace
import urllib.request

from protocol import cases, logical_capacity, microbatch_size


class SmallModelProtocolTest(unittest.TestCase):
    def test_capacity_is_not_double_counted(self):
        self.assertEqual(logical_capacity('ep2', 100000), 200000)
        self.assertEqual(logical_capacity('pp2', 100000), 100000)

    def test_capacity_and_context_reserves(self):
        rows = cases(dict(ep2=2400000, pp2=2600000))
        self.assertEqual([r['requests'] for r in rows], [8, 16, 32])
        for r in rows[1:]:
            self.assertEqual(r['cached_context_tokens'] % 64, 0)
            self.assertLess(r['global_cached_tokens'] + r['requests'] * 1152, 2400000)
            self.assertLess(r['cached_context_tokens'] + 512, 262144)
            self.assertEqual(microbatch_size(r['requests']), r['requests'] // 2)

    def test_native_bf16_no_fp8_override(self):
        source = ast.parse(Path(__file__).with_name('runner.py').read_text())
        funcs = [n for n in source.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('engine_args', 'arm_cfg')]
        ns = {}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), '<configuration>', 'exec'), ns)
        cfg = dict(port=19931, dist_port=19941, model_path='/model', attention_backend='fa3',
                   layouts=[dict(id='ep2', tp=2, ep=2, dp=2, pp=1),
                            dict(id='pp2', tp=1, ep=1, dp=1, pp=2)])
        ports = []
        for arm in ('ep2', 'pp2'):
            args = ns['engine_args'](ns['arm_cfg'](cfg, arm), arm)
            self.assertNotIn('--quantization', args)
            self.assertNotIn('--disable-cuda-graph', args)
            self.assertEqual(args[args.index('--max-running-requests') + 1], '32')
            ports.append(int(args[args.index('--port') + 1]))
        self.assertEqual(ports[1] - ports[0], 100)

    def test_gc_freeze_happens_after_http_readiness(self):
        source = Path(__file__).with_name('runner.py').read_text()
        function = next(n for n in ast.parse(source).body
                        if isinstance(n,ast.FunctionDef) and n.name=='wait_ready')
        calls = [n for n in ast.walk(function) if isinstance(n,ast.Call)
                 and isinstance(n.func,ast.Attribute) and n.func.attr=='http_json']
        endpoints = {n.args[1].value:n.lineno for n in calls}
        freeze = next(n for n in ast.walk(function) if isinstance(n,ast.Call)
                      and isinstance(n.func,ast.Name) and n.func.id=='freeze_gc')
        self.assertLess(endpoints['/get_server_info'], freeze.lineno)

    def test_gc_endpoint_acknowledges_text_not_json(self):
        source = ast.parse(Path(__file__).with_name('runner.py').read_text())
        function = next(n for n in source.body if isinstance(n,ast.FunctionDef)
                        and n.name=='freeze_gc')
        response=MagicMock();response.__enter__.return_value=response
        response.status=200;response.read.return_value=b'Garbage collection frozen.\n'
        opener=MagicMock(return_value=response)
        ns=dict(urllib=SimpleNamespace(request=SimpleNamespace(
            Request=urllib.request.Request,urlopen=opener)))
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<native action>','exec'),ns)
        self.assertEqual(ns['freeze_gc'](dict(port=19931),'test-key'),'Garbage collection frozen.')
        response.read.return_value=b'Unexpected response'
        with self.assertRaises(RuntimeError):
            ns['freeze_gc'](dict(port=19931),'test-key')

    def test_single_gpu_qualification_has_small_runtime_budget(self):
        source = ast.parse(Path(__file__).with_name('runner.py').read_text())
        ns={}
        exec(compile(ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef)
            and n.name in ('arm_cfg','engine_args')],type_ignores=[]),'<configuration>','exec'),ns)
        cfg=dict(port=19931,dist_port=19941,model_path='/model',attention_backend='fa3',
            layouts=[dict(id='single',tp=1,ep=1,dp=1,pp=1)])
        args=ns['engine_args'](ns['arm_cfg'](cfg,'single'),'single')
        self.assertEqual(args[args.index('--max-mamba-cache-size')+1],'12')
        self.assertNotIn('--pp-max-micro-batch-size',args)
        self.assertEqual(args[args.index('--port')+1],'20131')

    def test_graph_attention_combine_is_not_expert_combine(self):
        from analyze import decode_bucket
        self.assertEqual(decode_bucket('flash::FlashAttnFwdCombine'), 'attention_ms')
        self.assertEqual(decode_bucket('deep_ep::internode_ll::combine'), 'combine_ms')
        self.assertEqual(decode_bucket('deep_ep::internode_ll::dispatch'), 'dispatch_ms')
        self.assertEqual(decode_bucket('deep_gemm::bf16_kernel<(deep_gemm::GemmType)2>'),
                         'routed_gemm_ms')

    def test_rankless_review_is_exact_and_rejects_fallback_or_duplicates(self):
        from cache_review import rankless_single_loader
        key='a'*64
        line=f'[2026-10-25 15:58:36] CPU weight cache load: source=canonical_cpu_daemon key={key} elapsed=11.813s'
        self.assertEqual(rankless_single_loader(line.encode(),key)['loader_records'],1)
        for invalid in (line+'\n'+line,line.replace('canonical_cpu_daemon','disk'),
                        line.replace(key,'b'*64),line.replace('] CPU','] TP1 CPU')):
            with self.assertRaises(AssertionError):
                rankless_single_loader(invalid.encode(),key)

    def test_prefill_excludes_incidental_graph_even_under_cpu_layer_wrapper(self):
        from prefill_attribution import is_decode_kernel
        self.assertTrue(is_decode_kernel({'graph id':7},'step[EXTEND bs=1]'))
        self.assertTrue(is_decode_kernel({'graph id':0},'step[DECODE bs=4]'))
        self.assertFalse(is_decode_kernel({'graph id':0},'step[EXTEND bs=4]'))


if __name__ == '__main__':
    unittest.main()
