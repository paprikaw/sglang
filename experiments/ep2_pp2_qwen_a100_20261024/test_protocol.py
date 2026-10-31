import importlib.util
import gzip
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=pathlib.Path(__file__).resolve().parent
OLD=ROOT.parent/'qwen_a100_capacity_20261019_r1'
sys.path.insert(0,str(OLD))
spec=importlib.util.spec_from_file_location('process_helpers',OLD/'qualification.py')
h=importlib.util.module_from_spec(spec);sys.modules['process_helpers']=h;spec.loader.exec_module(h)
spec=importlib.util.spec_from_file_location('paired_runner',ROOT/'runner.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
spec=importlib.util.spec_from_file_location('paired_analysis',ROOT/'analyze.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


class ProtocolTests(unittest.TestCase):
    def test_equal_devices_and_compute_backend(self):
        cfg=dict(model_path='/model',port=19421,layouts=[dict(id='ep2',tp=2,ep=2,dp=2,pp=1),
                                                      dict(id='pp2',tp=1,ep=1,dp=1,pp=2)])
        for arm in ['ep2','pp2']:
            args=r.engine_args(cfg,arm)
            self.assertIn('--disable-cuda-graph',args)
            self.assertIn('--disable-overlap-schedule',args)
            self.assertEqual(args[args.index('--moe-runner-backend')+1],'marlin')
            self.assertEqual(args[args.index('--dtype')+1],'bfloat16')
        self.assertIn('--enable-dp-attention',r.engine_args(cfg,'ep2'))
        self.assertIn('--pp-max-micro-batch-size',r.engine_args(cfg,'pp2'))
        pp = r.engine_args(cfg,'pp2')
        self.assertEqual(pp[pp.index('--pp-async-batch-depth')+1], '0')

    def test_fixed_request_affinity(self):
        for index in range(8):
            ep=r.packet({},'ep2',[1,2],128,index)
            pp=r.packet({},'pp2',[1,2],128,index)
            self.assertEqual(ep['data_parallel_rank'],index%2)
            self.assertNotIn('data_parallel_rank',pp)
            ep.pop('data_parallel_rank')
            self.assertEqual(ep,pp)

    def test_native_flush_plaintext_and_failure(self):
        class Response:
            status=200
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return b'Cache flushed.\nPlease check backend logs.'
        with patch.object(r.urllib.request,'urlopen',return_value=Response()):
            self.assertTrue(r.flush_cache(dict(port=1),'private').startswith('Cache flushed.'))
        class Failed(Response):
            def read(self):return b'Cannot flush while requests are running.'
        with patch.object(r.urllib.request,'urlopen',return_value=Failed()):
            with self.assertRaises(RuntimeError):r.flush_cache(dict(port=1),'private')

    def test_union_does_not_add_overlapped_kernels(self):
        self.assertEqual(a.union_us([(0,5),(3,9),(11,12)]),10)
        self.assertEqual(a.union_us([]),0)

    def test_full_attention_method_gets_its_own_scope(self):
        import contextlib
        from types import SimpleNamespace
        spec = importlib.util.spec_from_file_location('scope_adapter', ROOT/'runtime_adapter.py')
        adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
        observed = []
        @contextlib.contextmanager
        def record(name):
            observed.append(name)
            yield
        torch = SimpleNamespace(autograd=SimpleNamespace(_profiler_enabled=lambda: True),
                                profiler=SimpleNamespace(record_function=record))
        class Layer:
            def self_attention(self, value): return value + 1
            def forward(self, value): return self.self_attention(value)
        layer = Layer()
        model = SimpleNamespace(named_modules=lambda: [('model.layers.3', layer)])
        adapter.attach_scopes(model, torch)
        self.assertEqual(layer.forward(5), 6)
        self.assertEqual(observed, ['layer.3', 'component.attention.full.3'])

    def test_native_profiler_plaintext_actions_and_failure(self):
        class Response:
            status = 200
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return self.body
        for path, body in [('/start_profile', b'Start profiling.\n'),
                           ('/stop_profile', b'Stop profiling. This will take some time.\n')]:
            with patch.object(r.urllib.request, 'urlopen', return_value=Response(body)) as request:
                self.assertEqual(r.profile_action(dict(port=1), 'private', path), body.decode())
                self.assertEqual(request.call_args.args[0].method, 'POST')
        with patch.object(r.urllib.request, 'urlopen', return_value=Response(b'not started')):
            with self.assertRaises(RuntimeError):
                r.profile_action(dict(port=1), 'private', '/start_profile')

    def test_timing_summary_uses_request_latency_not_inverse_throughput(self):
        rounds=[dict(requests=[dict(ttft_s=1,e2e_s=3,tpot_ms=20),
                               dict(ttft_s=2,e2e_s=4,tpot_ms=30)],
                     input_tokens_per_s=20,output_tokens_per_s=10)]
        row=r.summarize(rounds,'test')
        self.assertEqual(row['ttft_s']['mean'],1.5)
        self.assertEqual(row['e2e_s']['median'],3.5)
        self.assertEqual(row['tpot_ms']['mean'],25)
        self.assertFalse(row['profiler_enabled'])

    def test_gpu_attribution_uses_cpu_launch_not_cpu_scope_duration(self):
        def event(name,cat,ts,dur,**args):
            return dict(name=name,cat=cat,ph='X',ts=ts,dur=dur,tid=1,args=args)
        events=[event('step[DECODE bs=4]','user_annotation',0,500),
                event('layer.0','user_annotation',10,400),
                event('component.attention.full.0','user_annotation',20,200),
                event('cudaLaunchKernel','cuda_runtime',30,2,correlation=11),
                event('attention_kernel','kernel',100,12,correlation=11),
                event('layer.0','gpu_user_annotation',100,12),
                event('step[DECODE bs=4]','gpu_user_annotation',90,30)]
        with tempfile.TemporaryDirectory() as temp:
            path=pathlib.Path(temp)/'trace.json.gz'
            with gzip.open(path,'wt') as f:json.dump(dict(traceEvents=events),f)
            summary=a.trace_summary(path)
        row=summary['layer_means'][0]
        self.assertEqual(len(summary['invocations']), 1)
        self.assertEqual(row['attention.full'],.012)
        self.assertEqual(row['cpu_scope_mean_ms'],.4)
        self.assertEqual(row['step'],'step[DECODE bs=4]')

    def test_bridge_fixture_respects_keyword_only_native_wrapper(self):
        import ast
        tree=ast.parse((ROOT/'runtime_adapter.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Attribute) and n.func.attr in ('dispatch','combine')]
        self.assertEqual(len(calls),0)
        self.assertTrue(all(not n.args and n.keywords for n in calls))
        qualifier=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='qualify_model')
        native=[n for n in ast.walk(qualifier) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Name) and n.func.id=='layer']
        self.assertEqual(len(native),2)
        self.assertTrue(all({x.arg for x in n.keywords} == {'hidden_states','topk_output'} for n in native))
        bridge = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'BridgeMethod')
        apply = next(n for n in bridge.body if isinstance(n, ast.FunctionDef) and n.name == 'apply')
        self.assertEqual([x.arg for x in apply.args.args], ['self', 'layer', 'dispatch_output'])

    def test_worker_entry_preserves_success_exit_status(self):
        import argparse
        import ast
        import contextlib
        import io
        import traceback
        tree = ast.parse((ROOT/'runner.py').read_text())
        entry = tree.body[-1]
        module = compile(ast.Module(body=[entry], type_ignores=[]), str(ROOT/'runner.py'), 'exec')
        with tempfile.TemporaryDirectory() as temp:
            (pathlib.Path(temp)/'config.json').write_text('{}')
            for failed in [False, True]:
                def worker(*args):
                    if failed: raise RuntimeError('test worker failure')
                    return 0
                env = dict(__name__='__main__', argparse=argparse, Path=pathlib.Path,
                           json=json, sys=sys, traceback=traceback, worker=worker)
                with patch.object(sys, 'argv', ['runner.py','--worker','--run',temp]), \
                     contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as result:
                        exec(module, env)
                self.assertEqual(result.exception.code, 1 if failed else 0)


if __name__=='__main__':unittest.main()
