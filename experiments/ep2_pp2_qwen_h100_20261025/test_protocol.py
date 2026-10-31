import unittest
import ast
import statistics
import sys
import tempfile
import json
import gzip
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from protocol import cases, logical_capacity, microbatch_size


class ProtocolTest(unittest.TestCase):
    def test_pipeline_capacity_is_not_summed(self):
        self.assertEqual(logical_capacity('ep2', 100000), 200000)
        self.assertEqual(logical_capacity('pp2', 200000), 200000)

    def test_matrix_common_capacity_and_shapes(self):
        matrix = cases(dict(ep2=200000, pp2=240000))
        self.assertEqual([r['requests'] for r in matrix], [8, 8, 16])
        self.assertEqual(matrix[0]['prefill_cached_context_tokens'], 0)
        self.assertEqual(matrix[0]['new_prefill_tokens'], 4096)
        for row in matrix[1:]:
            self.assertEqual(row['cached_context_tokens'] % 64, 0)
            self.assertEqual(row['new_prefill_tokens'], 512)
            self.assertLess(row['global_cached_tokens'] + row['requests'] * (1024 + 128), 200001)

    def test_balanced_pipeline_slots(self):
        self.assertEqual(microbatch_size(8), 4)
        self.assertEqual(microbatch_size(16), 8)
        with self.assertRaises(ValueError):
            microbatch_size(7)

    def test_small_common_capacity_can_have_shorter_per_request_context(self):
        matrix = cases(dict(ep2=67712, pp2=480320))
        self.assertGreaterEqual(matrix[-1]['cached_context_tokens'], 1024)
        self.assertLess(matrix[-1]['cached_context_tokens'], 4096)

    def test_parallel_servers_have_disjoint_port_ranges(self):
        source = ast.parse(Path(__file__).with_name('runner.py').read_text())
        functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('arm_cfg', 'engine_args')]
        namespace = {}
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<pure configuration>', 'exec'), namespace)
        cfg = dict(port=19631, dist_port=19641, model_path='/model', attention_backend='fa3',
                   layouts=[dict(id='ep2', tp=2, ep=2, dp=2, pp=1),
                            dict(id='pp2', tp=1, ep=1, dp=1, pp=2)])
        ports = []
        for arm in ('ep2', 'pp2'):
            args = namespace['engine_args'](namespace['arm_cfg'](cfg, arm), arm)
            ports.append((int(args[args.index('--port') + 1]),
                          int(args[args.index('--dist-init-addr') + 1].split(':')[-1])))
            self.assertNotIn('--disable-cuda-graph', args)
            self.assertEqual(args[args.index('--moe-runner-backend') + 1], 'deep_gemm')
            self.assertEqual(args[args.index('--mamba-max-states-per-path') + 1], '1')
        self.assertGreaterEqual(ports[1][0] - ports[0][0], 100)
        self.assertGreaterEqual(ports[1][1] - ports[0][1], 100)

    def test_cached_prefill_rebuilds_base_outside_each_timed_trial(self):
        source = ast.parse(Path(__file__).with_name('runner.py').read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'measure_case')
        order, saved = [], []
        def prime(*args):
            order.append('prime')
            return {'preparation_s': 1}
        def cohort(*args):
            order.append('timed')
            return {'makespan_s': 1}
        fake_b = SimpleNamespace(cohort=cohort, summarize=lambda *a: {}, flush_cache=lambda *a: None)
        ns = dict(b=fake_b, h=SimpleNamespace(save=lambda p, d:saved.append(d)),
                  statistics=statistics, prime=prime, make_tokens=lambda *a:[[1]]*16,
                  payloads=lambda *a:[], set_microbatch=lambda *a:None,
                  check_cached=lambda *a:None)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<measurement>', 'exec'), ns)
        row = dict(id='test', requests=16, cached_context_tokens=13376,
                   prefill_cached_context_tokens=13376, new_prefill_tokens=512, output_tokens=128)
        ready = SimpleNamespace(set_policy=lambda *a:None)
        with patch.dict(sys.modules, {'pp_ready_send':ready}):
            ns['measure_case'](Path('/unused'), dict(repeats=3, policy_path='/unused'),
                               'pp2', 'key', row, dict(prefill_ready_send_selected=False), lambda:None)
        self.assertEqual(order[:8], ['prime', 'timed']*4)
        self.assertEqual(len(saved[0]['preparation']['trials']), 4)
        self.assertFalse(saved[0]['timing_includes_context_preparation'])
        self.assertEqual(order[8:], ['prime', 'timed', 'timed', 'timed', 'timed'])

    def test_priming_is_bounded_but_not_timed(self):
        source = ast.parse(Path(__file__).with_name('runner.py').read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'prime')
        cohorts = []
        ns = dict(b=SimpleNamespace(flush_cache=lambda *a:None,
                    packet=lambda cfg, arm, t, o, i:t, cohort=lambda cfg, key, p:cohorts.append(p)),
                  time=SimpleNamespace(monotonic=lambda:1))
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<priming>', 'exec'), ns)
        result = ns['prime']({}, 'pp2', 'key', [[i] for i in range(16)], lambda:None)
        self.assertEqual([len(c) for c in cohorts], [4,4,4,4])
        self.assertEqual([t for c in cohorts for t in c], [[i] for i in range(16)])
        self.assertFalse(result['measured_performance_sample'])

    def test_prefill_followup_excludes_priming_and_stops_after_all_chunks(self):
        from prefill_followup import profile_cell
        for cold in (True, False):
            with self.subTest(cold=cold), tempfile.TemporaryDirectory() as folder:
                run = Path(folder)
                order = []
                active = [False]
                def prime(*args):
                    self.assertFalse(active[0])
                    order.append('prepare')
                    return {'measured_performance_sample':False}
                def flush(*args):
                    self.assertFalse(active[0]); order.append('prepare')
                def cohort(cfg, key, packets):
                    order.append('profiled' if active[0] else 'warmup')
                    self.assertTrue(all(p['outputs'] == 1 for p in packets))
                    return {'requests':[{'cached_tokens':0 if cold else 4096}
                                        for _ in packets]}
                def profile(cfg, key, path, payload=None):
                    order.append(path)
                    if path == '/start_profile':
                        self.assertNotIn('num_steps', payload)
                        active[0] = True
                    else:
                        active[0] = False
                        for rank in (0, 1):
                            (run/'pp2/traces/test'/f'{rank}.trace.json.gz').write_bytes(
                                b'' if rank==1 else gzip.compress(b'{}'))
                fake_b = SimpleNamespace(flush_cache=flush, cohort=cohort,
                    packet=lambda c,a,t,o,i:dict(outputs=o), profile_action=profile)
                q = SimpleNamespace(b=fake_b, prime=prime,
                    set_microbatch=lambda *a:None,
                    make_tokens=lambda *a:[[1]] * 8,
                    payloads=lambda *a:[dict(outputs=1)] * 8,
                    check_cached=lambda r,c:self.assertTrue(all(v['cached_tokens']==c for v in r['requests'])),
                    h=SimpleNamespace(save=lambda p,d:p.write_text(json.dumps(d))))
                row = dict(id='test', requests=8, prefill_cached_context_tokens=0 if cold else 4096,
                           cached_context_tokens=4096, new_prefill_tokens=4096 if cold else 512)
                export_waits = []
                def finish_export(seconds):
                    export_waits.append(seconds)
                    (run/'pp2/traces/test/1.trace.json.gz').write_bytes(gzip.compress(b'{}'))
                with patch.dict(sys.modules, {'pp_ready_send':SimpleNamespace(set_policy=lambda *a:None)}), \
                     patch('prefill_followup.time.sleep', finish_export):
                    profile_cell(q,run,dict(policy_path='/unused',prefill_ready_send=True),
                                 'pp2','key',row,lambda:None)
                self.assertEqual(order, ['prepare','warmup','prepare','/start_profile',
                                         'profiled','/stop_profile'])
                result = json.loads((run/'pp2/traces/test/manifest.json').read_text())
                self.assertFalse(result['performance_sample'])
                self.assertEqual(result['new_tokens_per_request'], row['new_prefill_tokens'])
                self.assertTrue(result['all_prefill_chunks'])
                self.assertEqual(export_waits,[1])
                self.assertTrue(all(item['sha256'] != 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
                                    for item in result['files']))

    def test_gpu_kernel_attribution_uses_launch_scope_not_cpu_wall_duration(self):
        from prefill_analysis import owners, bucket, union_ms
        def event(cat,name,ts,dur,**args):
            return dict(ph='X',cat=cat,name=name,ts=ts,dur=dur,pid=1,tid=1,args=args)
        events = [event('user_annotation','step[EXTEND bs=1 toks=512 c_sq=512]',0,200),
                  event('user_annotation','component.layer.3',10,180),
                  event('user_annotation','component.attention.full.3',20,50),
                  event('cpu_op','aten::matmul',25,30,**{'External id':7}),
                  event('cuda_runtime','cudaLaunchKernel',30,2,correlation=17)]
        corr, ext = owners(events)
        self.assertEqual(corr[17][0], 'component.attention.full.3')
        self.assertEqual(ext[7][0], 'component.attention.full.3')
        self.assertEqual(bucket('dense projection GEMM',corr[17][0]),'attention_full_ms')
        self.assertEqual(bucket('void deep_ep::intranode::combine<int>',corr[17][0]),'combine_ms')
        self.assertEqual(union_ms([(0,10),(5,20),(30,40)]),.03)


if __name__ == '__main__':
    unittest.main()
