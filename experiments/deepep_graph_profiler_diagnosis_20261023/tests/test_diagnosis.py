import ast
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from common import (MODES,TRANSPORT,activity_config,environment,routes,expert_counts,
                    trace_counts,profile_valid,NATIVE_SHA)
from node_runner import command,clean_exit
from worker import cleanup_graph_and_buffer,mapped_library_paths


class DiagnosisTests(unittest.TestCase):
    def test_geometry_and_balanced_routes(self):
        counts=expert_counts()
        self.assertEqual(sum(counts),8*12*8)
        self.assertEqual(set(counts),{4,5})
        self.assertEqual([sum(counts[r*20:(r+1)*20]) for r in range(8)],[96]*8)
        for rank in range(8):
            rows=routes(rank)
            self.assertEqual(len(rows),12)
            for row in rows:
                self.assertEqual(len(set(row)),8)
                self.assertEqual([e//20 for e in row],list(range(8)))
        for bad in (-1,8):
            with self.assertRaises(ValueError):routes(bad)

    def test_every_mode_uses_identical_v7_transport(self):
        with patch.dict(os.environ,{'TEARDOWN_CUPTI':'1','DEEPEP_LL_PCIE_P2P_MAX_TOKENS':'999'}):
            envs=[environment({'python':'/test/python'},mode,'/run') for mode in MODES]
        self.assertEqual(envs[0],envs[1]);self.assertEqual(envs[1],envs[2])
        for env in envs:
            self.assertEqual({k:env[k] for k in TRANSPORT},TRANSPORT)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0,1,2,3')
            self.assertEqual(env['TEARDOWN_CUPTI'],'0')
            self.assertEqual(env['DISABLE_CUPTI_LAZY_REINIT'],'1')
            self.assertEqual(env['CUDA_MODULE_LOADING'],'EAGER')
            self.assertTrue(env['LD_PRELOAD'].endswith('trace_verbs.so'))
        self.assertEqual(len(NATIVE_SHA),64)

    def test_activity_filters_exclude_callbacks_only_in_third_mode(self):
        class Activity:CPU='CPU';CUDA='CUDA'
        self.assertEqual(activity_config(MODES[0],Activity),['CPU','CUDA'])
        self.assertEqual(activity_config(MODES[1],Activity),['CUDA'])
        self.assertEqual(activity_config(MODES[2],Activity),[{'CUDA':['CONCURRENT_KERNEL']}])
        with self.assertRaises(ValueError):activity_config('unknown',Activity)

    def test_exact_two_node_torchrun_fresh_process_per_mode(self):
        cfg=dict(python='/test/python',nodes=['spartan-gpgpu065','spartan-gpgpu167'],master_port=31000)
        for index,mode in enumerate(MODES):
            for node_rank,node in enumerate(cfg['nodes']):
                cmd=command(cfg,'/run',mode,node)
                self.assertIn('--nnodes=2',cmd);self.assertIn('--nproc-per-node=4',cmd)
                self.assertIn(f'--node-rank={node_rank}',cmd)
                self.assertIn(f'--master-port={31000+index}',cmd)
                self.assertIn('--max-restarts=0',cmd)
        with self.assertRaises(ValueError):command(cfg,'/run',MODES[0],'other')

    def test_missing_gpu_never_becomes_valid_profile(self):
        raw=trace_counts({'traceEvents':[dict(cat='cuda_runtime',name='cudaGraphLaunch',ts=1)]})
        self.assertEqual(raw['gpu_events'],0)
        for mode in MODES:self.assertFalse(profile_valid(mode,raw))

    def test_actual_marker_correlation_and_kernel_counts(self):
        events=[]
        for i in range(3):
            events.extend([dict(cat='user_annotation',name=f'stage-marker-proof:{i}',pid=1,tid=2,ts=i*10,dur=3),
                dict(cat='cuda_runtime',name='cudaLaunchKernel',pid=1,tid=2,ts=i*10+1,args={'correlation':i}),
                dict(cat='kernel',name='actual_spin_kernel',ts=100+i,args={'correlation':i})])
        events += [dict(cat='cuda_runtime',name='cudaGraphLaunch',ts=200+i) for i in range(3)]
        events += [dict(cat='kernel',name='deep_ep::dispatch',ts=300+i) for i in range(4)]
        events += [dict(cat='kernel',name='deep_ep::combine',ts=400+i) for i in range(4)]
        raw=trace_counts({'traceEvents':events})
        self.assertTrue(profile_valid(MODES[0],raw));self.assertTrue(profile_valid(MODES[1],raw))
        self.assertFalse(profile_valid(MODES[2],raw))
        filtered=trace_counts({'traceEvents':[e for e in events if e['cat']=='kernel']})
        self.assertTrue(profile_valid(MODES[2],filtered))
        self.assertFalse(profile_valid(MODES[0],filtered))

    def test_clean_crash_can_continue_but_dirty_exit_cannot(self):
        row=dict(returncode=-11,owned_processes_exited=True,gpu_cleanup={'passed':True})
        self.assertTrue(clean_exit(row))
        self.assertFalse(clean_exit(dict(row,owned_processes_exited=False)))
        self.assertFalse(clean_exit(dict(row,gpu_cleanup={'passed':False})))

    def test_graph_reset_precedes_buffer_and_group_teardown(self):
        calls=[]
        class Graph:
            def reset(self):calls.append('reset_graph')
        class Buffer:
            def destroy(self):calls.append('destroy_buffer')
        class Cuda:
            def synchronize(self):calls.append('sync')
        class Dist:
            def barrier(self):calls.append('barrier')
            def destroy_process_group(self):calls.append('destroy_group')
        cleanup_graph_and_buffer(Graph(),Buffer(),Cuda(),Dist())
        self.assertEqual(calls,['sync','reset_graph','barrier','sync','destroy_buffer','destroy_group'])

    def test_no_gpu_import_at_module_scope(self):
        for path in (ROOT/'runtime').glob('*.py'):
            tree=ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node,ast.Import):
                    self.assertFalse(any(x.name.startswith(('torch','deep_ep')) for x in node.names),str(path))
                elif isinstance(node,ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith(('torch','deep_ep')),str(path))

    def test_capture_and_warm_before_only_collector(self):
        source=(ROOT/'runtime/worker.py').read_text()
        self.assertEqual(source.count('collector.start()'),1)
        self.assertLess(source.index('with torch.cuda.graph('),source.index('collector.start()'))
        self.assertLess(source.index("checkpoint('unprofiled_graph_verified'"),source.index('collector.start()'))
        self.assertLess(source.index('collector.start()'),source.index("sample_libraries('first_profiler_started'"))
        self.assertLess(source.index('collector.start()'),source.index("with record_function('diagnostic:eager_deepep_before_graph')"))
        self.assertIn('return_recv_hook=True',source)
        self.assertIn('num_qps_per_rank=20',source)
        self.assertIn('buffer.low_latency_dispatch(x,indices,256,160',source)
        self.assertIn("x.copy_(data(1))",source)
        self.assertIn('changed_payload_graph=changed',source)

    def test_maps_only_before_collector_and_no_cupti_early_query(self):
        source=(ROOT/'runtime/worker.py').read_text()
        for line in source[:source.index('collector.start()')].splitlines():
            self.assertNotIn('sample_libraries(',line) if 'query_versions=True' in line else None
        self.assertIn("if query_versions and 'libcupti'",source)
        self.assertIn('os.RTLD_NOLOAD',source)
        text='\n'.join(['0000-1000 r-xp 0 00:00 1 /x/libcupti.so.12',
            '1000-2000 r-xp 0 00:00 1 /x/libcupti.so.12',
            '2000-3000 r-xp 0 00:00 1 /x/deep_ep_cpp.so'])
        self.assertEqual(mapped_library_paths(text),['/x/deep_ep_cpp.so','/x/libcupti.so.12'])

    def test_terminal_cleanup_after_progress_and_no_fake_profile(self):
        source=(ROOT/'runtime/controller.py').read_text()
        self.assertLess(source.rindex("save(task/'progress.json'"),source.rindex("save(task/'cleanup.json'"))
        self.assertIn('profile_valid=False,result_missing=True',source)
        self.assertIn('valid_profile_modes=',source)
        self.assertIn("sum(x['profile_valid']",source)


if __name__=='__main__':unittest.main()
