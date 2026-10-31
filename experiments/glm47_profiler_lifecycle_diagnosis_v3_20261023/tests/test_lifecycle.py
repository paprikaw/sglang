import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
from common import MODES, environment, trace_counts
from node_runner import command, can_continue
from worker import cleanup_graphs_and_group, mapped_library_paths, profile_valid


class LifecycleTests(unittest.TestCase):
    def test_both_fresh_modes_replace_inherited_flags(self):
        with patch.dict(os.environ, {'TEARDOWN_CUPTI': '1', 'DISABLE_CUPTI_LAZY_REINIT': '0'}):
            modes = [environment({'python': '/test/python'}, mode, Path('/run')) for mode in MODES]
        for value in modes:
            self.assertEqual(value['TEARDOWN_CUPTI'], '0')
            self.assertEqual(value['DISABLE_CUPTI_LAZY_REINIT'], '1')
            self.assertEqual(value['CUDA_VISIBLE_DEVICES'], '0,1')
        self.assertEqual(modes[0], modes[1])

    def test_candidate_flags_precede_torch_import(self):
        candidate = environment({'python': '/test/python'}, MODES[0], Path('/run'))
        self.assertEqual(candidate['TEARDOWN_CUPTI'], '0')
        self.assertEqual(candidate['DISABLE_CUPTI_LAZY_REINIT'], '1')

    def test_command_is_bounded_two_gpu_fresh_process(self):
        result = command({'python': '/test/python', 'master_port': 12345}, '/run', MODES[0])
        self.assertIn('--nproc-per-node=2', result)
        self.assertIn('--max-restarts=0', result)
        self.assertIn('--nnodes=1', result)
        self.assertEqual(result[-1], MODES[0])

    def test_absent_device_events_are_explicit(self):
        events = [dict(cat='cuda_runtime', name='cudaLaunchKernel', ts=1, dur=1, args={'correlation': 7})]
        row = trace_counts({'traceEvents': events})
        self.assertEqual(row['gpu_events'], 0)
        self.assertEqual(row['cuda_launches'], 1)

    def test_correlation_evidence_uses_real_scoped_launch(self):
        events = [dict(cat='user_annotation', name='stage-marker-proof:0', pid=1, tid=2, ts=10, dur=5),
            dict(cat='cuda_runtime', name='cudaLaunchKernel', pid=1, tid=2, ts=11, dur=1, args={'correlation': 7}),
            dict(cat='kernel', name='actual_spin_kernel', ts=20, dur=1, args={'correlation': 7, 'device': 0}),
            dict(cat='kernel', name='ncclDevKernel_SendRecv', ts=22, dur=2, args={'correlation': 8, 'device': 0})]
        row = trace_counts({'traceEvents': events})
        self.assertEqual(row['gpu_kernels'], 2)
        self.assertEqual(row['observed_nccl_kernels'], 1)
        self.assertEqual(row['scoped_proofs'][0]['kernels'][0]['name'], 'actual_spin_kernel')

    def test_static_worker_has_real_send_recv_capture_and_two_profiler_modes(self):
        source = (ROOT / 'runtime/worker.py').read_text()
        self.assertIn('dist.batch_isend_irecv', source)
        self.assertIn('torch.cuda.graph(graph, stream=stream)', source)
        self.assertIn('activities = [ProfilerActivity.CUDA]', source)
        self.assertIn('activities.insert(0, ProfilerActivity.CPU)', source)
        self.assertIn('bidirectional_payload_verified=verified', source)

    def test_all_live_graphs_reset_before_nccl_group_shutdown(self):
        calls = []
        class Graph:
            def reset(self): calls.append('graph_reset')
        class Cuda:
            def synchronize(self): calls.append('cuda_sync')
        class Dist:
            def barrier(self): calls.append('barrier')
            def destroy_process_group(self): calls.append('destroy_pg')
        graphs = [Graph(), Graph()]
        cleanup_graphs_and_group(graphs, Cuda(), Dist())
        self.assertEqual(graphs, [])
        self.assertEqual(calls, ['cuda_sync', 'graph_reset', 'graph_reset', 'barrier', 'cuda_sync', 'destroy_pg'])

    def test_terminal_cleanup_is_written_after_terminal_progress(self):
        source = (ROOT / 'runtime/controller.py').read_text()
        terminal_progress = source.rindex("save(task / 'progress.json'")
        terminal_cleanup = source.rindex("save(task / 'cleanup.json'")
        self.assertLess(terminal_progress, terminal_cleanup)

    def test_first_collector_is_after_capture_and_unprofiled_replay(self):
        source = (ROOT / 'runtime/worker.py').read_text()
        self.assertLess(source.index('with torch.cuda.graph('), source.index('collector.start()'))
        self.assertLess(source.index("checkpoint('unprofiled_graph_verified'"), source.index('collector.start()'))
        self.assertLess(source.index('collector.start()'), source.index("with record_function('diagnostic:eager_p2p_before_graph')"))
        self.assertLess(source.index("checkpoint('before_profiled_graph_replay'"), source.index('collector.stop()'))
        self.assertEqual(source.count('collector.start()'), 1)
        self.assertIn('torch.cuda._sleep(100000000)', source)
        self.assertNotIn('capture_trace(', source)

    def test_library_maps_deduplicate_actual_mapped_paths(self):
        text = '\n'.join(['0000-1000 r-xp 0 00:00 1 /some/cu12/libcupti.so.12',
            '1000-2000 r--p 0 00:00 1 /some/cu12/libcupti.so.12',
            '2000-3000 r-xp 0 00:00 1 /other/cu13/libcupti.so.13',
            '3000-4000 r-xp 0 00:00 1 /runtime/libcudart.so.12',
            '4000-5000 r-xp 0 00:00 1 /unrelated/libc.so.6'])
        self.assertEqual(mapped_library_paths(text), ['/other/cu13/libcupti.so.13',
            '/runtime/libcudart.so.12', '/some/cu12/libcupti.so.12'])
        source = (ROOT / 'runtime/worker.py').read_text()
        self.assertIn('os.RTLD_NOLOAD', source)
        self.assertIn('library.cuptiGetVersion', source)
        self.assertIn("sample_libraries('first_profiler_started')", source)

    def test_crash_continues_only_after_owned_cleanup(self):
        row = dict(passed=False, returncode=-11, owned_processes_exited=True, gpu_cleanup={'passed': True})
        self.assertTrue(can_continue(row, 0, False))
        self.assertFalse(can_continue(dict(row, owned_processes_exited=False), 0, False))
        self.assertFalse(can_continue(dict(row, gpu_cleanup={'passed': False}), 0, False))
        self.assertFalse(can_continue(row, 15, False))
        self.assertFalse(can_continue(row, 0, True))

    def test_missing_gpu_or_graph_evidence_is_invalid(self):
        counts = dict(cuda_graph_launches=3, observed_nccl_kernels=4, observed_spin_kernels=3,
            scoped_proofs=[dict(scope_count=1, kernels=[{'name': 'observed'}]) for _ in range(3)])
        self.assertTrue(profile_valid(MODES[0], counts))
        self.assertTrue(profile_valid(MODES[1], dict(counts, scoped_proofs=[])))
        for field in ('cuda_graph_launches', 'observed_nccl_kernels', 'observed_spin_kernels'):
            self.assertFalse(profile_valid(MODES[0], dict(counts, **{field: 0})))
        self.assertFalse(profile_valid(MODES[0], dict(counts,
            scoped_proofs=[dict(scope_count=1, kernels=[])])))

    def test_crashed_profiles_remain_explicit_not_successful_measurements(self):
        source = (ROOT / 'runtime/controller.py').read_text()
        self.assertIn('profile_valid=False, result_missing=True', source)
        self.assertIn('valid_profile_modes=valid_profiles', source)
        self.assertIn('diagnostic_cases_completed', source)


if __name__ == '__main__':
    unittest.main()
