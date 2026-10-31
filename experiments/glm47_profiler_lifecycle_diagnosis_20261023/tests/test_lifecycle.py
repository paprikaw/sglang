import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
from common import environment, trace_counts
from node_runner import command


class LifecycleTests(unittest.TestCase):
    def test_fresh_baseline_removes_inherited_flags(self):
        with patch.dict(os.environ, {'TEARDOWN_CUPTI': '1', 'DISABLE_CUPTI_LAZY_REINIT': '0'}):
            baseline = environment({'python': '/test/python'}, 'baseline', Path('/run'))
        self.assertNotIn('TEARDOWN_CUPTI', baseline)
        self.assertNotIn('DISABLE_CUPTI_LAZY_REINIT', baseline)
        self.assertEqual(baseline['CUDA_VISIBLE_DEVICES'], '0,1')

    def test_candidate_flags_precede_torch_import(self):
        candidate = environment({'python': '/test/python'}, 'preserve_cupti', Path('/run'))
        self.assertEqual(candidate['TEARDOWN_CUPTI'], '0')
        self.assertEqual(candidate['DISABLE_CUPTI_LAZY_REINIT'], '1')

    def test_command_is_bounded_two_gpu_fresh_process(self):
        result = command({'python': '/test/python', 'master_port': 12345}, '/run', 'baseline')
        self.assertIn('--nproc-per-node=2', result)
        self.assertIn('--max-restarts=0', result)
        self.assertIn('--nnodes=1', result)
        self.assertEqual(result[-1], 'baseline')

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
        self.assertIn('[ProfilerActivity.CUDA]', source)
        self.assertIn('bidirectional_payload_verified=True', source)


if __name__ == '__main__':
    unittest.main()
