import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from layouts import GLOBAL_LOGICAL_TOKENS, grid, layout, routes, summarize


class MatchedExpertWorkTests(unittest.TestCase):
    def test_exact_same_local_work_and_half_microbatch(self):
        for d in (1, 4):
            for n in GLOBAL_LOGICAL_TOKENS:
                ep = layout(n, 'ep8', duplication=d)
                slots = [layout(n, 'pp2_ep4', s, d) for s in (0, 1)]
                for pp in slots:
                    self.assertEqual(pp['microbatch_logical_tokens'] * 2, n)
                    self.assertEqual(pp['local_experts'], ep['local_experts'] * 2)
                    self.assertEqual(ep['local_assignments'], pp['local_assignments'])
                    self.assertEqual(ep['gemm_flops'], pp['gemm_flops'])
                    self.assertEqual(ep['local_assignments'], d * n)
                    self.assertLessEqual(max(pp['counts']) - min(pp['counts']), d)

    def test_partition_counts_match_global_top8_routing(self):
        for n in GLOBAL_LOGICAL_TOKENS:
            total = Counter(e for t in range(n) for e in routes(t))
            self.assertEqual(sum(total.values()), 8 * n)
            self.assertLessEqual(max(total.get(e, 0) for e in range(160)) -
                                 min(total.get(e, 0) for e in range(160)), 1)
            for t in range(n):
                self.assertEqual(len(set(routes(t))), 8)
            # The two PP slots recover exactly the full batch's counts for
            # their 40 local experts, including a non-divisible final block.
            s0, s1 = (layout(n, 'pp2_ep4', s)['counts'] for s in (0, 1))
            self.assertEqual([a + b for a, b in zip(s0, s1)],
                             [4 * total.get(e, 0) for e in range(40)])

    def test_active_weights_need_not_double_for_small_decode(self):
        self.assertEqual(layout(8, 'ep8')['active_experts'], 8)
        self.assertEqual(layout(8, 'pp2_ep4')['active_experts'], 8)
        self.assertEqual(layout(128, 'ep8')['active_experts'], 20)
        self.assertEqual(layout(128, 'pp2_ep4')['active_experts'], 40)

    def test_grid_and_production_tuning_estimate(self):
        self.assertEqual(len(grid()), 18)
        self.assertEqual(layout(4096, 'ep8')['expected_m'], 820)
        self.assertEqual(layout(4096, 'pp2_ep4')['expected_m'], 410)
        self.assertLessEqual(max(max(p['counts']) for p in grid()), 1024)

    def test_summary_and_invalid_geometry(self):
        self.assertEqual(summarize([1., 2., 3., 4.])['median_ms'], 2.5)
        for args in [(3, 'ep8'), (0, 'ep8'), (8, 'bad'), (8, 'pp2_ep4', 2)]:
            with self.assertRaises(ValueError):
                layout(*args)
        with self.assertRaises(ValueError):
            summarize([])

    def test_worker_refuses_missing_lease_before_importing_gpu_runtime(self):
        import sys
        import worker
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(KeyError):
                worker.authenticate(Path('/tmp/not-an-admitted-run'))
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
