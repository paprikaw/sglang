import unittest
from layouts import ACTIVE_EXPERTS, TOTAL_ROWS, point


class LayoutTests(unittest.TestCase):
    def test_fixed_work(self):
        for total in TOTAL_ROWS:
            points = [point(total, e) for e in ACTIVE_EXPERTS]
            self.assertEqual(len({p['gemm_flops'] for p in points}), 1)
            for p in points:
                self.assertEqual(sum(p['counts']), total)
                self.assertEqual(sum(c > 0 for c in p['counts']), p['active_experts'])
                self.assertEqual(len(set(p['active_ids'])), p['active_experts'])

    def test_fixed_resident_geometry(self):
        for total in TOTAL_ROWS:
            ps = [point(total, e) for e in ACTIVE_EXPERTS]
            self.assertEqual({len(p['counts']) for p in ps}, {40})
            self.assertEqual(len({p['expected_m'] for p in ps}), 1)


if __name__ == '__main__':
    unittest.main()
