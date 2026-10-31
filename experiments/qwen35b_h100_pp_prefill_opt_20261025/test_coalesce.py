import unittest
from coalesce import gather


class CoalesceTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
    def clock(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds
    def run_gather(self, first, polls, **kw):
        iterator = iter(polls)
        return gather(first, lambda: next(iterator, []), lambda x: isinstance(x, int),
                      clock=self.clock, sleep=self.sleep, **kw)
    def test_threshold_preserves_order_without_dropping_excess(self):
        self.assertEqual(self.run_gather([0], [[1,2], [3,4,5,6,7,8]]), list(range(9)))
    def test_partial_times_out(self):
        self.assertEqual(self.run_gather([0], [[1]]), [0,1])
        self.assertAlmostEqual(self.now, .005)
    def test_control_in_initial_poll_does_not_wait(self):
        self.assertEqual(self.run_gather([0, 'abort'], []), [0, 'abort'])
        self.assertEqual(self.now, 0)
    def test_control_in_later_poll_is_not_held(self):
        self.assertEqual(self.run_gather([0], [['flush']]), [0, 'flush'])
        self.assertLess(self.now, .001)
    def test_idle_and_full_do_not_wait(self):
        self.assertEqual(self.run_gather([], []), [])
        self.assertEqual(self.run_gather(list(range(8)), []), list(range(8)))
        self.assertEqual(self.now, 0)


if __name__ == '__main__':
    unittest.main()
