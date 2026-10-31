"""CPU-only checks of mutually exclusive device timeline accounting."""
import unittest
from analyze import partition

class TimelineTests(unittest.TestCase):
    def test_idle_overlap_and_unknown_are_not_hidden(self):
        events=[dict(cat='kernel',name='A',ts=1000,dur=2000),
                dict(cat='kernel',name='E',ts=2000,dur=2000),
                dict(cat='kernel',name='new_kernel',ts=5000,dur=1000)]
        p=partition(events,0,10000,{'A':'attention','E':'expert'},'marker')
        self.assertEqual(p['attention'],1)
        self.assertEqual(p['expert'],1)
        self.assertEqual(p['overlap_multiple_categories'],1)
        self.assertEqual(p['unknown_gpu'],1)
        self.assertEqual(p['no_traced_gpu_activity'],6)
        self.assertEqual(sum(p.values()),10)
    def test_clipped_window_and_same_category_overlap(self):
        events=[dict(cat='kernel',name='A',ts=0,dur=5000),dict(cat='kernel',name='A',ts=1000,dur=5000)]
        self.assertEqual(partition(events,2000,4000,{'A':'attention'},'marker'),{'attention':2,'no_traced_gpu_activity':0})

if __name__=='__main__':unittest.main()
