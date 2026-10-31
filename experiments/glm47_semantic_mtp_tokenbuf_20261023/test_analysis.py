"""CPU-only checks of mutually exclusive device timeline accounting."""
import unittest
from analyze import partition
from component_summary import average_partition, grouped

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
    def test_phase_averaging_retains_unknown_and_wait(self):
        result=average_partition([
            dict(envelope_ms=10,partition_ms=dict(attention=6,unknown_gpu=4)),
            dict(envelope_ms=12,partition_ms=dict(attention=4,pp_communication=8))])
        self.assertEqual(result['envelope_ms'],11)
        self.assertEqual(result['partition_ms'],dict(attention=5,pp_communication=4,unknown_gpu=2))
        groups=grouped(result['partition_ms'])
        self.assertEqual(groups['PP SendRecv including wait'],4)
        self.assertEqual(groups['Other / unknown / idle / overlap'],2)
        self.assertEqual(sum(groups.values()),11)
    def test_nonconserved_phase_is_not_published(self):
        with self.assertRaisesRegex(ValueError,'Nonconserved'):
            average_partition([dict(envelope_ms=10,partition_ms=dict(attention=6))])

if __name__=='__main__':unittest.main()
