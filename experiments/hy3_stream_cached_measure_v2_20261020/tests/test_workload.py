import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from workload import prefixes,query,page_hashes,near_capacity
from timing import extract_weight_load

class Workload(unittest.TestCase):
    def test_disjoint_deterministic_cache(self):
        rows=prefixes(1024)
        self.assertEqual(rows,prefixes(1024))
        self.assertEqual(len({page_hashes(x)[0] for x in rows}),8)
        self.assertTrue(all(len(page_hashes(x))==16 for x in rows))
    def test_uncached_warm_and_measured_queries(self):
        for i in range(8):
            self.assertNotEqual(query(i,256,True)[0],query(i,256,False)[0])
            self.assertEqual(len(query(i,256,False)),256)
    def test_page_aligned_with_generation_margin(self):
        p=near_capacity(81920,8)
        self.assertEqual(p%64,0)
        self.assertLessEqual(8*(p+384),int(.9*81920))
    def test_pp_global_load_identity(self):
        text='\n'.join(f'[t PP1 TP{i} EP{i}] Load weight begin.\n[t PP1 TP{i} EP{i}] Load weight end. elapsed=60.0 s' for i in range(4))
        self.assertEqual(extract_weight_load(text,[4,5,6,7])['rank_count'],4)
    def test_invalid_alignment_and_limit(self):
        for p in (0,65,262144):
            with self.assertRaises(ValueError):prefixes(p)

if __name__=='__main__':unittest.main()
