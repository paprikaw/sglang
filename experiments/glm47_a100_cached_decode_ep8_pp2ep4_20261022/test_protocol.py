"""CPU checks of workload, topology and timing boundaries (no model imports)."""
import sys,unittest,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runtime'))
import common,workload,measure
import summarize
class ProtocolTests(unittest.TestCase):
    def test_decode_receipt_mode_serialization(self):
        for mode in [2,'2','DECODE','ForwardMode.DECODE']:
            self.assertTrue(summarize.is_decode_mode(mode))
        for mode in [1,'1','EXTEND','ForwardMode.EXTEND']:
            self.assertFalse(summarize.is_decode_mode(mode))
    def test_workload_disjoint_pages_and_tail(self):
        seqs=workload.prefixes(81920)
        self.assertEqual(sum(map(len,seqs)),655360)
        self.assertEqual(len(set(tuple(x[:64]) for x in seqs)),8)
        self.assertEqual([len(workload.page_hashes(x)) for x in seqs],[1280]*8)
        self.assertTrue(all(workload.query(i,1,True)!=workload.query(i,1,False) for i in range(8)))
    def test_attention_and_expert_layout(self):
        cfg=dict(nodes=['n0','n1'],dist_port=10001,api_port=10002)
        for layout,tp,dp,ep,pp,chunk in [('ep8','8','2','8','1','2048'),('pp2ep4','4','1','4','2','1024')]:
            argv=common.server_args(dict(cfg,layout=layout),1)
            for flag,want in [('--tp-size',tp),('--dp-size',dp),('--ep-size',ep),('--pp-size',pp),('--chunked-prefill-size',chunk)]:
                self.assertEqual(argv[argv.index(flag)+1],want)
            self.assertNotIn('--max-total-tokens',argv)
            self.assertNotIn('--hicache-storage-backend',argv)
            self.assertIn('--disable-cuda-graph',argv)
            self.assertIn('--disable-overlap-schedule',argv)
    def test_usage_rank_partition(self):
        self.assertEqual(common.expected_ranks('ep8',1),['TP4','TP5','TP6','TP7'])
        self.assertEqual(common.expected_ranks('pp2ep4',1),['PP1 TP0','PP1 TP1','PP1 TP2','PP1 TP3'])
    def test_small_sample_percentile_labeled(self):
        d=measure.stats(list(range(8)))
        self.assertEqual(d['mean'],3.5);self.assertEqual(d['median'],3.5);self.assertEqual(d['p99'],7)
        self.assertEqual(d['p99_estimator'],'nearest_rank_N8')
    def test_decode_only_and_cpu_to_gpu_timed(self):
        source=(ROOT/'runtime/measure.py').read_text()
        self.assertIn("for phase in ('decode',):",source)
        self.assertIn('hicache_h2d_timed=True',source)
        self.assertIn("command(run,cfg,p,'cold'",source)
        for p in (ROOT/'runtime').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
if __name__=='__main__':unittest.main()
