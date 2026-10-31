"""Meaningful CPU ownership and balanced-rank/capture admission checks."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parents[1]/'.engine-worktrees/qwen35b-decode-ep-overlap-20261027/python/sglang/srt/layers/moe/decode_overlap.py'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=load('decode_overlap_cpu',SOURCE)
class Admission(unittest.TestCase):
    def test_idle_rank_bypasses_uniformly(self):
        for rows in ([0,16],[16,0],[1,16],[16,1],[]):self.assertFalse(m.eligible(16,rows))
    def test_real_balanced_and_unequal_sources(self):
        for rows in ([2,2],[4,4],[8,8],[16,16],[2,16]):self.assertTrue(m.eligible(min(rows),rows))
    def test_capture_padding_and_singleton(self):
        self.assertTrue(m.eligible(2,None,True));self.assertFalse(m.eligible(1,[32,32],True))
    def test_capture_context_retires(self):
        token=m._GRAPH_BUILD.set(True);self.assertTrue(m._GRAPH_BUILD.get());m._GRAPH_BUILD.reset(token);self.assertFalse(m._GRAPH_BUILD.get())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path);args=ap.parse_args()
    suite=unittest.TestSuite()
    for name,path in [('gap_cpu',ROOT.parent/'h100_pp_lookahead_workloads_20261027/test_gap.py'),
                      ('prefill_ownership_cpu',ROOT.parent/'h100_ep_prefill_overlap_20261027/validate_cpu.py')]:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(load(name,path)))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Admission))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    for path in list(ROOT.glob('*.py'))+[SOURCE]:ast.parse(path.read_text(),filename=str(path))
    receipt=dict(passed=result.wasSuccessful(),tests=result.testsRun,source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                 scope='PP lookahead ownership and boundaries, two independent DeepEP handles, fixed graph capture context and synchronized rank admission')
    dst=(args.run/'decode_overlap_cpu_qualification.json') if args.run else ROOT/'source_cpu_qualification.json'
    dst.write_text(json.dumps(receipt,indent=2)+'\n')
    if args.run:
        (args.run/'cpu_qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
