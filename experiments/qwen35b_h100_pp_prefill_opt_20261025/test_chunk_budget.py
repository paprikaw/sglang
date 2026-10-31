from types import SimpleNamespace
import unittest
from chunk_budget import wrap


class ChunkBudgetTests(unittest.TestCase):
    def test_each_frozen_mode_changes_both_native_budgets(self):
        scheduler=SimpleNamespace(ps=SimpleNamespace(pp_size=2))
        for mode,size in enumerate((4096,2048,1024,8192)):
            choose=wrap(lambda self,x:(self.chunked_prefill_size,self.max_prefill_tokens,x),
                        lambda:ord(str(mode)))
            self.assertEqual(choose(scheduler,'batch'),(size,size,'batch'))
    def test_non_pp_passthrough(self):
        obj=SimpleNamespace(ps=SimpleNamespace(pp_size=1))
        self.assertEqual(wrap(lambda self:42,lambda:ord('x'))(obj),42)
        self.assertFalse(hasattr(obj,'chunked_prefill_size'))
    def test_unknown_mode_fails(self):
        obj=SimpleNamespace(ps=SimpleNamespace(pp_size=2))
        with self.assertRaises(RuntimeError):wrap(lambda self:0,lambda:ord('x'))(obj)


if __name__=='__main__':unittest.main()
