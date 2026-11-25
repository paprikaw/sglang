"""Use the actual pinned AgentX tree ledger; no GPU/model is loaded."""
from types import SimpleNamespace
import unittest

from aiperf.common.enums import CacheBustTarget
from aiperf.timing.strategies import cache_bust
from paired_cache_bust import install


class PairedCacheBustTests(unittest.TestCase):
    def setUp(self):
        self.native = cache_bust.build_cache_bust_marker

    def tearDown(self):
        cache_bust.build_cache_bust_marker = self.native

    def ledger(self):
        return SimpleNamespace(session_marker={}, recycle_pass={})

    def marker(self, ledger, benchmark, root, *, lane=0, trace='trace', target=CacheBustTarget.FIRST_TURN_PREFIX):
        return cache_bust.resolve_tree_marker(ledger, root, benchmark_id=benchmark,
            trajectory_index=lane, conversation_id=trace, target=target)

    def test_independent_run_ids_preserve_identical_tree_marker_bytes(self):
        self.assertNotEqual(self.marker(self.ledger(), 'run-a', 'root-a'),
                            self.marker(self.ledger(), 'run-b', 'root-b'))
        install('a'*64)
        self.assertEqual(self.marker(self.ledger(), 'run-a', 'root-a'),
                         self.marker(self.ledger(), 'run-b', 'root-b'))

    def test_native_warmup_descendant_and_recycle_isolation_remain(self):
        install('a'*64)
        ledger=self.ledger()
        first=self.marker(ledger,'run-a','warm-root')
        self.assertEqual(first,self.marker(ledger,'run-a','warm-root',trace='trace::sa:0'))
        self.assertEqual(first,self.marker(ledger,'run-a','warm-root'))
        self.assertNotEqual(first,self.marker(ledger,'run-a','next-root'))
        self.assertNotEqual(first,self.marker(self.ledger(),'run-a','other-lane',lane=1))
        self.assertNotEqual(first,self.marker(self.ledger(),'run-a','other-trace',trace='other'))
        self.assertIsNone(self.marker(self.ledger(),'run-a','disabled',target=CacheBustTarget.NONE))

    def test_namespace_is_frozen_and_idempotent(self):
        install('a'*64);wrapper=cache_bust.build_cache_bust_marker
        install('a'*64)
        self.assertIs(wrapper,cache_bust.build_cache_bust_marker)
        with self.assertRaises(RuntimeError):install('b'*64)
        with self.assertRaises(ValueError):install('unfrozen')


if __name__=='__main__':
    unittest.main()
