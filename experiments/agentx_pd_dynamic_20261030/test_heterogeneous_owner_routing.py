"""CPU checks for the actual frozen replay router, without model imports."""
import collections
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
SEED = ROOT/'runs/two_p_six_d_ep4ep2_vs_3ep2_e8_v4_gdn256_20261031_pilot/code'
sys.path.insert(0, str(SEED))
spec = importlib.util.spec_from_file_location('qualified_replay_router', SEED/'agentx_replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
sys.path.insert(0, str(ROOT))
from heterogeneous_owner_routing import install
old_pick = install(replay.Router)
TURN = dict(input_tokens=8192, output_tokens=32, hash_ids=[])


def roots():
    # Replay sorts ready vertex keys lexically, including copy10 before copy2.
    for copy_id in sorted(range(16), key=str):
        for family in range(8):
            yield f'agentx-c128:copy{copy_id}:g{family}', copy_id, family


class OwnerRoutingTest(unittest.TestCase):
    def endpoints(self, degrees):
        return [dict(id=f'i{i}', ep=ep) for i, ep in enumerate(degrees)]

    def test_uniform_pool_and_prefill_unchanged(self):
        for degrees in ([1], [2, 2, 2], [4, 4]):
            a = replay.Router(self.endpoints(degrees), 'sticky_hash', 8, 16)
            b = replay.Router(self.endpoints(degrees), 'sticky_hash', 8, 16)
            for arrival, (sid, copy_id, _) in enumerate(roots()):
                lane = arrival + copy_id
                self.assertEqual(old_pick(a, sid, TURN, lane), b.pick(sid, TURN, lane))
            self.assertEqual(a.active, b.active)
            self.assertEqual(a.pending_output, b.pending_output)

    def test_old_unequal_pool_has_twice_load_per_ep2_owner(self):
        router = replay.Router(self.endpoints([2, 4]), 'sticky_hash', 8, 16)
        counts = collections.Counter()
        for arrival, (sid, copy_id, _) in enumerate(roots()):
            target, _ = old_pick(router, sid, TURN, arrival + copy_id)
            counts[target['id'], target['rank']] += 1
        self.assertEqual([counts['i0', rank] for rank in range(2)], [32, 32])
        self.assertEqual([counts['i1', rank] for rank in range(4)], [16]*4)

    def test_unequal_pool_balances_owners_and_each_family(self):
        router = replay.Router(self.endpoints([2, 4]), 'sticky_hash', 8, 16)
        counts, families = collections.Counter(), collections.defaultdict(collections.Counter)
        for arrival, (sid, copy_id, family) in enumerate(roots()):
            target, switched = router.pick(sid, TURN, arrival + copy_id)
            owner = target['id'], target['rank']
            self.assertFalse(switched)
            counts[owner] += 1
            families[family][owner] += 1
        self.assertEqual(len(counts), 6)
        # Finite copy blocks need not divide six owners. Bound the imbalance
        # while also ensuring that each source family reaches every owner.
        self.assertLessEqual(max(counts.values())-min(counts.values()), 2)
        for family in families.values():
            self.assertEqual(sorted(family.values()), [2, 2, 3, 3, 3, 3])

    def test_conversation_stays_on_same_owner_and_finish_balances(self):
        router = replay.Router(self.endpoints([2, 4]), 'sticky_hash', 8, 16)
        sid, lane = 'agentx-c128:copy10:child', 150
        targets = [router.pick(sid, TURN, lane) for _ in range(3)]
        self.assertTrue(all(target == targets[0] for target in targets))
        for target, switched in targets:
            self.assertFalse(switched)
            router.finish(sid, target, TURN, lane, dict(tpot_ms=7))
        self.assertTrue(all(value == 0 for value in router.active.values()))
        self.assertTrue(all(value == 0 for value in router.pending_output.values()))

    def test_physical_owner_and_handoff_path_match_three_ep2_control(self):
        a = replay.Router(self.endpoints([2, 4]), 'sticky_hash', 8, 16)
        b = replay.Router(self.endpoints([2, 2, 2]), 'sticky_hash', 8, 16)
        physical_a = {('i0', 0): 'local2', ('i0', 1): 'local3',
                      **{('i1', rank): f'remote{rank}' for rank in range(4)}}
        physical_b = {('i0', 0): 'local2', ('i0', 1): 'local3',
                      ('i1', 0): 'remote0', ('i1', 1): 'remote1',
                      ('i2', 0): 'remote2', ('i2', 1): 'remote3'}
        for arrival, (sid, copy_id, _) in enumerate(roots()):
            ta, _ = a.pick(sid, TURN, arrival + copy_id)
            tb, _ = old_pick(b, sid, TURN, arrival + copy_id)
            self.assertEqual(physical_a[ta['id'], ta['rank']], physical_b[tb['id'], tb['rank']])

    def test_low_load_uses_six_distinct_owners_before_repeating(self):
        router = replay.Router(self.endpoints([2, 4]), 'sticky_hash', 8, 1)
        owners = [router.pick(f'agentx-c8:copy0:g{i}', TURN, i)[0] for i in range(8)]
        self.assertEqual(len({(x['id'], x['rank']) for x in owners[:6]}), 6)
        self.assertEqual(sum(x['id'] == 'i0' for x in owners), 3)
        self.assertEqual(sum(x['id'] == 'i1' for x in owners), 5)


if __name__ == '__main__':
    unittest.main()
