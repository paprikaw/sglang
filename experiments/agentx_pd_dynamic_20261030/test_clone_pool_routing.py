"""Verify asymmetric P:D clone placement without changing the current matrix."""
import collections
import unittest

import agentx_replay_v2 as old
import agentx_replay_v3 as replay
import test_agentx_replay as prior

prior.replay = replay
ReplayTests = prior.ReplayTests


def endpoints(n, ep):
    return [dict(id=f'i{i}', ep=ep, pp=1) for i in range(n)]


class PoolCloneTests(unittest.TestCase):
    def test_six_decode_lanes_spread_every_family_in_lexical_copy_order(self):
        lanes = {}; counts = collections.Counter(); eps=endpoints(3,2)
        for cp in sorted(range(16),key=lambda x:f'copy{x}:'):
            for f in range(8):
                ns=f'test:copy{cp}:';sid=ns+f'family{f}'
                lane=replay.assign_clone_balanced_lane(lanes,sid,ns)
                adjusted=replay.pool_clone_lane(sid,lane,eps,8,16)
                counts[f,adjusted%6]+=1
        for f in range(8):
            self.assertEqual(sum(counts[f,r] for r in range(6)),16)
            self.assertLessEqual(max(counts[f,r] for r in range(6))-min(counts[f,r] for r in range(6)),1)

    def test_current_matrix_routes_unchanged_even_for_interleaved_children(self):
        for n,ep in ((2,4),(1,4),(4,1),(8,1),(4,2),(1,2)):
            eps=endpoints(n,ep)
            for i in range(200):
                cp=(i*7)%16; sid=f'test:copy{cp}:child{i}';lane=i+cp
                self.assertEqual(lane,replay.pool_clone_lane(sid,lane,eps,8,16))

    def test_unreplicated_round_robin_unchanged_for_six_lanes(self):
        for i in range(100):
            self.assertEqual(i,replay.pool_clone_lane('test:copy0:child',i,endpoints(3,2),8,1))

    def test_cache_affinity_survives_new_conversations(self):
        router=replay.Router(endpoints(3,2),'sticky_hash',8,16)
        turn=dict(hash_ids=[1,2],input_tokens=128,output_tokens=16)
        sid='test:copy10:root';lane=18
        a,_=router.pick(sid,turn,lane);router.finish(sid,a,turn,lane,{'tpot_ms':7})
        for i in range(40):
            router.pick(f'test:copy{i%16}:child{i}',turn,i+i%16)
        b,switched=router.pick(sid,turn,lane)
        self.assertEqual((a['id'],a['rank']),(b['id'],b['rank']))
        self.assertFalse(switched)


if __name__=='__main__':unittest.main()
