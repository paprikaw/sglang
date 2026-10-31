"""Regression for workload-copy aliasing and preserved cache ownership."""
import collections
import unittest

import agentx_replay_v2 as replay
import test_agentx_replay as prior

# Exercise the existing DAG, cache-prefix, and DP-rank checks on the new replay.
prior.replay = replay
ReplayTests = prior.ReplayTests


class CloneRoutingTests(unittest.TestCase):
    def test_sixteen_copies_of_eight_families_spread_each_family_evenly(self):
        lanes = {}
        counts = collections.Counter()
        # Match replay's lexical sorting of copy labels, including copy10.
        for cp in sorted(range(16), key=lambda x: f'copy{x}:'):
            for family in range(8):
                namespace = f'regression:copy{cp}:'
                sid = namespace + f'family{family}'
                lane = replay.assign_clone_balanced_lane(lanes, sid, namespace)
                counts[family, lane % 8] += 1
        self.assertEqual(counts, {(family, rank): 2 for family in range(8) for rank in range(8)})

    def test_session_cache_owner_survives_interleaved_new_sessions(self):
        lanes = {}
        first = replay.assign_clone_balanced_lane(lanes, 'trace:copy7:root', 'trace:copy7:')
        for i in range(20):
            replay.assign_clone_balanced_lane(lanes, f'trace:copy{i}:child', f'trace:copy{i}:')
        self.assertEqual(first, replay.assign_clone_balanced_lane(lanes, 'trace:copy7:root', 'trace:copy7:'))

    def test_copy_zero_preserves_existing_round_robin(self):
        lanes = {}
        got = [replay.assign_clone_balanced_lane(lanes, f'root{i}', 'test:copy0:') for i in range(32)]
        self.assertEqual(got, list(range(32)))


if __name__ == '__main__':
    unittest.main()
