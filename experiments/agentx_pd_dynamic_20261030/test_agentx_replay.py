import copy
import time
import unittest
from unittest.mock import patch

import agentx_replay as replay


def turn(index, **extra):
    result = dict(timestamp=index*1000, delay=0, api_time_ms=100,
                  source_outer_idx=index, source_inner_idx=None, source_kind='weka_main',
                  output_tokens=3, input_tokens=64, hash_ids=[1],
                  branch_ids=[], prerequisites=[], replay_predecessors=[])
    result.update(extra)
    return result


def fixture():
    return {'id': 'root', 'block_size': 64, 'conversations': [
        {'session_id': 'root', 'branches': [{'branch_id': 'spawn', 'child_conversation_ids': ['child'],
          'dispatch_timing': 'post', 'mode': 'spawn', 'is_background': False}],
         'turns': [turn(0, branch_ids=['spawn']),
                   turn(2, prerequisites=[{'kind': 'spawn_join', 'branch_id': 'spawn'}])]},
        {'session_id': 'child', 'branches': [], 'turns': [turn(1)]},
    ]}


class ReplayTests(unittest.TestCase):
    def test_replay_preserves_spawn_and_join_with_idle_compression(self):
        def serve(p, d, key, tokens, output, lane):
            start = time.perf_counter()
            time.sleep(.005)
            end = time.perf_counter()
            return dict(started=start, finished=end, ttft_ms=1., tpot_ms=2., e2e_ms=5.,
                        output_ids=[1]*output, output_tokens=output, meta_info={})
        ep = {'id': 'fake', 'ep': 1, 'role': 'colocated', 'host': 'unused', 'port': 0}
        with patch.object(replay, 'request_one', serve):
            result = replay.replay([fixture()], [ep], 'sticky_hash', 1, '', [10], lambda: None,
                                   namespace='test', idle_gap_cap_s=0)
        rows = {(r['conversation_id'], r['turn']): r for r in result['requests']}
        parent0 = rows['test:copy0:root', 0]
        child = rows['test:copy0:child', 0]
        parent1 = rows['test:copy0:root', 1]
        self.assertGreaterEqual(child['started'], parent0['finished'])
        self.assertGreaterEqual(parent1['started'], child['finished'])
        self.assertEqual(result['completed_requests'], 3)
        self.assertGreater(result['idle_compressed_s'], 0)

    def test_rejects_cross_stream_cycle(self):
        graph = fixture()
        graph['conversations'][0]['turns'][0]['replay_predecessors'] = [
            {'conversation_id': 'child', 'turn_index': 0}]
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            replay.build_vertices(graph, '')

    def test_block_prefix_is_reused_but_session_clones_do_not_alias(self):
        graph = fixture()
        cache = {}
        short = replay.token_ids(graph, turn(0), 'clone0', [99], cache)
        long = replay.token_ids(graph, turn(1, input_tokens=129, hash_ids=[1, 2]), 'clone0', [99], cache)
        other = replay.token_ids(graph, turn(0), 'clone1', [99], cache)
        self.assertEqual(short, long[:64])
        self.assertEqual(len(long), 129)
        self.assertNotEqual(short, other)

    def test_replica_and_attention_dp_assignment_use_all_ranks(self):
        endpoints = [{'id': str(i), 'ep': 2} for i in range(4)]
        router = replay.Router(endpoints, 'sticky_hash')
        assignments = [router.pick(str(i), turn(0), i)[0] for i in range(8)]
        self.assertEqual({(e['id'], e['rank']) for e in assignments},
                         {(str(i), r) for i in range(4) for r in range(2)})


if __name__ == '__main__':
    unittest.main()
