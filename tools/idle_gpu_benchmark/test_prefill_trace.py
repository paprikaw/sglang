"""CPU tests for independent completion-driven prefill trace cursors."""
import csv
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
from prefill_trace import PrefillTrace


class PrefillTraceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'trace.csv'
        self.nodes = ['spartan-gpgpu065', 'spartan-gpgpu069']
        self.handles = []

    def tearDown(self):
        for stream in self.handles:
            stream.close()

    def write(self, rows):
        with self.source.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['Timestamp', 'Model', 'Request tokens', 'Response tokens'])
            writer.writerows(rows)

    def open(self, node=None, **kwargs):
        node = node or self.nodes[0]
        cursor = kwargs.pop('cursor_path', self.root / f'{node}.json')
        stream = PrefillTrace(self.source, cursor, node, kwargs.pop('nodes', self.nodes), **kwargs)
        self.handles.append(stream)
        return stream

    def test_fixed_partition_is_disjoint_balanced_and_chronological(self):
        self.write([(index, 'GPT-4', 100 + index, 10) for index in range(12)])
        first, second = [self.open(node, batch_size=6).peek_batch() for node in self.nodes]
        indices = [[row['source_index'] for row in batch['requests']] for batch in (first, second)]
        self.assertEqual(indices, [list(range(0, 12, 2)), list(range(1, 12, 2))])
        self.assertEqual(set(indices[0]) | set(indices[1]), set(range(12)))
        self.assertFalse(set(indices[0]) & set(indices[1]))
        self.assertTrue(all('offset_s' not in row for row in first['requests']))

    def test_long_original_output_does_not_filter_prefill(self):
        self.write([(0, 'GPT-4', 32767, 1000000), (1, 'ChatGPT', 12, 1)])
        batch = self.open(batch_size=1).peek_batch()
        row = batch['requests'][0]
        self.assertEqual((row['prompt_tokens'], row['output_tokens'], row['original_output_tokens']),
                         (32767, 1, 1000000))
        self.assertIsNone(batch['selection']['max_original_total_tokens'])
        self.assertEqual(batch['summary']['total_output_tokens'], 1)

    def test_context_exclusion_is_own_partition_and_durable_only_on_commit(self):
        self.write([(0, 'GPT-4', 32768, 1), (1, 'GPT-4', 10, 100000),
                    (2, 'GPT-4', 32767, 50000), (3, 'GPT-4', 40000, 2)])
        stream = self.open(batch_size=2)
        batch = stream.peek_batch()
        self.assertEqual([row['source_index'] for row in batch['requests']], [2])
        self.assertEqual(batch['stream']['excluded_for_context'], 1)
        self.assertEqual(stream.state['excluded_for_context'], 0)
        self.assertFalse(stream.cursor_path.exists())
        stream.commit()
        self.assertEqual(stream.state['excluded_for_context'], 1)
        self.assertEqual(stream.state['other_partition_rows'], 2)

    def test_retry_is_independent_copy_and_restart_keeps_uncommitted_batch(self):
        self.write([(index, 'GPT-4', index + 10, 1) for index in range(8)])
        stream = self.open(batch_size=1)
        initial = stream.peek_batch()
        initial['requests'][0]['prompt_tokens'] = -1
        self.assertEqual(stream.peek_batch()['requests'][0]['prompt_tokens'], 10)
        stream.commit()
        pending = stream.peek_batch()
        stream.close()
        restarted = self.open(batch_size=1)
        self.assertEqual(pending, restarted.peek_batch())
        self.assertEqual(restarted.state['committed_requests'], 1)

    def test_node_can_advance_while_other_node_stays_offline(self):
        self.write([(index, 'GPT-4', 10, 2) for index in range(8)])
        first = self.open(self.nodes[0], batch_size=1)
        self.assertEqual(first.peek_batch()['requests'][0]['source_index'], 0)
        first.commit()
        self.assertEqual(first.peek_batch()['requests'][0]['source_index'], 2)
        first.commit()
        second = self.open(self.nodes[1], batch_size=1)
        self.assertEqual(second.peek_batch()['requests'][0]['source_index'], 1)
        self.assertEqual(second.state['committed_requests'], 0)

    def test_empty_own_batch_can_commit_and_progress(self):
        self.write([(0, 'GPT-4', 40000, 1), (1, 'GPT-4', 10, 1),
                    (2, 'GPT-4', 11, 1), (3, 'GPT-4', 12, 1)])
        stream = self.open(batch_size=1)
        self.assertEqual(stream.peek_batch()['requests'], [])
        stream.commit()
        self.assertEqual(stream.state['committed_requests'], 0)
        self.assertEqual(stream.peek_batch()['requests'][0]['source_index'], 2)

    def test_one_node_partition_and_eof_cycle_preserve_source_identity(self):
        self.write([(0, 'GPT-4', 10, 1), (1, 'GPT-4', 20, 2), (2, 'GPT-4', 30, 3)])
        stream = self.open(nodes=[self.nodes[0]], batch_size=4)
        batch = stream.peek_batch()
        self.assertEqual([row['source_index'] for row in batch['requests']], [0, 1, 2])
        stream.commit()
        next_batch = stream.peek_batch()
        self.assertEqual([row['source_cycle'] for row in next_batch['requests']], [1, 1, 1])

    def test_invalid_source_rows_are_excluded_without_output_length_cap(self):
        self.write([(0, 'GPT-4', 10, 0), (1, 'GPT-4', 10, -1),
                    (2, 'GPT-4', 0, 1), (3, 'GPT-4', 12, 1), (4, 'GPT-4', 13, 90000)])
        stream = self.open(batch_size=4)
        batch = stream.peek_batch()
        self.assertEqual([row['source_index'] for row in batch['requests']], [4])
        stream.commit()
        self.assertEqual(stream.state['source_cursor']['exclusion_counts'],
                         {'failed_or_nonpositive_output': 2, 'nonpositive_prompt': 1})

    def test_cursor_is_private_metadata_only_and_changed_partition_rejected(self):
        self.write([(0, 'GPT-4', 10, 2), (1, 'GPT-4', 20, 3)])
        stream = self.open(batch_size=1)
        stream.peek_batch()
        stream.commit()
        self.assertEqual(stat.S_IMODE(stream.cursor_path.stat().st_mode), 0o600)
        raw = json.loads(stream.cursor_path.read_text())
        for forbidden in ('input_ids', 'requests', 'prompt', 'response', 'payload'):
            self.assertNotIn(forbidden, raw)
        with self.assertRaisesRegex(ValueError, 'partition/context differs'):
            self.open(batch_size=1, max_context_tokens=100)
        with self.assertRaisesRegex(ValueError, 'partition/context differs'):
            self.open(self.nodes[1], cursor_path=stream.cursor_path)

    def test_write_failure_cannot_advance_durable_cursor_on_restart(self):
        self.write([(index, 'GPT-4', 10, 1) for index in range(8)])
        stream = self.open(batch_size=1)
        stream.peek_batch()
        stream.commit()
        before = stream.cursor_path.read_bytes()
        expected = stream.peek_batch()
        with patch('prefill_trace._save', side_effect=OSError('disk unavailable')):
            with self.assertRaisesRegex(OSError, 'disk unavailable'):
                stream.commit()
        self.assertEqual(stream.cursor_path.read_bytes(), before)
        with self.assertRaisesRegex(RuntimeError, 'reopen'):
            stream.peek_batch()
        stream.close()
        self.assertEqual(self.open(batch_size=1).peek_batch(), expected)

    def test_source_hash_and_cursor_source_identity_are_checked(self):
        self.write([(0, 'GPT-4', 10, 1), (1, 'GPT-4', 20, 1)])
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        stream = self.open(expected_source_sha256=digest)
        stream.peek_batch()
        stream.commit()
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.open(expected_source_sha256='0' * 64)
        self.source.write_text(self.source.read_text() + '2,GPT-4,20,1\n')
        with self.assertRaisesRegex(ValueError, 'different source SHA256'):
            self.open()

    def test_commit_requires_pending_batch(self):
        self.write([(0, 'GPT-4', 10, 1)])
        with self.assertRaisesRegex(RuntimeError, 'no pending'):
            self.open().commit()


if __name__ == '__main__':
    unittest.main()
