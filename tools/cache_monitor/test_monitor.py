"""CPU-only fixtures for cache accounting and read-only boundaries."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location('cache_monitor_' + name, HERE / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


probe = module('probe')
server = module('server')


class ObserverTests(unittest.TestCase):
    def test_rank_geometry(self):
        self.assertEqual(probe.rank_identity('abc_3_4.bin'), (3, 4, 1, 0))
        self.assertEqual(probe.rank_identity('abc_2_4_2_1.bin'), (2, 4, 2, 1))
        self.assertIsNone(probe.rank_identity('unclassified.bin'))

    def test_unknown_weight_is_not_zero(self):
        value = probe.observe_weight(None, 'unused', {'no_swap': True})
        self.assertIsNone(value['bytes'])
        self.assertIsNone(value['resident_bytes'])

    def test_missing_pid_does_not_prove_no_live_readers(self):
        with patch.object(probe, 'process_start', side_effect=FileNotFoundError):
            value = probe.observe_weight({'manifest': {'pid': 10}, 'socket_path': '/missing/cache.sock',
                                          'max_bytes': 100}, 'unused', {'no_swap': True})
        self.assertIsNone(value['bytes'])
        self.assertEqual(value['reserved_bytes'], 100)

    def test_missing_swap_fields_are_not_residency_proof(self):
        with patch.object(Path, 'read_text', return_value='MemTotal: 10 kB\nMemAvailable: 5 kB\n'):
            self.assertFalse(probe.read_memory()['no_swap'])

    def test_empty_and_missing_ram_bank(self):
        with tempfile.TemporaryDirectory(dir='/dev/shm', prefix='cache-monitor-test-') as temp:
            result = probe.observe_dataset({'root': temp}, {'no_swap': True}, 'boot', set(), float('inf'))
            self.assertEqual(result['logical_bytes'], 0)
            self.assertEqual(result['resident_bytes'], 0)
            missing = probe.observe_dataset({'root': temp + '/missing'}, {'no_swap': True}, 'boot', set(), float('inf'))
            self.assertEqual(missing['status'], 'missing')

    def test_inode_dedup_unclassified_and_boot_layout(self):
        with tempfile.TemporaryDirectory(dir='/dev/shm', prefix='cache-monitor-test-') as temp:
            root = Path(temp)
            (root / 'a_0_4_2_1.bin').write_bytes(b'abc')
            (root / 'unknown.bin').write_bytes(b'xy')
            (root / 'b_0_4_2_1.bin').hardlink_to(root / 'a_0_4_2_1.bin')
            d = dict(root=temp, boot_id='old', layout={'pp': 1}, layer_partition=[92])
            result = probe.observe_dataset(d, {'no_swap': True}, 'new', set(), float('inf'))
            self.assertEqual(result['logical_bytes'], 5)
            self.assertEqual(result['files'], 2)
            self.assertEqual(result['unclassified_bytes'], 2)
            self.assertFalse(result['registered_boot_matches'])
            self.assertIsNone(result['rank_entries'][0]['layer_start'])
            self.assertEqual(result['layout']['pp'], 2)  # Native file geometry, not old registry.
            swapped = probe.observe_dataset(d, {'no_swap': False}, 'new', set(), float('inf'))
            self.assertIsNone(swapped['resident_bytes'])

    def test_parent_symlink_rejected(self):
        with tempfile.TemporaryDirectory(dir='/dev/shm', prefix='cache-monitor-test-') as temp:
            root = Path(temp)
            (root / 'actual').mkdir()
            (root / 'alias').symlink_to(root / 'actual', target_is_directory=True)
            value = probe.observe_dataset({'root': str(root / 'alias')}, {'no_swap': True}, '', set(), float('inf'))
            self.assertEqual(value['status'], 'unverified')
            self.assertIsNone(value['logical_bytes'])

    def test_scan_failure_is_not_zero(self):
        with tempfile.TemporaryDirectory(dir='/dev/shm', prefix='cache-monitor-test-') as temp:
            with patch.object(probe.os, 'walk', side_effect=PermissionError):
                result = probe.observe_dataset({'root': temp}, {'no_swap': True}, '', set(), float('inf'))
            self.assertFalse(result['observation_complete'])
            self.assertIsNone(result['logical_bytes'])

    def test_aliases_are_one_bank_with_layout_candidates(self):
        def entry(name, pp, state):
            return dict(checkpoint_id=name, state=state, last_verified_at=1,
                        compatibility=dict(instances=[dict(id='s', nodes=['n'], tp_size=4, pp_size=pp,
                                                             layer_partition=[46, 46] if pp == 2 else [92])]),
                        locations=[dict(node='n', path='/dev/shm/same-bank', instance_id='s', boot_id='boot')])
        result = server.dataset_descriptors({'entries': {'a': entry('a', 2, 'complete'),
                                                         'b': entry('b', 1, 'invalid')}})
        self.assertEqual(len(result['n']), 1)
        bank = next(iter(result['n'].values()))
        self.assertEqual(bank['checkpoint_ids'], ['a', 'b'])
        self.assertEqual(len(bank['layout_candidates']), 2)

    def test_cancelled_and_unknown_nodes_do_not_inflate_totals(self):
        raw = dict(node='n', status='live', weight=dict(bytes=None, resident_bytes=None),
                   kv=dict(logical_bytes=None, resident_bytes=None))
        with patch.object(server, 'running_bookings', side_effect=[{'n': ['1']}, {}]), \
             patch.object(server, 'read_registry', return_value={}), \
             patch.object(server, 'probe_node', return_value=raw):
            result = server.collect_snapshot()
        self.assertEqual(result['nodes'][0]['status'], 'unbooked')
        self.assertEqual(result['summary']['live_nodes'], 0)
        self.assertIsNone(result['summary']['weight_bytes'])

    def test_no_secrets_in_remote_payload(self):
        payload = server.node_request('n', dict(max_bytes=20, secret='hidden',
            token='capability', manifest=dict(pid=1, process_start_ticks=2, raw_environment='secret')), {})
        serialized = json.dumps(payload)
        self.assertNotIn('hidden', serialized)
        self.assertNotIn('capability', serialized)
        self.assertNotIn('raw_environment', serialized)


if __name__ == '__main__':
    unittest.main()
