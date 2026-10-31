"""CPU-only checks for dashboard NIC/PCIe evidence and cacheless sampling."""
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


DASHBOARD = Path(__file__).resolve().parents[2] / 'experiments/idle_gpu_recovery_20261020/dashboard'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hardware = load('resource_hardware_test_target', DASHBOARD / 'hardware_inventory.py')
overview = load('resource_overview_test_target', DASHBOARD / 'resource_overview.py')


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class HardwareEvidenceTests(unittest.TestCase):
    def test_actual_link_and_capability_remain_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            device = Path(folder)
            for name, value in {'current_link_speed': '8.0 GT/s PCIe',
                                'current_link_width': '8',
                                'max_link_speed': '16.0 GT/s PCIe',
                                'max_link_width': '16'}.items():
                write(device / name, value)
            result = hardware.link(device)
        self.assertEqual(result['pcie_generation'], 3)
        self.assertEqual(result['link_width'], 8)
        self.assertEqual(result['max_pcie_generation'], 4)
        self.assertEqual(result['max_link_width'], 16)

    def test_unknown_current_never_inherits_capability(self):
        with tempfile.TemporaryDirectory() as folder:
            device = Path(folder)
            write(device / 'current_link_speed', 'Unknown speed')
            write(device / 'max_link_speed', '16.0 GT/s PCIe')
            write(device / 'max_link_width', '16')
            result = hardware.link(device)
        self.assertIsNone(result['pcie_generation'])
        self.assertIsNone(result['link_speed_gt_s'])
        self.assertIsNone(result['link_width'])
        self.assertEqual(result['max_pcie_generation'], 4)
        self.assertEqual(result['max_link_width'], 16)
        self.assertIsNone(hardware.generation(12.0))
        for invalid in ('N/A', 'Unknown speed 16 GT/s', '-1', '0', 'inf'):
            self.assertIsNone(hardware.number(invalid), invalid)

    def test_sysfs_nic_model_ids_parent_link_and_duplicate_hca(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bridge = root / 'devices/pci0000:b0/0000:b0:04.0'
            device = bridge / '0000:b1:00.0'
            for name, value in {'vendor': '0x15b3', 'device': '0x101d',
                                'current_link_speed': '16.0 GT/s PCIe',
                                'current_link_width': '8',
                                'max_link_speed': '16.0 GT/s PCIe',
                                'max_link_width': '16'}.items():
                write(device / name, value)
            write(bridge / 'current_link_speed', '8.0 GT/s PCIe')
            write(bridge / 'current_link_width', '8')
            for hca in ('mlx5_0', 'mlx5_bond_0'):
                hca_path = root / 'class/infiniband' / hca
                hca_path.mkdir(parents=True)
                (hca_path / 'device').symlink_to(device, target_is_directory=True)
            rows = hardware.nics(root)
        self.assertEqual(len(rows), 1)
        nic = rows[0]
        self.assertEqual(nic['pci_address'], '0000:b1:00.0')
        self.assertEqual(nic['model'], 'ConnectX-6 Dx')
        self.assertEqual(nic['pcie_generation'], 4)
        self.assertEqual(nic['link_width'], 8)
        self.assertEqual(nic['max_link_width'], 16)
        self.assertEqual(nic['ancestors'][0]['pcie_generation'], 3)

    def test_pci_description_and_id_fallback_do_not_infer_from_node(self):
        self.assertEqual(hardware.model_name(
            'Ethernet controller: Mellanox MT28800 Family [ConnectX-5 Ex]',
            '0x15b3', '0x1019'), 'ConnectX-5 Ex')
        self.assertEqual(hardware.model_name(None, '0x15b3', '0x1019'), 'ConnectX-5 Ex')
        self.assertEqual(hardware.model_name(None, '0xffff', '0x1019'), '0xffff / 0x1019')
        self.assertIsNone(hardware.model_name(None, None, None))

    def test_gpu_index_and_na_preserve_partial_evidence(self):
        rows = hardware.gpus('0, 00000000:17:00.0, N/A, 4, N/A, 16\n'
                             '1, 00000000:65:00.0, 3, 4, 8, 16\n'
                             'N/A, missing, 4, 4, 16, 16\n'
                             '2, too,few\n')
        self.assertEqual([row['index'] for row in rows], [0, 1])
        self.assertIsNone(rows[0]['pcie_generation'])
        self.assertIsNone(rows[0]['pcie_width'])
        self.assertEqual(rows[0]['max_pcie_generation'], 4)
        self.assertEqual(rows[0]['max_pcie_width'], 16)
        self.assertEqual(rows[1]['pcie_generation'], 3)
        self.assertEqual(rows[1]['pcie_width'], 8)


    def test_gpu_utilization_preserves_zero_and_rejects_invalid_readings(self):
        rows = hardware.gpus('0, pci0, 4, 4, 16, 16, 0, 0, 81920\n'
                             '1, pci1, 4, 4, 16, 16, 100, 1000, 81920\n'
                             '2, pci2, 4, 4, 16, 16, 101, -1, N/A\n'
                             '3, pci3, 4, 4, 16, 16, nan, inf, 81920\n')
        self.assertEqual([r['utilization_gpu_pct'] for r in rows], [0, 100, None, None])
        self.assertEqual(rows[0]['memory_used_mib'], 0)
        self.assertIsNone(rows[2]['memory_used_mib'])
        self.assertIsNone(rows[2]['memory_total_mib'])
        self.assertIsNone(rows[3]['memory_used_mib'])


class SnapshotHardwareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = dict(fleet_root=str(self.root / 'fleet'),
                        registry_root=str(self.root / 'registry'),
                        kv_registry=str(self.root / 'kv.json'), squeue='fixture-squeue')
        write(self.root / 'registry/gpu_registry/registry.json', '{"leases":[]}')
        write(self.root / 'registry/cpu_weight_daemons/registry.json', '{"entries":[]}')
        write(self.root / 'fleet/all_nodes_state.json', '{"nodes":{}}')
        self.calls = []

    def subprocess(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[0] == 'fixture-squeue':
            return subprocess.CompletedProcess(args, 0, '123|RUNNING|spartan-gpgpu065|6-00:00:00|gres/gpu:A100:4\n', '')
        if args[0] == 'ssh':
            return subprocess.CompletedProcess(args, 0, json.dumps(self.live), '')
        raise AssertionError('Unexpected subprocess: ' + repr(args))

    def sample(self, hardware_record):
        self.live = dict(sampled_at=1000, weights=None, kv=[], hardware=hardware_record)
        with patch.object(overview.subprocess, 'run', side_effect=self.subprocess):
            return overview.snapshot(self.cfg)

    def test_booked_node_without_model_cache_still_probes_hardware(self):
        result = self.sample(dict(status='current', node='spartan-gpgpu065',
                                  sampled_at=1000, nics=[dict(model='ConnectX-6 Dx')], gpus=[]))
        row = result['nodes'][0]
        self.assertEqual(row['hardware']['status'], 'current')
        self.assertEqual(row['hardware']['job_id'], '123')
        self.assertEqual(row['weights']['status'], 'unknown')
        ssh_calls = [(args, kwargs) for args, kwargs in self.calls if args[0] == 'ssh']
        self.assertEqual(len(ssh_calls), 1)
        args, kwargs = ssh_calls[0]
        self.assertIn('spartan-gpgpu065', args)
        self.assertIn('python3 -B', args[-1])
        payload = json.loads(kwargs['input'])
        self.assertEqual(payload['bank'], {})
        self.assertEqual(payload['hardware_probe'], str(DASHBOARD / 'hardware_inventory.py'))

    def test_hardware_from_another_node_is_rejected(self):
        result = self.sample(dict(status='current', node='spartan-gpgpu069',
                                  sampled_at=1000, nics=[dict(model='wrong-node')], gpus=[]))
        hardware_record = result['nodes'][0]['hardware']
        self.assertEqual(hardware_record['status'], 'unknown')
        self.assertEqual(hardware_record['error'], 'NodeIdentityMismatch')
        self.assertEqual(hardware_record['nics'], [])
        self.assertEqual(result['nodes'][0]['gpu_telemetry']['status'], 'unknown')

    def test_foreground_node_still_collects_physical_gpu_utilization(self):
        write(self.root / 'registry/gpu_registry/registry.json', json.dumps(dict(leases=[dict(
            nodes=['spartan-gpgpu065'], state='active', heartbeat_at=1000, ttl_s=15,
            experiment_id='my-experiment', resource_priority='normal')])) )
        record = dict(status='current', node='spartan-gpgpu065', sampled_at=1000,
                      nics=[], gpus=[dict(index=0, utilization_gpu_pct=0),
                                     dict(index=1, utilization_gpu_pct=100)])
        with patch.object(overview.time, 'time', return_value=1000):
            row = self.sample(record)['nodes'][0]
        self.assertEqual(row['state'], 'foreground')
        self.assertEqual(row['gpu_telemetry']['status'], 'current')
        self.assertEqual(row['gpu_utilization_pct'], 50)
        self.assertEqual(row['gpu_telemetry']['gpus'][0]['utilization_gpu_pct'], 0)

    def test_gpu_freshness_uses_sample_time_and_does_not_refresh_on_http_read(self):
        old = dict(timestamp=1000, nodes=[dict(node='spartan-gpgpu065',
                   hardware=dict(status='current', sampled_at=1000, nics=[], gpus=[]),
                   gpu_telemetry=dict(status='current', sampled_at=909, utilization_pct=80, gpus=[]),
                   weights=dict(status='unknown'), kv={})], errors=[])
        with patch.object(overview.time, 'time', return_value=1000):
            row = overview.refresh_ownership(old, self.cfg)['nodes'][0]
        self.assertEqual(row['gpu_telemetry']['status'], 'historical')
        self.assertEqual(row['gpu_telemetry']['sampled_at'], 909)
        self.assertEqual(old['nodes'][0]['gpu_telemetry']['status'], 'current')

    def test_missing_recipe_does_not_claim_full_weight_bank_is_incomplete(self):
        write(self.root / 'fleet/all_nodes_state.json', json.dumps(dict(nodes={
            'spartan-gpgpu065': dict(state_reason='waiting_for_compatible_recipe')})))
        row=self.sample(dict(status='current',node='spartan-gpgpu065',sampled_at=1000,nics=[],gpus=[]))
        current=overview.refresh_ownership(row,self.cfg)['nodes'][0]
        self.assertEqual(current['state_label'],'等待匹配模型配置')
        self.assertEqual(current['state_reason'],'waiting_for_compatible_recipe')

    def test_invalid_or_missing_gpu_values_never_become_zero(self):
        sample = overview.gpu_telemetry(dict(status='current', sampled_at=1000,
            gpus=[dict(index=0, utilization_gpu_pct=None), dict(index=1, utilization_gpu_pct=101)]), 1000)
        self.assertEqual(sample['status'], 'unknown')
        self.assertIsNone(sample['utilization_pct'])

    def test_qualified_cache_keeps_existing_weight_status_rpc(self):
        run = self.root / 'cached-run'
        spec_path = self.root / 'node_spec.json'
        spec = dict(source_commit='abc123', engine_root='/fixture/engine', model_path='/fixture/Qwen')
        raw = json.dumps(spec).encode()
        spec_path.write_bytes(raw)
        binding = dict(daemon_id='cache-1', node_spec=dict(path=str(spec_path),
                       sha256=hashlib.sha256(raw).hexdigest()))
        write(run / 'dashboard_guard/model_cache_bindings.json', json.dumps(dict(nodes={'spartan-gpgpu065': binding})))
        write(self.root / 'fleet/all_nodes_state.json', json.dumps(dict(nodes={'spartan-gpgpu065': dict(run=str(run))})))
        bank = dict(node='spartan-gpgpu065', state='active', expected_job_id='123',
                    source_commit='abc123', daemon_id='cache-1', socket_path='/fixture/cache.sock',
                    manifest=dict(pid=99), cache_scheme='canonical_full_weight',
                    status=dict(entries=[dict(key='weights', state='ready')]))
        write(self.root / 'registry/cpu_weight_daemons/registry.json', json.dumps(dict(entries=[bank])))
        self.live = dict(sampled_at=1000, kv=[], hardware=dict(status='current', node='spartan-gpgpu065', nics=[], gpus=[]),
                         weights=dict(ok=True, current_bytes=4096, entries=[dict(key='weights', state='ready')]))
        with patch.object(overview.subprocess, 'run', side_effect=self.subprocess):
            result = overview.snapshot(self.cfg)
        self.assertEqual(result['nodes'][0]['weights']['status'], 'ready')
        self.assertEqual(result['nodes'][0]['weights']['model'], 'Qwen')
        self.assertEqual(result['nodes'][0]['weights']['bytes'], 4096)
        self.assertEqual(result['nodes'][0]['weights']['kind'], '完整')
        args, kwargs = next((args, kwargs) for args, kwargs in self.calls if args[0] == 'ssh')
        payload = json.loads(kwargs['input'])
        self.assertEqual(payload['bank']['manifest'], dict(pid=99))
        self.assertEqual(payload['bank']['socket_path'], '/fixture/cache.sock')
        self.assertEqual(payload['engine'], '/fixture/engine')

    def test_ssh_failure_reports_unknown_hardware(self):
        def failed(args, **kwargs):
            if args[0] == 'ssh':
                raise subprocess.CalledProcessError(255, args, stderr='unreachable')
            return self.subprocess(args, **kwargs)
        with patch.object(overview.subprocess, 'run', side_effect=failed):
            result = overview.snapshot(self.cfg)
        row = result['nodes'][0]
        self.assertEqual(row['hardware']['status'], 'unknown')
        self.assertEqual(row['hardware']['error'], 'CalledProcessError')
        self.assertEqual(row['weights']['status'], 'unknown')

    def test_refresh_marks_old_hardware_historical_without_mutating_sample(self):
        hardware_record = dict(status='current', sampled_at=879, nics=[dict(model='ConnectX-6 Dx')], gpus=[])
        old = dict(timestamp=1000, nodes=[dict(node='spartan-gpgpu065', hardware=hardware_record,
                    weights=dict(status='unknown'), kv=dict(runtime_usage_pct=None, runtime_used_tokens=None))], errors=[])
        with patch.object(overview.time, 'time', return_value=1000):
            result = overview.refresh_ownership(old, self.cfg)
        self.assertEqual(result['nodes'][0]['hardware']['status'], 'historical')
        self.assertEqual(result['nodes'][0]['hardware']['nics'][0]['model'], 'ConnectX-6 Dx')
        self.assertEqual(old['nodes'][0]['hardware']['status'], 'current')

    def test_remote_hardware_probe_runs_when_bank_is_empty(self):
        helper = self.root / 'hardware_probe.py'
        helper.write_text("def probe():\n return {'status':'current','node':'fixture-node','nics':[],'gpus':[]}\n")
        payload = dict(hardware_probe=str(helper), bank={}, engine=None, kv=[])
        with patch('sys.stdin', io.StringIO(json.dumps(payload) + '\n')), patch('sys.stdout', new_callable=io.StringIO) as output:
            exec(overview.REMOTE, {})
        result = json.loads(output.getvalue())
        self.assertEqual(result['hardware']['node'], 'fixture-node')
        self.assertEqual(result['hardware']['status'], 'current')
        self.assertIsNone(result['weights'])
        self.assertEqual(result['weight_error'], 'KeyError')


if __name__ == '__main__':
    unittest.main()
