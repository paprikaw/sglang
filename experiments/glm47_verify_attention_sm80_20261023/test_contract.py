"""CPU-only tests of the explicitly authorized, isolated Guard contract."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'guard_candidate/scripts'))
import model_cache_adapter as adapter


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((ROOT / 'run_v1/dashboard_spec.json').read_text())

    def test_valid(self):
        adapter.validate_contract(self.spec, launch=True)
        adapter.validate_command(self.spec, self.spec['model_cache']['command'])

    def test_reject_expansion(self):
        changes = [
            lambda s: s['workload'].update(expert_compute=True),
            lambda s: s['workload'].update(model_loaded=True),
            lambda s: s['workload'].update(batch_size=8),
            lambda s: s['tasks'][0].update(nodes=['spartan-gpgpu167']),
            lambda s: s['tasks'].append(copy.deepcopy(s['tasks'][0])),
            lambda s: s['model_cache']['command'].append('--unreviewed'),
            lambda s: s['model_cache'].update(policy='not_applicable'),
            lambda s: s.update(hicache_size=32),
        ]
        for change in changes:
            candidate = copy.deepcopy(self.spec)
            change(candidate)
            with self.subTest(change=repr(change)), self.assertRaises((ValueError, RuntimeError)):
                adapter.validate_contract(candidate, launch=True)


if __name__ == '__main__':
    unittest.main()
