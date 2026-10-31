"""CPU-only checks: performance uses unchanged eligibility; diagnostic resets."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest


class SelectionTest(unittest.TestCase):
    def test_native_and_eager_are_explicit_and_reversible(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / 'policy'
            policy.write_bytes(b'0')
            policy.chmod(0o600)
            calls = []

            class Graph:
                def can_run_graph(self, batch):
                    calls.append(batch)
                    return batch == 'supported'

            module_name = 'sglang.srt.model_executor.runner.decode_cuda_graph_runner'
            previous = sys.modules.get(module_name)
            sys.modules[module_name] = types.SimpleNamespace(DecodeCudaGraphRunner=Graph)
            old_environment = os.environ.get('STEADY_EAGER_POLICY_PATH')
            os.environ['STEADY_EAGER_POLICY_PATH'] = str(policy)
            try:
                path = Path(__file__).with_name('diagnostic_switch.py')
                spec = importlib.util.spec_from_file_location('switch_under_test', path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.install()
                module.install()
                graph = Graph()
                self.assertTrue(graph.can_run_graph('supported'))
                self.assertFalse(graph.can_run_graph('unsupported'))
                policy.write_bytes(b'1')
                self.assertFalse(graph.can_run_graph('supported'))
                self.assertEqual(len(calls), 2)
                policy.write_bytes(b'0')
                self.assertTrue(graph.can_run_graph('supported'))
                policy.write_bytes(b'x')
                with self.assertRaises(RuntimeError):
                    graph.can_run_graph('supported')
                module._POLICY.close()
            finally:
                if previous is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = previous
                if old_environment is None:
                    os.environ.pop('STEADY_EAGER_POLICY_PATH', None)
                else:
                    os.environ['STEADY_EAGER_POLICY_PATH'] = old_environment


if __name__ == '__main__':
    unittest.main()
