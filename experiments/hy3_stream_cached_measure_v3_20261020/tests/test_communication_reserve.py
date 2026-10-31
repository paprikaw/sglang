import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from communication_reserve import reserve


class Reserve(unittest.TestCase):
    def runner(self, shapes):
        self.calls = []
        def buffer():
            self.calls.append('allocate')
            return 'actual buffer'
        modules = [SimpleNamespace(dispatcher=SimpleNamespace(
            _low_latency_dispatcher=SimpleNamespace(
                hidden_size=h, params_bytes=b,
                num_max_dispatch_tokens_per_rank=t, num_experts=e,
                _get_buffer=buffer))) for h, b, t, e in shapes]
        return SimpleNamespace(model=SimpleNamespace(modules=lambda: modules))

    def test_once_for_all_same_shape_layers(self):
        self.assertEqual(reserve(self.runner([(4096, 2, 256, 192)] * 79)), 'actual buffer')
        self.assertEqual(self.calls, ['allocate'])

    def test_tbo_wrapper_is_unwrapped(self):
        runner = self.runner([(4096, 2, 256, 192)])
        module = list(runner.model.modules())[0]
        module.dispatcher = SimpleNamespace(_inners=[module.dispatcher])
        runner.model.modules = lambda: [module]
        self.assertEqual(reserve(runner), 'actual buffer')

    def test_missing_or_mixed_shape_fails_before_allocation(self):
        for shapes in ([], [(4096, 2, 16, 192)],
                       [(4096, 2, 256, 192), (4096, 2, 256, 160)]):
            runner = self.runner(shapes)
            with self.assertRaises(RuntimeError):
                reserve(runner)
            self.assertEqual(self.calls, [])
