"""No CUDA imports: test that the opt-in adapter rejects unqualified paths."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('verify_fix', ROOT / 'runtime/verify_as_decode.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SupportTest(unittest.TestCase):
    def setUp(self):
        self.kw = dict(sm=(8, 0), deterministic=True, topk=1, draft_tokens=3,
                       is_verify=True, causal=True, q_heads=24, kv_heads=2,
                       head_dim=128, value_dim=128, page_size=64,
                       q_dtype='torch.bfloat16', kv_dtype='torch.bfloat16',
                       sliding=-1, logit_cap=0., sinks=None, score_mod=None, temperature=-1)

    def test_exact_qualified_shape(self):
        self.assertTrue(module.supported(**self.kw))

    def test_unsupported_shapes_and_semantics_fall_back(self):
        for key, value in dict(sm=(9, 0), deterministic=False, topk=2,
                               draft_tokens=1, is_verify=False, causal=False,
                               q_heads=12, kv_heads=1, head_dim=256, value_dim=256,
                               page_size=1, kv_dtype='torch.float8_e4m3fn',
                               sliding=128, logit_cap=30., sinks=object(),
                               score_mod=object(), temperature=1024).items():
            with self.subTest(key=key):
                self.assertFalse(module.supported(**dict(self.kw, **{key: value})))


if __name__ == '__main__':
    unittest.main()
