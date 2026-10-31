"""Run the pinned native Decode constructor with allocation calls stubbed out."""
import ast
import logging
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT.parents[1] / '.engine-worktrees/agentx-gdn-pd-hicache-r5-20261031/python/sglang/srt/disaggregation/decode.py'


def allocate_shape(size, prealloc, configured):
    tree = ast.parse(NATIVE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HybridMambaDecodeReqToTokenPool')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))

    class DummyReqPool:
        def __init__(self, **kwargs):
            self.request_shape = kwargs

    class Capture:
        def _init_mamba_pool(self, **kwargs):
            self.state_shape = kwargs

    namespace = dict(DecodeReqToTokenPool=DummyReqPool, logger=logging.getLogger('native-slot-check'))
    exec(compile(module, str(NATIVE), 'exec'), namespace)
    result = Capture()
    namespace['__init__'](result, size=size, max_context_len=262144, device='cuda',
        enable_memory_saver=False, cache_params=None, mamba_layer_ids=[],
        speculative_num_draft_tokens=None, enable_mamba_extra_buffer=True,
        pre_alloc_size=prealloc, enable_overlap_schedule=True, mamba_size=configured)
    return result.state_shape['mamba_size']


class DecodeSlotBudget(unittest.TestCase):
    def test_original_defaults_reproduce_unbudgeted_growth(self):
        self.assertEqual(allocate_shape(16, 32, 80), 144)

    def test_ep4_eight_transfer_slots_preserve_declared_pool(self):
        self.assertEqual(allocate_shape(16, 8, 80), 80)

    def test_pp_transfer_queue_fits_stage_pool(self):
        self.assertEqual(allocate_shape(32, 8, 160), 160)

    def test_single_gpu_requires_its_own_budget(self):
        self.assertEqual(allocate_shape(8, 8, 40), 48)
        self.assertEqual(allocate_shape(8, 2, 40), 40)


if __name__ == '__main__':
    unittest.main(verbosity=2)
