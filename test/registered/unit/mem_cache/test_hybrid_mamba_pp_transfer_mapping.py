import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler


class TestMambaPipelineTransferMapping(unittest.TestCase):
    def _check_stage(self, start, end):
        full_ids = [layer for layer in range(start, end) if layer % 4 == 3]
        mamba_ids = [layer for layer in range(start, end) if layer % 4 != 3]
        full = dict(zip(full_ids, range(len(full_ids))))
        mamba = dict(zip(mamba_ids, range(len(mamba_ids))))
        cache = MagicMock()
        kv = SimpleNamespace(start_layer=start,
                             full_attention_layer_id_mapping=full.copy(),
                             full_kv_pool=object(), use_mla=False)
        params = SimpleNamespace(req_to_token_pool=SimpleNamespace(
            mamba_map=mamba.copy(), mamba_pool=object()))
        with patch.object(assembler, 'build_hybrid_mamba_stack',
                          return_value=(MagicMock(), MagicMock())) as build:
            result = assembler._MambaStrategy().build(
                cache=cache, kvcache=kv, params=params,
                server_args=object(), load_cache_event=object())
        mappings = build.call_args.kwargs
        full_mapper = assembler._make_layer_mapper(
            mappings['full_layer_mapping'], result.transfer_layer_num)
        mamba_mapper = assembler._make_layer_mapper(
            mappings['mamba_layer_mapping'], result.transfer_layer_num)
        # Exercise the same local iteration as L2TransferEngine. Every model
        # layer must restore exactly one physical buffer, including PP stage 1.
        restored = []
        for local_layer in range(result.transfer_layer_num):
            for name, mapper in [('kv', full_mapper), ('mamba', mamba_mapper)]:
                index = mapper(local_layer)
                if index is not None:
                    restored.append((start + local_layer, name, index))
        expected = [(layer, 'kv', full[layer]) if layer in full
                    else (layer, 'mamba', mamba[layer]) for layer in range(start, end)]
        self.assertEqual(restored, expected)
        self.assertEqual(result.transfer_layer_num, end - start)
        # Forward and RDMA paths still index the original global layer maps.
        self.assertEqual(kv.full_attention_layer_id_mapping, full)
        self.assertEqual(params.req_to_token_pool.mamba_map, mamba)

    def test_first_pipeline_stage(self):
        self._check_stage(0, 20)

    def test_second_pipeline_stage(self):
        self._check_stage(20, 40)

    def test_nonuniform_pipeline_boundary(self):
        self._check_stage(13, 31)


if __name__ == '__main__':
    unittest.main()
