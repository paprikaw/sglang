from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.elastic_hybrid_pool import ElasticHybridLinearKVPool
from sglang.srt.mem_cache.memory_pool import MambaPool


@pytest.fixture
def hybrid_pool_factory():
    def make(*, start_layer, end_layer, full_pool_class=None):
        ids = [i for i in (0, 2) if start_layer <= i < end_layer]
        # CPU tensor state avoids GPU-only Mamba constructor warmup; all
        # transactional pool and allocator methods below are the real ones.
        mamba = MambaPool.__new__(MambaPool)
        mamba.size = 4
        mamba.device = "cpu"
        mamba.num_mamba_layers = len(ids)
        mamba.mamba_layer_ids = ids
        mamba.mamba_cache = MambaPool.State(
            conv=[torch.zeros(len(ids), 5, 2, 3, dtype=torch.bfloat16)],
            temporal=torch.zeros(len(ids), 5, 2, 3, 3),
        )
        req_pool = SimpleNamespace(
            size=4,
            device="cpu",
            mamba_pool=mamba,
            mamba_map={layer_id: i for i, layer_id in enumerate(ids)},
            mamba_ckpt_pool=None,
            mamba_allocator=MambaSlotAllocator(size=4, device="cpu"),
            free_slots=[1, 2, 3, 4],
            req_generation=torch.zeros(5, dtype=torch.int64),
            req_index_to_mamba_index_mapping=torch.zeros(5, dtype=torch.int32),
        )
        pool = ElasticHybridLinearKVPool(
            req_to_token_pool=req_pool,
            all_full_attention_layer_ids=(1, 3),
            total_num_layers=4,
            full_attention_layer_ids=[
                i for i in (1, 3) if start_layer <= i < end_layer
            ],
            start_layer=start_layer,
            end_layer=end_layer,
            size=8,
            page_size=1,
            dtype=torch.float32,
            head_num=1,
            head_dim=2,
            device="cpu",
            mamba_pool=mamba,
            enable_memory_saver=False,
            full_kv_pool_class=full_pool_class,
        )
        pool.full_kv_pool._elastic_pp_enabled = True
        return pool

    return make
