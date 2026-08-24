"""Single-GPU smoke test for elastic-PP KVCacheD layer ownership.

Run this with the autoscaling KVCacheD checkout first on ``PYTHONPATH`` and
with layer stacking enabled. The test intentionally inspects KVCacheD's local
worker mapping registry so a virtual-view-only implementation cannot pass.
"""

from __future__ import annotations

import os

import torch

from sglang.srt.elastic_pp.topology import LayerRange
from sglang.srt.mem_cache.kvcached_pool import (
    KVCacheDMHATokenToKVPool,
    KVCacheDTokenToKVPoolAllocator,
)
from sglang.srt.runtime_context import get_parallel


def _mapped_offsets(group_id: int = 0) -> set[int]:
    from kvcached import tp_ipc_util

    with tp_ipc_util._WORKER_MAPPING_LOCK:
        return set(tp_ipc_util._WORKER_MAPPED_OFFSETS.get(group_id, set()))


def _require_layer_stacking_environment() -> None:
    expected = {
        "KVCACHED_LAYER_STACKING": "true",
        "KVCACHED_CONTIGUOUS_LAYOUT": "false",
        "KVCACHED_PAGE_PREALLOC_ENABLED": "false",
    }
    mismatches = {
        name: os.getenv(name)
        for name, value in expected.items()
        if os.getenv(name, "").lower() != value
    }
    if mismatches:
        raise RuntimeError(
            "KVCacheD physical smoke requires explicit layer-stacking envs: "
            f"mismatches={mismatches}"
        )


def test_kvcached_physical_prepare_commit_cleanup_and_abort() -> None:
    _require_layer_stacking_environment()
    assert torch.cuda.is_available()
    device = "cuda:0"

    with get_parallel().override(
        world_size=1,
        world_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=2,
        pp_rank=0,
    ):
        pool = KVCacheDMHATokenToKVPool(
            size=512,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=8,
            head_dim=128,
            layer_num=2,
            total_num_layers=4,
            device=device,
            enable_memory_saver=False,
            start_layer=0,
            end_layer=2,
            enable_alt_stream=False,
        )
        pool._elastic_pp_enabled = True
        allocator = KVCacheDTokenToKVPoolAllocator(
            size=512,
            dtype=torch.bfloat16,
            device=device,
            kvcache=pool,
            need_sort=False,
        )

        manager = pool.kvcached_allocator
        assert pool.physical_layer_partitioned
        assert manager.layer_group_granularity == 2
        assert manager.num_layer_groups == 2
        map_size = int(manager.physical_group_page_size)
        group_span = pool._layer_group_total_span()
        assert group_span == 3 * map_size

        live_slots = allocator.alloc(260)
        assert live_slots is not None
        assert torch.equal(live_slots.cpu(), torch.arange(1, 261))
        assert _mapped_offsets() == {0, map_size}

        prepared = pool.prepare_elastic_pp_kv_layers(
            target=LayerRange(2, 4),
            generation=1,
            total_num_layers=4,
        )
        assert prepared == frozenset({2, 3})
        assert _mapped_offsets() == {
            0,
            map_size,
            group_span,
            group_span + map_size,
        }

        expected_k = {}
        expected_v = {}
        for layer_id in sorted(prepared):
            expected_k[layer_id] = torch.full(
                (live_slots.numel(), 8, 128),
                float(layer_id + 1),
                dtype=torch.bfloat16,
                device=device,
            )
            expected_v[layer_id] = torch.full(
                (live_slots.numel(), 8, 128),
                float(layer_id + 11),
                dtype=torch.bfloat16,
                device=device,
            )
            pool.import_elastic_pp_kv_slots(
                generation=1,
                layer_id=layer_id,
                slot_ids=live_slots,
                cache_k=expected_k[layer_id],
                cache_v=expected_v[layer_id],
            )

        pool.mark_elastic_pp_kv_layers_loaded(generation=1, layer_ids=set(prepared))
        assert pool.commit_elastic_pp_kv_layers(generation=1) == LayerRange(2, 4)
        for layer_id in sorted(prepared):
            torch.testing.assert_close(
                pool.get_key_buffer(layer_id).index_select(0, live_slots),
                expected_k[layer_id],
            )
            torch.testing.assert_close(
                pool.get_value_buffer(layer_id).index_select(0, live_slots),
                expected_v[layer_id],
            )

        released = pool.cleanup_elastic_pp_kv_layers(generation=1)
        assert len(released) == 4
        assert _mapped_offsets() == {group_span, group_span + map_size}

        remaining_slots = allocator.alloc(252)
        assert remaining_slots is not None
        assert torch.equal(remaining_slots.cpu(), torch.arange(261, 513))
        assert _mapped_offsets() == {
            group_span,
            group_span + map_size,
            group_span + 2 * map_size,
        }

        assert pool.prepare_elastic_pp_kv_layers(
            target=LayerRange(0, 2),
            generation=2,
            total_num_layers=4,
        ) == frozenset({0, 1})
        assert _mapped_offsets() == {
            0,
            map_size,
            2 * map_size,
            group_span,
            group_span + map_size,
            group_span + 2 * map_size,
        }
        pool.abort_elastic_pp_kv_layers(generation=2)
        assert _mapped_offsets() == {
            group_span,
            group_span + map_size,
            group_span + 2 * map_size,
        }

        stats = pool.kvcached_physical_stats()
        assert stats["physical_layer_partitioned"] is True
        assert stats["active_start_layer"] == 2
        assert stats["active_end_layer"] == 4
        assert stats["mapped_memory_scope"] == "local_worker_offset_ledger"
        assert stats["local_worker_mapped_offset_count"] == 3
        assert stats["local_worker_mapped_layer_groups"] == [1]
        assert stats["mapped_memory_bytes"] == 3 * map_size
        assert (
            stats["allocator_estimated_mapped_memory_bytes"]
            > stats["mapped_memory_bytes"]
        )

        manager.shutdown()
        from kvcached.integration.vllm.interfaces import shutdown_kvcached

        shutdown_kvcached()


if __name__ == "__main__":
    test_kvcached_physical_prepare_commit_cleanup_and_abort()
    print("KVCacheD physical elastic-PP smoke passed")
