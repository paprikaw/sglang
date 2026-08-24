import torch

from sglang.srt.elastic_pp.topology import LayerRange
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_pool(*, start_layer: int, layer_num: int) -> MHATokenToKVPool:
    pool = MHATokenToKVPool(
        size=8,
        page_size=1,
        dtype=torch.float32,
        head_num=1,
        head_dim=2,
        layer_num=layer_num,
        device="cpu",
        enable_memory_saver=False,
        start_layer=start_layer,
        end_layer=start_layer + layer_num,
        enable_alt_stream=False,
    )
    # The production constructor reads this from RuntimeContext.  Keep this
    # focused CPU test independent of server/bootstrap initialization.
    pool._elastic_pp_enabled = True
    return pool


def test_pool_keeps_old_mapping_live_until_atomic_commit():
    pool = _make_pool(start_layer=0, layer_num=2)
    pool.k_buffer[1][3].fill_(7)
    overlap_k_ptr = pool.k_buffer[1].data_ptr()
    old_layer_zero = (pool.k_buffer[0], pool.v_buffer[0])

    assert pool.prepare_elastic_pp_kv_layers(
        target=LayerRange(1, 3), generation=1, total_num_layers=4
    ) == frozenset({2})
    # Preparation must not disturb the source generation's buffer indexing.
    assert pool.start_layer == 0
    assert pool.end_layer == 2
    assert torch.equal(pool.get_key_buffer(1)[3], torch.tensor([[7.0, 7.0]]))

    slots = torch.tensor([3], dtype=torch.long)
    pool.import_elastic_pp_kv_slots(
        generation=1,
        layer_id=2,
        slot_ids=slots,
        cache_k=torch.tensor([[[5.0, 5.0]]]),
        cache_v=torch.tensor([[[6.0, 6.0]]]),
    )
    pool.mark_elastic_pp_kv_layers_loaded(generation=1, layer_ids={2})
    assert pool.commit_elastic_pp_kv_layers(generation=1) == LayerRange(1, 3)

    assert pool.start_layer == 1
    assert pool.end_layer == 3
    assert pool.layer_num == 2
    assert pool.get_key_buffer(1).data_ptr() == overlap_k_ptr
    assert torch.equal(pool.get_key_buffer(2)[3], torch.tensor([[5.0, 5.0]]))
    released = pool.cleanup_elastic_pp_kv_layers(generation=1)
    assert {item.data_ptr() for item in released} == {
        old_layer_zero[0].data_ptr(),
        old_layer_zero[1].data_ptr(),
    }


def test_zero_layer_middle_rank_can_prepare_and_publish_backing():
    pool = _make_pool(start_layer=2, layer_num=0)
    assert pool.k_buffer == []
    assert pool.prepare_elastic_pp_kv_layers(
        target=LayerRange(2, 3), generation=1, total_num_layers=4
    ) == frozenset({2})
    pool.import_elastic_pp_kv_slots(
        generation=1,
        layer_id=2,
        slot_ids=torch.empty(0, dtype=torch.long),
        cache_k=torch.empty((0, 1, 2)),
        cache_v=torch.empty((0, 1, 2)),
    )
    pool.mark_elastic_pp_kv_layers_loaded(generation=1, layer_ids={2})
    assert pool.commit_elastic_pp_kv_layers(generation=1) == LayerRange(2, 3)
    assert len(pool.k_buffer) == len(pool.v_buffer) == 1
    assert pool.data_ptrs.numel() == 2
    assert pool.cleanup_elastic_pp_kv_layers(generation=1) == ()
