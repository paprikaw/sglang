import pytest
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


@pytest.mark.parametrize("end_layer", [0, 1, 3])
def test_hybrid_head_dim_does_not_require_resident_attention(
    hybrid_pool_factory, end_layer
):
    pool = hybrid_pool_factory(start_layer=0, end_layer=end_layer)
    assert pool.get_v_head_dim() == 2


def test_hybrid_retained_recurrent_state_is_copied_at_commit(hybrid_pool_factory):
    pool = hybrid_pool_factory(start_layer=0, end_layer=3)
    old_state = pool.mamba_pool.mamba_cache
    old_state.temporal[1].fill_(7)
    assert pool.prepare_elastic_pp_kv_layers(
        target=LayerRange(2, 4), generation=1, total_num_layers=4
    ) == {3}
    # Serving continues after prepare; a background snapshot would be stale.
    old_state.temporal[1].fill_(11)
    old_state.conv[0][1].fill_(13)
    assert pool.req_to_token_pool.mamba_map == {0: 0, 2: 1}
    pool.import_elastic_pp_kv_slots(
        generation=1,
        layer_id=3,
        slot_ids=torch.tensor([7]),
        cache_k=torch.full((1, 1, 2), 17.0),
        cache_v=torch.full((1, 1, 2), 19.0),
    )
    pool.mark_elastic_pp_kv_layers_loaded(generation=1, layer_ids={3})
    pool.mamba_pool._conv_slot_desc = object()
    assert pool.commit_elastic_pp_kv_layers(generation=1) == LayerRange(2, 4)
    assert pool.full_attention_layer_id_mapping == {3: 1}
    assert pool.full_kv_pool.start_layer == 1
    assert pool.req_to_token_pool.mamba_map == {2: 0}
    assert torch.equal(pool.mamba_pool.mamba_cache.temporal[0], old_state.temporal[1])
    assert torch.equal(pool.mamba_pool.mamba_cache.conv[0][0], old_state.conv[0][1])
    assert torch.all(pool.get_key_buffer(3)[7] == 17)
    assert "_conv_slot_desc" not in pool.mamba_pool.__dict__
    released = pool.cleanup_elastic_pp_kv_layers(generation=1)
    assert any(t is old_state.temporal for t in released)


def test_hybrid_abort_preserves_live_state_and_readiness_needs_both_kinds(
    hybrid_pool_factory,
):
    pool = hybrid_pool_factory(start_layer=2, end_layer=4)
    old_state = pool.mamba_pool.mamba_cache
    old_k_ptr = pool.get_key_buffer(3).data_ptr()
    assert pool.prepare_elastic_pp_kv_layers(
        target=LayerRange(0, 4), generation=1, total_num_layers=4
    ) == {0, 1}
    pool.mark_elastic_pp_kv_layers_loaded(generation=1, layer_ids={1})
    assert not pool.elastic_pp_kv_ready(generation=1)
    with pytest.raises(RuntimeError, match="not ready"):
        pool.commit_elastic_pp_kv_layers(generation=1)
    pool.abort_elastic_pp_kv_layers(generation=1)
    assert pool.mamba_pool.mamba_cache is old_state
    assert pool.get_key_buffer(3).data_ptr() == old_k_ptr
    assert pool.req_to_token_pool.mamba_map == {2: 0}
    assert pool.prepare_elastic_pp_kv_layers(
        target=LayerRange(0, 4), generation=2, total_num_layers=4
    ) == {0, 1}


def test_hybrid_recurrent_import_rejects_wrong_tensor_layout(hybrid_pool_factory):
    pool = hybrid_pool_factory(start_layer=4, end_layer=4)
    pool.prepare_elastic_pp_kv_layers(
        target=LayerRange(2, 4), generation=1, total_num_layers=4
    )
    with pytest.raises(ValueError, match="shape/dtype"):
        pool.import_recurrent_slots(
            generation=1,
            layer_id=2,
            slot_ids=torch.tensor([3]),
            tensors={"conv_0": torch.ones(1, 2, 3), "temporal": torch.ones(1, 2, 3, 3)},
        )
    assert pool.req_to_token_pool.mamba_map == {}
    assert pool.mamba_pool.mamba_cache.temporal.shape[0] == 0
