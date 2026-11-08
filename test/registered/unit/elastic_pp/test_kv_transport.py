import pytest
import torch

from sglang.srt.elastic_pp.hybrid_kv_transport import ElasticPPHybridKVTransport
from sglang.srt.elastic_pp.kv_transport import (
    ElasticPPKVTransport,
    flatten_kv_chunk,
    unflatten_kv_chunk,
)
from sglang.srt.elastic_pp.topology import (
    DynamicPPTransition,
    LayerRange,
    LayerTransfer,
    PPTopology,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Mailbox:
    def __init__(self):
        self.payloads = {}


class _Group:
    def __init__(self, mailbox: _Mailbox, rank: int):
        self.mailbox = mailbox
        self.rank = rank

    def send_tensor_dict(self, payload, dst, async_send):
        assert async_send
        self.mailbox.payloads.setdefault((self.rank, dst), []).append(payload)
        return []

    def recv_tensor_dict(self, src):
        return self.mailbox.payloads[(src, self.rank)].pop(0)


class _Allocator:
    def __init__(self, slots):
        self.slots = slots
        self.finished = False
        self.tracking = False

    def begin_elastic_pp_slot_tracking(self, generation):
        del generation
        self.tracking = True
        return self.slots

    def elastic_pp_live_slots(self):
        return self.slots

    def elastic_pp_delta_slots(self, generation):
        del generation
        return self.slots[:0]

    def finish_elastic_pp_slot_tracking(self, generation):
        del generation
        self.finished = True
        self.tracking = False


class _Runner:
    def __init__(self, pool, *, fail_after_prepare=False):
        self.token_to_kv_pool = pool
        self.fail_after_prepare = fail_after_prepare

    def prepare_elastic_pp_kv_layers(self, target, generation):
        prepared = self.token_to_kv_pool.prepare_elastic_pp_kv_layers(
            target=target, generation=generation, total_num_layers=4
        )
        if self.fail_after_prepare:
            raise RuntimeError("injected failure after KV preparation")
        return prepared

    def abort_elastic_pp_kv_layers(self, generation):
        return self.token_to_kv_pool.abort_elastic_pp_kv_layers(generation=generation)


def _make_pool(*, start_layer, layer_num):
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
    pool._elastic_pp_enabled = True
    return pool


def test_kv_chunk_codec_validates_generation():
    transfer = LayerTransfer(0, 1, LayerRange(1, 2))
    slots = torch.tensor([1, 3])
    payload = flatten_kv_chunk(
        generation=4,
        transfer=transfer,
        phase="bulk",
        chunk_index=0,
        total_chunks=1,
        slot_ids=slots,
        layer_kv={1: (torch.ones(2, 1, 2), torch.zeros(2, 1, 2))},
    )
    decoded_slots, decoded = unflatten_kv_chunk(
        payload,
        generation=4,
        transfer=transfer,
        phase="bulk",
        chunk_index=0,
        total_chunks=1,
    )
    assert torch.equal(decoded_slots, slots)
    assert torch.equal(decoded[1][0], torch.ones(2, 1, 2))
    with pytest.raises(ValueError, match="generation"):
        unflatten_kv_chunk(
            payload,
            generation=5,
            transfer=transfer,
            phase="bulk",
            chunk_index=0,
            total_chunks=1,
        )


def test_sparse_transport_populates_unpublished_zero_rank_backing():
    transition = DynamicPPTransition.build(
        PPTopology.from_layer_counts([2, 0, 0, 2], generation=0),
        PPTopology.from_layer_counts([1, 1, 1, 1], generation=1),
    )
    slots = torch.tensor([1, 3], dtype=torch.long)
    source_pool = _make_pool(start_layer=0, layer_num=2)
    target_pool = _make_pool(start_layer=2, layer_num=0)
    fence_generations = []
    target_pool.prepare_elastic_pp_kv_fence = (
        lambda *, generation: fence_generations.append(generation)
    )
    source_pool.k_buffer[1][slots] = torch.tensor([[[11.0, 12.0]], [[31.0, 32.0]]])
    source_pool.v_buffer[1][slots] = torch.tensor([[[13.0, 14.0]], [[33.0, 34.0]]])

    mailbox = _Mailbox()
    source = ElasticPPKVTransport(
        group=_Group(mailbox, 0),
        model_runner=_Runner(source_pool),
        allocator=_Allocator(slots),
        rank=0,
    )
    target = ElasticPPKVTransport(
        group=_Group(mailbox, 1),
        model_runner=_Runner(target_pool),
        allocator=_Allocator(slots),
        rank=1,
    )

    source.prepare_transition(transition)
    target.prepare_transition(transition)
    source_slots = source.fence_slots(transition)
    target_slots = target.fence_slots(transition)
    assert source.send_phase(transition, phase="bulk", slot_ids=source_slots) == 1
    assert target_slots.tolist() == [1, 3]
    assert target.receive_phase(transition, phase="bulk") == frozenset({1})
    assert not target_pool.elastic_pp_kv_ready(generation=1)
    assert source.send_phase(transition, phase="delta", slot_ids=source_slots[:0]) == 1
    assert target.receive_phase(transition, phase="delta") == frozenset({1})
    assert target.apply_staged(transition) == frozenset({1})
    assert fence_generations == [1]
    assert target_pool.elastic_pp_kv_ready(generation=1)
    target_pool.commit_elastic_pp_kv_layers(generation=1)
    assert torch.equal(
        target_pool.get_key_buffer(1)[slots], source_pool.k_buffer[1][slots]
    )
    assert torch.equal(
        target_pool.get_value_buffer(1)[slots], source_pool.v_buffer[1][slots]
    )


def test_global_exchange_uses_source_slots_with_rank_skew():
    transition = DynamicPPTransition.build(
        PPTopology.from_layer_counts([2, 2], generation=0),
        PPTopology.from_layer_counts([1, 3], generation=1),
    )
    slots = torch.tensor([1, 3, 5], dtype=torch.long)
    source_pool = _make_pool(start_layer=0, layer_num=2)
    target_pool = _make_pool(start_layer=2, layer_num=2)
    source_pool.k_buffer[1][slots] = torch.tensor(
        [[[11.0, 12.0]], [[31.0, 32.0]], [[51.0, 52.0]]]
    )
    source_pool.v_buffer[1][slots] = torch.tensor(
        [[[13.0, 14.0]], [[33.0, 34.0]], [[53.0, 54.0]]]
    )

    mailbox = _Mailbox()
    source = ElasticPPKVTransport(
        group=_Group(mailbox, 0),
        model_runner=_Runner(source_pool),
        allocator=_Allocator(slots),
        rank=0,
        slot_chunk_size=2,
    )
    target_allocator = _Allocator(torch.tensor([2, 4], dtype=torch.long))
    target = ElasticPPKVTransport(
        group=_Group(mailbox, 1),
        model_runner=_Runner(target_pool),
        allocator=target_allocator,
        rank=1,
        slot_chunk_size=2,
    )

    source.prepare_transition(transition)
    target.prepare_transition(transition)
    source_slots = source.fence_slots(transition)
    target_slots = target.fence_slots(transition)
    assert target_slots.tolist() == [2, 4]
    assert (
        source.exchange_phase(transition, phase="bulk", slot_ids=source_slots)
        == frozenset()
    )
    assert target.exchange_phase(
        transition, phase="bulk", slot_ids=target_slots
    ) == frozenset({1})
    source_pool.k_buffer[1][3] = torch.tensor([[91.0, 92.0]])
    source_pool.v_buffer[1][3] = torch.tensor([[93.0, 94.0]])
    delta = torch.tensor([3], dtype=torch.long)
    source.exchange_phase(transition, phase="delta", slot_ids=delta)
    target.exchange_phase(transition, phase="delta", slot_ids=slots[:0])
    assert target.apply_staged(transition) == frozenset({1})
    target_pool.commit_elastic_pp_kv_layers(generation=1)
    assert torch.equal(
        target_pool.get_key_buffer(1)[slots], source_pool.k_buffer[1][slots]
    )
    assert torch.equal(
        target_pool.get_value_buffer(1)[slots], source_pool.v_buffer[1][slots]
    )


def test_empty_source_snapshot_still_delivers_one_chunk():
    transition = DynamicPPTransition.build(
        PPTopology.from_layer_counts([2, 2], generation=0),
        PPTopology.from_layer_counts([1, 3], generation=1),
    )
    source_pool = _make_pool(start_layer=0, layer_num=2)
    target_pool = _make_pool(start_layer=2, layer_num=2)
    mailbox = _Mailbox()
    source = ElasticPPKVTransport(
        group=_Group(mailbox, 0),
        model_runner=_Runner(source_pool),
        allocator=_Allocator(torch.empty(0, dtype=torch.long)),
        rank=0,
    )
    target = ElasticPPKVTransport(
        group=_Group(mailbox, 1),
        model_runner=_Runner(target_pool),
        allocator=_Allocator(torch.tensor([2, 4], dtype=torch.long)),
        rank=1,
    )

    source.prepare_transition(transition)
    target.prepare_transition(transition)
    source_slots = source.fence_slots(transition)
    assert (
        source.exchange_phase(transition, phase="bulk", slot_ids=source_slots)
        == frozenset()
    )
    assert target.exchange_phase(
        transition, phase="bulk", slot_ids=torch.tensor([2, 4])
    ) == frozenset({1})
    source.exchange_phase(
        transition, phase="delta", slot_ids=torch.empty(0, dtype=torch.long)
    )
    target.exchange_phase(
        transition, phase="delta", slot_ids=torch.empty(0, dtype=torch.long)
    )
    assert target.apply_staged(transition) == frozenset({1})
    assert mailbox.payloads[(0, 1)] == []
    assert target_pool.elastic_pp_kv_ready(generation=1)


def test_prepare_transition_rolls_back_before_coordinator_flags_exist():
    transition = DynamicPPTransition.build(
        PPTopology.from_layer_counts([2, 2], generation=0),
        PPTopology.from_layer_counts([1, 3], generation=1),
    )
    pool = _make_pool(start_layer=2, layer_num=2)
    allocator = _Allocator(torch.tensor([1, 3], dtype=torch.long))
    transport = ElasticPPKVTransport(
        group=_Group(_Mailbox(), 1),
        model_runner=_Runner(pool, fail_after_prepare=True),
        allocator=allocator,
        rank=1,
    )

    with pytest.raises(RuntimeError, match="injected failure"):
        transport.prepare_transition(transition)

    assert not allocator.finished
    assert not allocator.tracking
    assert pool._elastic_pp_pending_kv == {}
    assert pool._elastic_pp_residency.pending is None


@pytest.mark.parametrize(
    "source_counts,target_counts", [([4, 0], [1, 3]), ([3, 1], [2, 2])]
)
def test_hybrid_transport_uses_request_slots_not_token_slots(
    hybrid_pool_factory, source_counts, target_counts
):
    transition = DynamicPPTransition.build(
        PPTopology.from_layer_counts(source_counts, generation=0),
        PPTopology.from_layer_counts(target_counts, generation=1),
    )
    pools = [
        hybrid_pool_factory(
            start_layer=transition.source.layer_range(rank).start,
            end_layer=transition.source.layer_range(rank).end,
        )
        for rank in range(2)
    ]
    source_state = pools[0].mamba_pool.mamba_cache
    source_state.temporal[1].copy_(torch.arange(5 * 18).reshape(5, 2, 3, 3))
    source_state.conv[0][1].fill_(23)
    token_slots = torch.tensor([6, 7])  # Outside the five recurrent slots.
    for layer_id in pools[0].full_attention_layer_id_mapping:
        pools[0].get_key_buffer(layer_id)[token_slots] = 10 + layer_id
        pools[0].get_value_buffer(layer_id)[token_slots] = 20 + layer_id
    mailbox = _Mailbox()
    transports = [
        ElasticPPHybridKVTransport(
            group=_Group(mailbox, rank),
            model_runner=_Runner(pools[rank]),
            allocator=_Allocator(token_slots),
            rank=rank,
            slot_chunk_size=1,
            recurrent_chunk_size=2,
        )
        for rank in range(2)
    ]
    assert (
        transports[0].fence_state_signature() == transports[1].fence_state_signature()
    )
    for transport in transports:
        assert transport.prepare_transition(transition).numel() == 0
        assert not transport.allocator.tracking
    for phase, slots in (("bulk", token_slots), ("delta", token_slots[:0])):
        for transport in transports:
            transport.exchange_phase(transition, phase=phase, slot_ids=slots)
    assert pools[1].req_to_token_pool.mamba_map == {}
    assert not pools[1].elastic_pp_kv_ready(generation=1)
    for transport in transports:
        transport.apply_staged(transition)
    pools[1].commit_elastic_pp_kv_layers(generation=1)
    assert pools[1].req_to_token_pool.mamba_map == {2: 0}
    assert torch.equal(
        pools[1].mamba_pool.mamba_cache.temporal[0], source_state.temporal[1]
    )
    assert torch.equal(
        pools[1].mamba_pool.mamba_cache.conv[0][0], source_state.conv[0][1]
    )
    for layer_id in (1, 3):
        if (
            layer_id in pools[0].full_attention_layer_id_mapping
            and layer_id in pools[1].full_attention_layer_id_mapping
        ):
            assert torch.equal(
                pools[1].get_key_buffer(layer_id)[token_slots],
                pools[0].get_key_buffer(layer_id)[token_slots],
            )
    assert not mailbox.payloads[(0, 1)]


def test_hybrid_codec_requires_explicit_sparse_attention_subset():
    transfer = LayerTransfer(0, 1, LayerRange(0, 4))
    slots = torch.tensor([7])
    args = dict(
        generation=1, transfer=transfer, phase="bulk", chunk_index=0, total_chunks=1
    )
    layer_kv = {
        1: (torch.ones(1, 1, 2), torch.zeros(1, 1, 2)),
        3: (torch.ones(1, 1, 2), torch.zeros(1, 1, 2)),
    }
    with pytest.raises(ValueError, match="exactly cover"):
        flatten_kv_chunk(**args, slot_ids=slots, layer_kv=layer_kv)
    payload = flatten_kv_chunk(
        **args, slot_ids=slots, layer_kv=layer_kv, layer_ids=(1, 3)
    )
    with pytest.raises(ValueError, match="one K/V pair"):
        unflatten_kv_chunk(payload, **args)
    decoded_slots, decoded = unflatten_kv_chunk(payload, **args, layer_ids=(1, 3))
    assert torch.equal(slots, decoded_slots)
    assert set(decoded) == {1, 3}


def test_active_kv_cpu_snapshot_is_independent_and_released_on_abort():
    transition = DynamicPPTransition.build(
        PPTopology.from_layer_counts([2, 2], generation=0),
        PPTopology.from_layer_counts([1, 3], generation=1),
    )
    pool = _make_pool(start_layer=0, layer_num=2)
    pool.k_buffer[1].fill_(7)
    transport = ElasticPPKVTransport(
        group=_Group(_Mailbox(), 0),
        model_runner=_Runner(pool),
        allocator=_Allocator(torch.tensor([1, 3])),
        rank=0,
    )
    assert transport.snapshot_at_fence(transition) > 0
    snapshot = transport._fence_snapshot
    assert set(snapshot["layers"]) == {1}
    assert snapshot["slots"].tolist() == [1, 3]
    pool.k_buffer[1].zero_()
    assert snapshot["layers"][1][0].eq(7).all()
    transport.abort_staged(transition)
    assert transport._fence_snapshot is None


def test_active_gdn_cpu_snapshot_uses_allocated_state_slots(hybrid_pool_factory):
    transition = DynamicPPTransition.build(
        PPTopology.from_layer_counts([2, 2], generation=0),
        PPTopology.from_layer_counts([0, 4], generation=1),
    )
    pool = hybrid_pool_factory(start_layer=0, end_layer=2)
    allocated = pool.req_to_token_pool.mamba_allocator.alloc(1)
    slot = int(allocated[0])
    pool.mamba_pool.mamba_cache.conv[0][0, slot].fill_(9)
    transport = ElasticPPHybridKVTransport(
        group=_Group(_Mailbox(), 0),
        model_runner=_Runner(pool),
        allocator=_Allocator(torch.tensor([1, 3])),
        rank=0,
    )
    assert transport.snapshot_at_fence(transition) > 0
    snapshot = transport._fence_snapshot
    assert snapshot["recurrent_slots"].tolist() == [0, slot]
    pool.mamba_pool.mamba_cache.conv[0].zero_()
    assert snapshot["recurrent"][0]["conv_0"][1].eq(9).all()
    transport.release_fence_snapshot(transition)
    assert transport._fence_snapshot is None
