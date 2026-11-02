"""A PP1 prefill must deliver every KV/GDN entry to its owning decode stage."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager, PrefillServerInfo
from sglang.srt.disaggregation.mooncake.conn import KVArgsRegisterInfo, MooncakeKVManager
from sglang.srt.disaggregation.utils import build_transfer_entry_pairs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def manager(pp_size=2, pp_rank=0, cls=MooncakeKVManager):
    obj = object.__new__(cls)
    obj.pp_size = pp_size
    obj.pp_rank = pp_rank
    obj.attn_tp_size = obj.attn_cp_size = obj.dcp_size = 1
    obj.attn_cp_rank = 0
    obj.is_mla_backend = obj.is_hybrid_mla_backend = False
    obj.enable_custom_mem_pool = False
    obj.kv_args = SimpleNamespace(
        engine_rank=pp_rank,
        kv_layer_ids=[3, 7, 3, 7],
        kv_data_ptrs=[1000, 2000, 3000, 4000],
        state_types=[StateType.MAMBA],
        state_layer_ids=[[0, 1, 0, 1]],
        state_data_ptrs=[[5000, 6000, 7000, 8000]],
    )
    obj._transfer_data = Mock(return_value=0)
    return obj


def prefill(pp_size):
    return PrefillServerInfo(
        attn_tp_size=1, attn_cp_size=1, dp_size=2, pp_size=pp_size,
        page_size=None, kv_cache_dtype=None, follow_bootstrap_room=False,
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_pp1_sender_waits_for_both_decode_stage_destinations(rank):
    obj, info = manager(pp_rank=rank), prefill(1)
    obj._resolve_rank_mapping(info)
    assert info.target_pp_ranks == [0]
    assert info.required_dst_info_num == 2
    assert info.required_prefill_response_num == 1


@pytest.mark.parametrize("pp_size,pp_rank,prefill_pp,targets,responses", [
    (2, 1, 2, [1], 1),
    (1, 0, 2, [0, 1], 2),
    (1, 0, 1, [0], 1),
])
def test_existing_pp_routes_keep_their_rank_and_response_counts(
    pp_size, pp_rank, prefill_pp, targets, responses
):
    obj, info = manager(pp_size, pp_rank), prefill(prefill_pp)
    obj._resolve_rank_mapping(info)
    assert info.target_pp_ranks == targets
    assert info.required_dst_info_num == 1
    assert info.required_prefill_response_num == responses


def test_other_backends_do_not_gain_unimplemented_pp_expansion():
    obj = manager(cls=CommonKVManager)
    with pytest.raises(AssertionError, match="qualified layer-aware backend"):
        obj._resolve_rank_mapping(prefill(1))


@pytest.mark.parametrize("missing", ["kv_layer_ids", "state_layer_ids"])
def test_pp_expansion_requires_explicit_layer_registrations(missing):
    obj = manager()
    setattr(obj.kv_args, missing, [])
    with pytest.raises(AssertionError, match="qualified layer-aware backend"):
        obj._resolve_rank_mapping(prefill(1))


def test_pp_expansion_does_not_enable_tp_relayout():
    obj = manager()
    obj.attn_tp_size = 2
    assert not obj._supports_pp1_to_pp_decode(prefill(1))


def test_unqualified_pp_degree_stays_rejected():
    assert not manager(pp_size=4)._supports_pp1_to_pp_decode(prefill(1))


def test_send_kvcache_propagates_explicit_destination_pp_size():
    obj = manager(pp_size=1)
    obj.kv_args.kv_item_lens = [4] * 4
    obj._validate_envelope_kv_layout = Mock()
    obj._send_kvcache_generic = Mock(return_value=0)
    obj.send_kvcache("decode", np.array([2]), [10000, 20000], np.array([5]),
                     executor=None, dst_layer_ids=[3, 3], dst_pp_size=2)
    assert obj._send_kvcache_generic.call_args.kwargs["allow_destination_subset"]


@pytest.mark.parametrize("stage_ids,source_indices", [
    ([3, 7, 3, 7], [0, 1, 4, 5]),
    ([23, 27, 23, 27], [2, 3, 6, 7]),
])
def test_kv_rdma_writes_only_the_target_stage_and_preserves_k_v_occurrences(
    stage_ids, source_indices
):
    obj = manager(pp_size=1)
    assert obj._send_kvcache_generic(
        mooncake_session_id="decode",
        src_data_ptrs=[1000 * (i + 1) for i in range(8)],
        dst_data_ptrs=[10000 * (i + 1) for i in range(4)],
        item_lens=[4] * 8,
        prefill_data_indices=np.array([2, 3], dtype=np.int32),
        dst_data_indices=np.array([5, 6], dtype=np.int32),
        executor=None,
        src_layer_ids=[3, 7, 23, 27, 3, 7, 23, 27],
        dst_layer_ids=stage_ids,
        allow_destination_subset=True,
    ) == 0
    obj._transfer_data.assert_called_once_with("decode", [
        (1000 * (i + 1) + 8, 10000 * (j + 1) + 20, 8)
        for j, i in enumerate(source_indices)
    ])


@pytest.mark.parametrize("stage_ids,source_indices", [
    ([0, 1, 0, 1], [0, 1, 4, 5]),
    ([20, 21, 20, 21], [2, 3, 6, 7]),
])
def test_gdn_conv_and_recurrent_state_use_their_own_slot_offsets(
    stage_ids, source_indices
):
    obj = manager(pp_size=1)
    lens = [16] * 4 + [32] * 4
    assert obj._send_mamba_state(
        SimpleNamespace(mooncake_session_id="decode"), [3],
        [1000 * (i + 1) for i in range(8)], lens,
        [10000 * (i + 1) for i in range(4)], [7],
        [0, 1, 20, 21, 0, 1, 20, 21], stage_ids, dst_pp_size=2,
    ) == 0
    obj._transfer_data.assert_called_once_with("decode", [
        (1000 * (i + 1) + 3 * lens[i],
         10000 * (j + 1) + 7 * lens[i], lens[i])
        for j, i in enumerate(source_indices)
    ])


@pytest.mark.parametrize("source_ids,destination_ids", [
    ([3, 7, 3, 7], []),
    ([], [3, 3]),
    ([3, 7, 3, 7], [3]),  # cannot drop V while keeping K
    ([3, 7, 3, 7], [23, 23]),
    ([3, 7, 3, 7], [3, 3, 3]),
])
def test_subset_transfer_rejects_incomplete_or_incompatible_metadata(
    source_ids, destination_ids
):
    with pytest.raises(RuntimeError):
        build_transfer_entry_pairs(
            source_ids, destination_ids, len(source_ids), len(destination_ids),
            allow_destination_subset=True,
        )


def test_legacy_transfer_does_not_silently_drop_source_layers():
    with pytest.raises(RuntimeError, match="missing a transfer entry"):
        build_transfer_entry_pairs([3, 7, 3, 7], [3, 3], 4, 2)


def test_registration_pp_size_is_explicit_and_old_wire_defaults_to_one():
    message = [
        b"None", b"127.0.0.1", b"1234", b"session", b"", b"", b"",
        b"0", b"1", b"128", b"", b"", b"", b"", b"", b"", b"1", b"0",
    ]
    assert KVArgsRegisterInfo.from_zmq(message).dst_pp_size == 1
    assert KVArgsRegisterInfo.from_zmq(message + [b"2"]).dst_pp_size == 2
