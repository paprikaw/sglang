"""Hybrid MHA PP stages must transfer global K/V layers, not dense offsets."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def manager():
    obj = object.__new__(MooncakeKVManager)
    obj.is_mla_backend = False
    obj.is_hybrid_mla_backend = False
    obj.pp_size = 2
    obj.enable_custom_mem_pool = False
    obj.kv_args = SimpleNamespace(prefill_start_layer=20)
    obj._transfer_data = Mock(return_value=0)
    return obj


def transfer(obj, source_ids, destination_ids):
    return obj._send_kvcache_generic(
        mooncake_session_id="decode",
        src_data_ptrs=[1000, 2000, 3000, 4000],
        dst_data_ptrs=[10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000],
        item_lens=[4, 8, 12, 16],
        prefill_data_indices=np.array([2, 3], dtype=np.int32),
        dst_data_indices=np.array([5, 6], dtype=np.int32),
        executor=None,
        src_layer_ids=source_ids,
        dst_layer_ids=destination_ids,
    )


def test_second_pp_stage_maps_sparse_full_attention_layers_and_kv_occurrences():
    obj = manager()
    assert transfer(obj, [23, 27, 23, 27], [3, 7, 23, 27, 3, 7, 23, 27]) == 0
    obj._transfer_data.assert_called_once_with("decode", [
        (1008, 30020, 8),
        (2016, 40040, 16),
        (3024, 70060, 24),
        (4032, 80080, 32),
    ])


def test_first_pp_stage_does_not_confuse_k_and_v_buffer_positions():
    obj = manager()
    obj.kv_args.prefill_start_layer = 0
    assert transfer(obj, [3, 7, 3, 7], [3, 7, 23, 27, 3, 7, 23, 27]) == 0
    blocks = obj._transfer_data.call_args.args[1]
    assert [dst for _, dst, _ in blocks] == [10020, 20040, 50060, 60080]


@pytest.mark.parametrize("source_ids,destination_ids,error", [
    ([23, 27, 23, 27], [], "both PD peers"),
    ([23, 31, 23, 31], [3, 7, 23, 27, 3, 7, 23, 27], "missing a transfer entry"),
    ([23, 27], [3, 7, 23, 27, 3, 7, 23, 27], "length must match"),
])
def test_invalid_or_incomplete_metadata_never_submits_a_partial_write(
    source_ids, destination_ids, error
):
    obj = manager()
    with pytest.raises(RuntimeError, match=error):
        transfer(obj, source_ids, destination_ids)
    obj._transfer_data.assert_not_called()
