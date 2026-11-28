from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.elastic_pp.prefill_kv_compaction import (
    PrefillKVCompaction,
    relocation_rows,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType


def test_prefill_shrink_defers_when_the_working_set_does_not_fit():
    with pytest.raises(RuntimeError, match="defer Prefill shrink"):
        relocation_rows({1, 2, 4, 6}, capacity=3)


def test_prefill_cpu_roundtrip_keeps_request_and_tree_aliases(monkeypatch):
    from sglang.srt.elastic_pp import backing_spill

    checked_budget = []
    monkeypatch.setattr(backing_spill, "_require_cpu_space", checked_budget.append)
    allocator = NS(
        _live_page_ids={1, 6, 7}, serving_capacity=8, device="cpu",
        free_pages=torch.tensor([2, 3, 4, 5, 8]), release_pages=torch.empty(0),
    )
    prefix = torch.tensor([1, 6])
    request_rows = torch.tensor([[1, 6, 7, 0]], dtype=torch.int32)
    req = NS(prefix_indices=prefix.clone(), req_pool_idx=0, kv=NS(kv_allocated_len=3))
    tree = NS(_node_arena={
        0: NS(component_data={ComponentType.FULL: NS(value=[])}),
        1: NS(component_data={ComponentType.FULL: NS(value=prefix)}),
    })
    scheduler = NS(
        req_to_token_pool=NS(req_to_token=request_rows), tree_cache=NS(tree_core=tree),
        _elastic_pp_live_requests=lambda: [req],
    )
    k, v = torch.arange(9).reshape(-1, 1), 100 + torch.arange(9).reshape(-1, 1)

    def save(rows):
        return k[rows].clone(), v[rows].clone()

    def restore(cpu, rows):
        k[rows], v[rows] = cpu

    pool = NS(
        _logical_allocator=allocator, k_buffer=[k], v_buffer=[v],
        get_cpu_copy=save, load_cpu_copy=restore, cpu_offloading_chunk_size=1,
    )
    plan = PrefillKVCompaction(scheduler, pool, 4)
    plan.stage_on_free_rows()
    assert checked_budget == [16]
    assert prefix.tolist() == [1, 6]
    assert allocator._live_page_ids == {1, 6, 7}
    plan.commit()
    assert k[2:4, 0].tolist() == [6, 7]
    assert v[2:4, 0].tolist() == [106, 107]
    assert allocator._live_page_ids == {1, 2, 3}
    assert allocator.free_pages.tolist() == [4, 5, 6, 7, 8]
    assert request_rows.tolist() == [[1, 2, 3, 0]]
    assert prefix.tolist() == req.prefix_indices.tolist() == [1, 2]
    assert plan.cpu_bytes == 32
