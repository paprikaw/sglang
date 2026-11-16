from types import SimpleNamespace

import pytest
import torch

from sglang.srt.elastic_pp.state_content_audit import (
    FencedStateContentAudit,
    _rows,
    _tensor_record,
)
from sglang.srt.elastic_pp.topology import DynamicPPTransition, PPTopology


class Pool:
    recurrent_slot_count = 6

    def __init__(self, interval, *, capacity=9):
        self.pairs = {}
        for layer in (1, 3):
            if interval.start <= layer < interval.end:
                self.pairs[layer] = tuple(
                    (
                        torch.arange(capacity * width).reshape(capacity, 1, width)
                        + layer * 100
                        + i * 10
                    ).to(torch.bfloat16)
                    for i, width in enumerate((3, 2))
                )
        ids = self.recurrent_layer_ids(interval)
        conv = (
            torch.stack([torch.arange(36).reshape(6, 2, 3) + l * 100 for l in ids])
            if ids
            else torch.empty((0, 6, 2, 3))
        )
        temporal = conv.clone() + 7
        self.req_to_token_pool = SimpleNamespace(
            mamba_map={l: i for i, l in enumerate(ids)},
            mamba_allocator=SimpleNamespace(free_slots=torch.tensor([2, 3, 5])),
        )
        self.mamba_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(
                at_layer_idx=lambda i: SimpleNamespace(
                    conv=[conv[i]], temporal=temporal[i]
                )
            )
        )

    @staticmethod
    def elastic_pp_kv_layer_ids(interval):
        return tuple(l for l in (1, 3) if interval.start <= l < interval.end)

    @staticmethod
    def recurrent_layer_ids(interval):
        return tuple(l for l in (0, 2) if interval.start <= l < interval.end)

    def get_key_buffer(self, layer):
        return self.pairs[layer][0]

    def get_value_buffer(self, layer):
        return self.pairs[layer][1]


@pytest.fixture
def audit_case():
    source = PPTopology.from_layer_counts([2, 0, 2])
    target = PPTopology.from_layer_counts([1, 2, 1], generation=1)
    transition = DynamicPPTransition.build(source, target)
    snapshots = [
        FencedStateContentAudit(Pool(source.layer_range(r)), r).capture(
            transition, torch.tensor([3, 1])
        )
        for r in range(3)
    ]
    return transition, snapshots


@pytest.mark.parametrize("rank", range(3))
def test_moved_retained_and_reindexed_gdn_match_after_capacity_growth(audit_case, rank):
    transition, snapshots = audit_case
    audit = FencedStateContentAudit(
        Pool(transition.target.layer_range(rank), capacity=13), rank
    )
    audit.verify(transition, snapshots)
    assert audit.stats["passed"]
    assert audit.stats["verified_bytes"] > 0


@pytest.mark.parametrize("kind", ["k", "v", "conv_0", "temporal"])
def test_single_corrupted_live_row_is_rejected(audit_case, kind):
    transition, snapshots = audit_case
    pool = Pool(transition.target.layer_range(1), capacity=13)
    audit = FencedStateContentAudit(pool, 1)
    for _, component, tensor in audit._tensors(transition.target.layer_range(1)):
        if component == kind:
            tensor[1].reshape(-1)[0] += 1000
    with pytest.raises(RuntimeError, match=f"component={kind}"):
        audit.verify(transition, snapshots)
    assert not audit.stats["passed"]


def test_unused_rows_do_not_invent_state_corruption(audit_case):
    transition, snapshots = audit_case
    pool = Pool(transition.target.layer_range(1), capacity=13)
    audit = FencedStateContentAudit(pool, 1)
    for _, kind, tensor in audit._tensors(transition.target.layer_range(1)):
        tensor[2 if kind in ("k", "v") else 3].fill_(-999)
    audit.verify(transition, snapshots)
    assert audit.stats["passed"]


def test_padding_gdn_slot_zero_is_verified(audit_case):
    transition, snapshots = audit_case
    pool = Pool(transition.target.layer_range(0))
    pool.mamba_pool.mamba_cache.at_layer_idx(0).temporal[0].fill_(-1)
    with pytest.raises(RuntimeError, match="component=temporal"):
        FencedStateContentAudit(pool, 0).verify(transition, snapshots)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_rank",
        "duplicate_rank",
        "stale_generation",
        "wrong_owner",
        "missing_tensor",
    ],
)
def test_incomplete_or_stale_evidence_cannot_pass(audit_case, fault):
    transition, snapshots = audit_case
    if fault == "missing_rank":
        snapshots.pop()
    elif fault == "duplicate_rank":
        snapshots[1]["rank"] = 0
    elif fault == "stale_generation":
        snapshots[1]["generation"] = 100
    elif fault == "wrong_owner":
        snapshots[0]["records"][0]["layer"] = 3
    else:
        snapshots[0]["records"].pop(0)
    audit = FencedStateContentAudit(Pool(transition.target.layer_range(1)), 1)
    with pytest.raises((RuntimeError, ValueError)):
        audit.verify(transition, snapshots)


def test_nan_payload_and_signed_zero_are_audited_as_bytes():
    tensor = torch.tensor([[float("nan"), -0.0], [1.0, 2.0]])
    rows = _rows([0, 1])
    first = _tensor_record(0, "k", tensor, rows, chunk_rows=1)
    assert first == _tensor_record(0, "k", tensor.clone(), rows, chunk_rows=2)
    tensor[0, 1] = 0.0
    assert first["digest"] != _tensor_record(0, "k", tensor, rows)["digest"]


@pytest.mark.parametrize("rows", [[1, 1], [-1], [[1, 2]]])
def test_bad_indices_cannot_be_hashed(rows):
    with pytest.raises(ValueError):
        _rows(rows)


def test_cpu_spill_source_is_explicitly_out_of_diagnostic_scope(audit_case):
    transition, _ = audit_case
    pool = Pool(transition.source.layer_range(0))
    pool._elastic_pp_cpu_spill = object()
    with pytest.raises(NotImplementedError, match="live source"):
        FencedStateContentAudit(pool, 0).capture(transition, torch.tensor([1]))
