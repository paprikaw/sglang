"""Check the mathematical route/padding contract, including masked EP routes."""

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size_deterministic,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("block", [8, 16, 32])
@pytest.mark.parametrize(
    "shape,experts",
    [((0, 8), 256), ((1, 8), 256), ((64, 8), 256), ((32, 2), 4)],
)
def test_stable_alignment_matches_route_reference(shape, experts, block):
    generator = torch.Generator().manual_seed(11)
    routes = torch.randint(-1, experts, shape, generator=generator, dtype=torch.int32)
    actual, expert_ids, padded = moe_align_block_size_deterministic(
        routes, block, experts
    )
    flat = routes.flatten().tolist()
    expected, expected_experts = [], []
    for expert in range(-1, experts):
        rows = [i for i, value in enumerate(flat) if value == expert]
        rows += [len(flat)] * (-len(rows) % block)
        expected.extend(rows)
        expected_experts.extend([expert] * (len(rows) // block))
    assert padded.item() == len(expected)
    assert actual[: len(expected)].tolist() == expected
    assert actual[len(expected) :].eq(len(flat)).all()
    assert expert_ids[: len(expected_experts)].tolist() == expected_experts
    assert expert_ids[len(expected_experts) :].eq(-1).all()
    repeated = moe_align_block_size_deterministic(routes, block, experts)
    assert all(
        torch.equal(a, b) for a, b in zip((actual, expert_ids, padded), repeated)
    )
