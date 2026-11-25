from types import SimpleNamespace

import pytest
import torch

from sglang.srt.elastic_pp.batch_numerics_probe import (
    compare_batches,
    probe_dense,
    probe_gdn,
    synthetic_rows,
)


@pytest.mark.parametrize("changed", [False, True])
def test_compare_detects_composition_without_confusing_repeatability(changed):
    def run(size):
        return {"output": torch.ones(8) + (0.25 if changed and size > 1 else 0)}

    result = compare_batches(run)
    assert result["reference"]["output"]["nonzero_values"] == 8
    assert result["single_repeat"]["output"]["equal"]
    for case in result["cases"].values():
        assert case["output"]["equal"] is not changed
        assert case["output"]["batch_repeat_equal"]


def test_synthetic_anchor_is_independent_of_batch_size():
    inputs = synthetic_rows((4, 2, 3), device="cpu", dtype=torch.bfloat16)
    single = synthetic_rows((1, 2, 3), device="cpu", dtype=torch.bfloat16)
    assert torch.equal(inputs[:1], single)
    assert inputs.count_nonzero() > 0
    assert not torch.equal(inputs[0], inputs[1])


def test_dense_uses_real_parameters_and_a_private_workspace():
    layer = SimpleNamespace(
        weight=torch.arange(12).reshape(4, 3).float(),
        weight_scale=torch.ones(1),
        workspace=torch.full((8,), 7),
        input_size_per_partition=4,
        output_size_per_partition=3,
    )

    def forward(method, private, inputs, bias):
        assert private.weight is layer.weight
        assert private.weight_scale is layer.weight_scale
        assert private.workspace.data_ptr() != layer.workspace.data_ptr()
        private.workspace.add_(1)
        return inputs @ private.weight + bias

    result = probe_dense(None, layer, torch.ones(3), forward)
    assert result["reference"]["output"]["nonzero_values"] > 0
    assert all(case["output"]["equal"] for case in result["cases"].values())
    assert layer.workspace.tolist() == [7] * 8


@pytest.mark.parametrize("composition_dependent", [False, True])
def test_gdn_resets_sandbox_and_does_not_write_live_state(composition_dependent):
    state = torch.full((6, 2, 3, 3), 17.0)
    original_state = state.clone()
    actual_inputs = torch.full((2, 8), 23.0)
    kwargs = dict(
        ssm_states=state,
        cache_indices=torch.tensor([3, 5], dtype=torch.int32),
        A_log=torch.ones(2),
        dt_bias=torch.ones(2),
        scale=0.125,
        num_v_heads=2,
        head_v_dim=3,
        replayssm_d=None,
    )

    def forward(mixed_qkv, a, b, ssm_states, cache_indices, **options):
        assert ssm_states.data_ptr() != state.data_ptr()
        assert options["A_log"] is kwargs["A_log"]
        assert cache_indices.tolist() == list(range(mixed_qkv.shape[0]))
        ssm_states.add_(mixed_qkv.sum(-1)[:, None, None, None])
        if composition_dependent and mixed_qkv.shape[0] > 1:
            ssm_states.add_(0.125)
        return ssm_states.mean(-1).unsqueeze(0)

    result = probe_gdn(
        actual_inputs, actual_inputs[:, :2], actual_inputs[:, :2], kwargs, forward
    )
    assert torch.equal(state, original_state)
    assert torch.all(actual_inputs == 23)
    assert result["single_repeat"]["state"]["equal"]
    assert result["single_repeat"]["output"]["equal"]
    for case in result["cases"].values():
        assert case["state"]["equal"] is not composition_dependent
        assert case["state"]["batch_repeat_equal"]


def test_gdn_does_not_silently_ignore_replayssm_state():
    result = probe_gdn(None, None, None, {"replayssm_d": torch.ones(1)}, None)
    assert "skipped" in result
