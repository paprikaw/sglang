import pytest
import torch

from sglang.srt.elastic_pp.marlin_batch_probe import maybe_probe, probe


@pytest.fixture
def inputs():
    return dict(
        hidden_states=torch.arange(16).reshape(4, 4).float() + 1,
        w1=torch.empty((16, 1, 1)),
        w2=torch.empty((16, 1, 1)),
        topk_ids=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        topk_weights=torch.full((4, 2), 0.5),
        gating_output=torch.zeros((4, 16)),
        deterministic=True,
        inplace=False,
        workspace=torch.zeros(8, dtype=torch.int32),
    )


@pytest.mark.parametrize("composition_dependent", [False, True])
def test_probe_holds_anchor_constant_and_detects_only_composition(
    inputs, composition_dependent
):
    seen = []

    def forward(**kwargs):
        seen.append(kwargs)
        assert torch.equal(kwargs["hidden_states"][0], inputs["hidden_states"][0])
        assert torch.equal(kwargs["topk_ids"][0], inputs["topk_ids"][0])
        assert torch.equal(kwargs["topk_weights"][0], inputs["topk_weights"][0])
        assert kwargs["w1"] is inputs["w1"] and kwargs["w2"] is inputs["w2"]
        output = kwargs["hidden_states"].clone() * 2
        if composition_dependent and kwargs["topk_ids"].unique().numel() > 2:
            output.add_(0.125)
        return output

    result = probe(inputs, forward)
    assert result["single_repeat_equal"]
    assert result["cases"]["same_routes"]["anchor_equal"]
    assert (
        result["cases"]["shifted_routes"]["anchor_equal"] is not composition_dependent
    )
    assert result["cases"]["shifted_routes"]["batch_repeat_equal"]
    assert result["cases"]["shifted_routes"]["max_abs_delta"] == (
        0.125 if composition_dependent else 0
    )
    assert len(seen) == 6
    assert inputs["topk_ids"][0].tolist() == [1, 2]


def test_probe_rejects_expert_sharding(inputs):
    with pytest.raises(ValueError, match="unsharded"):
        probe(dict(inputs, expert_map=torch.arange(16)), lambda **kwargs: None)


def test_probe_never_runs_inside_graph_capture(monkeypatch, inputs, tmp_path):
    import sglang.srt.elastic_pp.marlin_batch_probe as module

    monkeypatch.setattr(module, "_finished", False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    maybe_probe(
        inputs, lambda **kwargs: pytest.fail("probe entered capture"), str(tmp_path)
    )
    assert not module._finished
    assert not list(tmp_path.iterdir())


def test_startup_zero_input_does_not_replace_nonzero_checks(
    monkeypatch, inputs, tmp_path
):
    import json

    import sglang.srt.elastic_pp.marlin_batch_probe as module

    monkeypatch.setattr(module, "_finished", False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    inputs["hidden_states"].zero_()
    maybe_probe(inputs, lambda **kwargs: kwargs["hidden_states"] * 2, str(tmp_path))
    result = json.loads((tmp_path / "marlin_batch_gpu0.json").read_text())
    assert result["reference_nonzero_values"] == 0
    assert len(result["synthetic_nonzero"]) == 3
    for case in result["synthetic_nonzero"].values():
        assert case["input_rms"] > 0 and case["reference_nonzero_values"] > 0
        assert case["cases"]["shifted_routes"]["anchor_equal"]
