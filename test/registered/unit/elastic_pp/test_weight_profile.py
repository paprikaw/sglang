import pytest
import torch
from torch import nn

from sglang.srt.elastic_pp.capacity import estimate_kv_capacity
from sglang.srt.elastic_pp.weight_profile import (
    ElasticPPWeightProfile,
    FirstLoadWeightProfiler,
)
from sglang.srt.layers.utils import PPMissingLayer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Layer(nn.Module):
    def __init__(self, elements: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(elements, dtype=torch.float32))


class _ResidentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_Layer(3), PPMissingLayer(), PPMissingLayer()]
        )
        self.model.embed_tokens = _Layer(5)


def _consume(iterable):
    return list(iterable)


def test_first_stream_profiles_all_layers_and_calibrates_resident_bytes():
    profiler = FirstLoadWeightProfiler(num_hidden_layers=3)
    checkpoint = [
        ("model.layers.0.weight", torch.zeros(2, dtype=torch.float16)),
        ("model.layers.1.weight", torch.zeros(3, dtype=torch.float16)),
        ("model.layers.2.weight", torch.zeros(4, dtype=torch.float16)),
        ("model.embed_tokens.weight", torch.zeros(7, dtype=torch.float16)),
    ]

    assert _consume(profiler.record_initial_stream(checkpoint)) == checkpoint
    profile = profiler.finalize(_ResidentModel())

    # Layer 0 uses its packed/resident 3 * fp32 bytes. Layers 1 and 2 use
    # complete first-stream checkpoint sizes despite being absent locally.
    assert profile.layer_weight_sizes == (12, 6, 8)
    assert profile.non_layer_weight_size == 20
    assert profile.total_weight_size == 32
    assert profile.checkpoint_total_weight_size == 32
    assert profile.checkpoint_non_layer_weight_size == 14
    assert profile.checkpoint_tensor_count == 4
    assert profile.resident_tensor_count == 2
    assert profile.transformer_weight_size(1, 3) == 14
    assert profile.target_weight_size(1, 3) == 34


def test_subsequent_dynamic_stream_does_not_mutate_initial_profile():
    profiler = FirstLoadWeightProfiler(num_hidden_layers=1)
    _consume(
        profiler.record_initial_stream(
            [("model.layers.0.weight", torch.zeros(2, dtype=torch.float16))]
        )
    )
    first_profile = profiler.finalize(_Layer(1))

    second = [("model.layers.0.weight", torch.zeros(99, dtype=torch.float32))]
    assert _consume(profiler.record_initial_stream(second)) == second
    assert profiler.finalize(_Layer(7)) is first_profile
    assert first_profile.layer_weight_sizes == (4,)


def test_finalize_rejects_incomplete_layer_coverage():
    profiler = FirstLoadWeightProfiler(num_hidden_layers=2)
    _consume(
        profiler.record_initial_stream([("model.layers.0.weight", torch.zeros(1))])
    )

    with pytest.raises(ValueError, match=r"missing=\[1\]"):
        profiler.finalize(_Layer(1))


def test_finalize_requires_complete_initial_stream_consumption():
    profiler = FirstLoadWeightProfiler(num_hidden_layers=1)
    stream = profiler.record_initial_stream([("model.layers.0.weight", torch.zeros(1))])
    next(stream)

    with pytest.raises(ValueError, match="not completely consumed"):
        profiler.finalize(_Layer(1))
    stream.close()


def test_capacity_uses_exact_layer_weights_and_heterogeneous_kv_pages():
    profile = ElasticPPWeightProfile(
        layer_weight_sizes=(10, 20, 30),
        non_layer_weight_size=5,
        total_weight_size=65,
        checkpoint_total_weight_size=65,
        checkpoint_non_layer_weight_size=5,
        checkpoint_tensor_count=4,
        resident_tensor_count=4,
    )

    estimate = estimate_kv_capacity(
        profile=profile,
        start_layer=1,
        end_layer=3,
        total_device_memory_bytes=200,
        device_memory_utilization=0.8,
        kv_block_bytes_by_layer=(2, 4, 6),
        runtime_overhead_bytes=5,
        block_multiple=4,
    )

    assert estimate.usable_memory_bytes == 160
    assert estimate.target_weight_bytes == 55
    assert estimate.kv_budget_bytes == 100
    assert estimate.kv_bytes_per_block == 10
    assert estimate.raw_num_blocks == 10
    assert estimate.aligned_num_blocks == 8
