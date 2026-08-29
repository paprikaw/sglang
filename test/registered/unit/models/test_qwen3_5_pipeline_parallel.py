import unittest
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.elastic_pp.hybrid_config import validate_qwen35_elastic_args
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models import qwen3_5
from sglang.srt.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestQwen3_5PipelineParallel(CustomTestCase):
    @staticmethod
    def _get_num_fused_shared_experts(layers, start_layer, end_layer):
        model = SimpleNamespace(
            model=SimpleNamespace(
                layers=layers,
                start_layer=start_layer,
                end_layer=end_layer,
            )
        )
        return Qwen3_5MoeForConditionalGeneration._get_num_fused_shared_experts(model)

    def test_get_num_fused_shared_experts_returns_zero_without_layers(self):
        model = SimpleNamespace(model=SimpleNamespace())

        num_fused_shared_experts = (
            Qwen3_5MoeForConditionalGeneration._get_num_fused_shared_experts(model)
        )

        self.assertEqual(num_fused_shared_experts, 0)

    def test_get_num_fused_shared_experts_uses_local_pp_layers(self):
        layers = [
            PPMissingLayer(),
            PPMissingLayer(),
            SimpleNamespace(
                mlp=SimpleNamespace(num_fused_shared_experts=1),
            ),
            SimpleNamespace(
                mlp=SimpleNamespace(num_fused_shared_experts=1),
            ),
        ]

        num_fused_shared_experts = self._get_num_fused_shared_experts(
            layers,
            start_layer=2,
            end_layer=4,
        )

        self.assertEqual(num_fused_shared_experts, 1)

    def test_get_num_fused_shared_experts_returns_zero_without_local_fusion(self):
        layers = [
            PPMissingLayer(),
            SimpleNamespace(mlp=SimpleNamespace()),
        ]

        num_fused_shared_experts = self._get_num_fused_shared_experts(
            layers,
            start_layer=1,
            end_layer=2,
        )

        self.assertEqual(num_fused_shared_experts, 0)


def test_elastic_wrapper_retains_output_endpoint_and_ignores_mtp_profile(monkeypatch):
    config = SimpleNamespace(num_hidden_layers=2, vocab_size=8, hidden_size=4)

    def init_base(self, *args):
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = None
        self.model = nn.Module()
        self.model.start_layer, self.model.end_layer = 0, 1
        self.model.layers = nn.ModuleList(
            [nn.Linear(4, 4, bias=False), PPMissingLayer()]
        )
        self.lm_head = PPMissingLayer()

    monkeypatch.setattr(qwen3_5.Qwen3VLForConditionalGeneration, "__init__", init_base)
    monkeypatch.setattr(
        qwen3_5,
        "get_parallel",
        lambda: SimpleNamespace(enable_elastic_pp=True, enable_dp_lm_head=False),
    )
    monkeypatch.setattr(
        qwen3_5,
        "ParallelLMHead",
        lambda vocab, hidden, **kw: nn.Linear(hidden, vocab, bias=False),
    )
    model = Qwen3_5MoeForConditionalGeneration.__new__(
        Qwen3_5MoeForConditionalGeneration
    )
    qwen3_5._Qwen3_5ElasticPPWrapperMixin.__init__(model, config)
    assert not isinstance(model.lm_head, PPMissingLayer)
    weights = [
        ("model.language_model.layers.0.weight", torch.ones(4, 4)),
        ("model.language_model.layers.1.qweight", torch.ones(4, 2, dtype=torch.int32)),
        ("mtp.layers.0.weight", torch.ones(20, 20)),
    ]
    assert len(list(qwen3_5._record_elastic_qwen35_weights(model, weights))) == 2
    profile = model.finalize_elastic_pp_weight_profile()
    assert profile.checkpoint_tensor_count == 2
    assert profile.checkpoint_total_weight_size == 16 * 4 + 8 * 4
    assert model.start_layer == 0 and model.end_layer == 1
    model._elastic_pp_loading_layer_ids = frozenset({1})
    assert not qwen3_5._skip_qwen35_pp_weight(model, 1)
    assert qwen3_5._skip_qwen35_pp_weight(model, 0)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"disable_radix_cache": False}, "prefix caching"),
        ({"max_running_requests": None}, "request capacity"),
        ({"speculative_algorithm": "EAGLE"}, "speculative decoding"),
    ],
)
def test_elastic_hybrid_unsupported_modes_fail_before_loading(override, message):
    args = dict(disable_radix_cache=True, max_running_requests=8, kv_cache_dtype="auto")
    validate_qwen35_elastic_args(SimpleNamespace(**args))
    args.update(override)
    with pytest.raises(ValueError, match=message):
        validate_qwen35_elastic_args(SimpleNamespace(**args))


@pytest.mark.parametrize(
    "arch", ["Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"]
)
def test_qwen35_text_only_launch_uses_existing_language_backbone(arch):
    args = SimpleNamespace(
        language_model_only=True,
        encoder_only=False,
        language_only=False,
        enable_prefix_mm_cache=False,
        enable_broadcast_mm_inputs_process=False,
        mm_enable_dp_encoder=False,
        disaggregation_mode="null",
        LANGUAGE_MODEL_ONLY_ARCHITECTURES=ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES,
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[arch])
        ),
    )
    ServerArgs._handle_language_model_only(args)
    model = SimpleNamespace(language_model_only=True)
    text = ("model.language_model.layers.0.weight", torch.ones(1))
    visual = ("model.visual.patch_embed.proj.weight", torch.ones(2))
    assert list(qwen3_5._record_elastic_qwen35_weights(model, [text, visual])) == [text]


if __name__ == "__main__":
    unittest.main()
