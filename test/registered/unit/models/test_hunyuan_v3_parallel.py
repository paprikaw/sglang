"""CPU-only HYV3 PP/DP/DeepEP wiring tests, without importing CUDA backends.

The model's class bodies are executed with lightweight layer doubles. This
checks model orchestration and loader behavior, not numerical GPU kernels.
"""

import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional, Tuple, Union
from unittest.mock import Mock

import torch
from torch import nn


SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/models/hunyuan_v3.py"


class Layer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args, self.kwargs = args, kwargs
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, *args, **kwargs):
        return x, None


class Missing(Layer):
    pass


class Norm(Layer):
    def forward(self, x, residual=None):
        return x if residual is None else (x + residual, None)


class TopK(Layer):
    def forward(self, *args, **kwargs):
        self.forward_args = args, kwargs
        return "topk"

    def empty_topk_output(self, device):
        return "empty_topk"


class Experts(Layer):
    def forward(self, hidden_states, topk_output):
        self.last_topk = topk_output
        return hidden_states + 2

    @staticmethod
    def make_expert_params_mapping(**kwargs):
        return [("experts.w13_weight", "experts.0.gate_proj.weight", 0, "w1")]


class Communicator:
    def __init__(self, **kwargs):
        self.kwargs, self.calls = kwargs, []

    def prepare_attn(self, x, residual, batch):
        self.calls.append("attention")
        return x, residual

    def prepare_mlp(self, x, residual, batch):
        self.calls.append("mlp")
        return x, residual

    def postprocess_layer(self, x, residual, batch):
        self.calls.append("output")
        return x, residual


class Param:
    def __init__(self):
        self.loaded = []

    def weight_loader(self, *args, **kwargs):
        self.loaded.append((args, kwargs))


def namespace(rank=0, size=1, a2a=False, dense_dp=False):
    pp = SimpleNamespace(
        is_first_rank=rank == 0,
        is_last_rank=rank == size - 1,
        rank_in_group=rank,
        world_size=size,
    )
    parallel = SimpleNamespace(
        tp_size=8, attn_tp_size=4, attn_tp_rank=2, moe_tp_size=1, moe_ep_size=8
    )
    env = dict(
        torch=torch,
        nn=nn,
        Optional=Optional,
        Tuple=Tuple,
        Iterable=Iterable,
        Union=Union,
        QuantizationConfig=object,
        PretrainedConfig=object,
        ForwardBatch=object,
        PPProxyTensors=dict,
        get_pp_group=lambda: pp,
        get_parallel=lambda: parallel,
        get_stream=lambda _: None,
        is_cuda=lambda: False,
        is_dp_attention_enabled=lambda: True,
        get_moe_a2a_backend=lambda: SimpleNamespace(is_none=lambda: not a2a),
        enable_moe_dense_fully_dp=lambda: dense_dp,
        should_skip_post_experts_all_reduce=lambda **_: False,
        moe_expert_parallel_all_reduce=Mock(side_effect=AssertionError("EP reduce")),
        moe_tensor_model_parallel_all_reduce=Mock(
            side_effect=AssertionError("TP reduce")
        ),
        get_is_capture_mode=lambda: False,
        get_moe_impl_class=lambda _: Experts,
        MergedColumnParallelLinear=Layer,
        QKVParallelLinear=Layer,
        ReplicatedLinear=Layer,
        RowParallelLinear=Layer,
        RMSNorm=Norm,
        SiluAndMul=nn.Identity,
        RadixAttention=Layer,
        VocabParallelEmbedding=Layer,
        ParallelLMHead=Layer,
        PPMissingLayer=Missing,
        TopK=TopK,
        FusedMoE=Experts,
        get_rope=lambda *a, **k: Layer(),
        get_rope_config=lambda _: (10000, None),
        LogitsProcessor=lambda _: Mock(return_value="logits"),
        LayerCommunicator=Communicator,
        LayerScatterModes=SimpleNamespace(init_new=lambda **k: k),
        ExpertLocationDispatchInfo=SimpleNamespace(init_new=lambda x: x),
        ModelConfigForExpertLocation=lambda **k: k,
        default_weight_loader=lambda p, w: p.weight_loader(p, w),
        get_layer_id=lambda name: (
            int(re.search(r"layers\.(\d+)", name)[1])
            if re.search(r"layers\.(\d+)", name)
            else None
        ),
    )
    tree = ast.parse(SOURCE.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    exec(compile(tree, str(SOURCE), "exec"), env)
    return env, pp


def config(**changes):
    fields = dict(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        hidden_act="silu",
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        route_norm=True,
        rms_norm_eps=1e-5,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=2,
        num_hidden_layers=4,
        vocab_size=32,
        first_k_dense_replace=1,
        num_nextn_predict_layers=1,
        tie_word_embeddings=False,
        router_scaling_factor=2.826,
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


class TestHYV3Parallel(unittest.TestCase):
    def test_expert_location_metadata_uses_canonical_module(self):
        imports = [
            node
            for node in ast.parse(SOURCE.read_text()).body
            if isinstance(node, ast.ImportFrom)
        ]
        modules = [
            node.module
            for node in imports
            if any(alias.name == "ExpertLocationDispatchInfo" for alias in node.names)
        ]
        self.assertEqual(modules, ["sglang.srt.eplb.expert_location_dispatch"])

    def test_attention_uses_attention_group_not_global_tp(self):
        env, _ = namespace()
        layer = env["HYV3Attention"](config(), 16, 8, 4)
        self.assertEqual((layer.num_heads, layer.num_kv_heads), (2, 1))
        for projection in (layer.qkv_proj, layer.o_proj):
            self.assertEqual(
                (projection.kwargs["tp_rank"], projection.kwargs["tp_size"]), (2, 4)
            )
        self.assertFalse(layer.o_proj.kwargs["reduce_results"])

    def test_shared_experts_are_local_for_dispatch(self):
        env, _ = namespace(a2a=True)
        layer = env["HYV3MoEFused"](config(), 1)
        self.assertIsInstance(layer.experts, Experts)
        for projection in (layer.shared_mlp.gate_up_proj, layer.shared_mlp.down_proj):
            self.assertEqual(
                (projection.kwargs["tp_rank"], projection.kwargs["tp_size"]), (0, 1)
            )
        self.assertFalse(layer.experts.kwargs["inplace"])
        self.assertEqual(layer.topk.kwargs["layer_id"], 1)

    def test_empty_dispatch_rank_still_combines_without_gate(self):
        env, _ = namespace(a2a=True)
        layer = env["HYV3MoEFused"](config(), 1)
        layer.gate.forward = Mock(side_effect=AssertionError("empty gate"))
        layer.shared_mlp.forward = Mock(side_effect=AssertionError("empty shared"))
        output = layer(torch.empty(0, 16), SimpleNamespace(num_token_non_padded=0))
        self.assertEqual(tuple(output.shape), (0, 16))
        self.assertEqual(layer.experts.last_topk, "empty_topk")
        env["moe_expert_parallel_all_reduce"].assert_not_called()

    def test_dispatch_shared_addition_and_route_metadata(self):
        env, _ = namespace(a2a=True)
        layer = env["HYV3MoEFused"](config(), 2)
        layer.shared_mlp = nn.Identity()
        output = layer(torch.ones(3, 16), SimpleNamespace(num_token_non_padded=3))
        torch.testing.assert_close(output, torch.full((3, 16), 4.0))
        self.assertEqual(layer.topk.forward_args[1]["num_token_non_padded"], 3)
        self.assertEqual(layer.topk.forward_args[1]["expert_location_dispatch_info"], 2)
        env["moe_expert_parallel_all_reduce"].assert_not_called()
        env["moe_tensor_model_parallel_all_reduce"].assert_not_called()

    def test_dense_sparse_boundary_and_communicator_order(self):
        env, _ = namespace(a2a=True, dense_dp=True)
        dense = env["HYV3DecoderLayer"](config(), 0)
        self.assertEqual(dense.mlp.gate_up_proj.kwargs["tp_size"], 1)
        self.assertFalse(dense.layer_scatter_modes["is_layer_sparse"])
        self.assertTrue(dense.layer_scatter_modes["is_next_layer_sparse"])
        sparse = env["HYV3DecoderLayer"](config(), 1)
        self.assertFalse(sparse.layer_scatter_modes["is_previous_layer_sparse"])
        sparse.self_attn = nn.Identity()
        sparse.self_attn.forward = lambda **k: k["hidden_states"]
        sparse(
            torch.ones(2, 16),
            torch.ones(2, 16),
            SimpleNamespace(num_token_non_padded=2),
            None,
        )
        self.assertEqual(
            sparse.layer_communicator.calls, ["attention", "mlp", "output"]
        )
        self.assertFalse(sparse.layer_communicator.kwargs["allow_reduce_scatter"])

    def test_pipeline_allocates_only_local_layers_and_returns_proxy(self):
        for rank in (0, 1):
            env, pp = namespace(rank, 2)
            built = []

            class Decoder(Layer):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    built.append(kwargs["layer_id"])

                def forward(self, pos, hidden, batch, residual):
                    return hidden + 1, hidden

            env["HYV3DecoderLayer"] = Decoder

            def make_layers(count, factory, pp_rank, pp_size, prefix):
                start, end = pp_rank * 2, (pp_rank + 1) * 2
                return (
                    nn.ModuleList(
                        [
                            (
                                factory(i, f"{prefix}.{i}")
                                if start <= i < end
                                else Missing()
                            )
                            for i in range(count)
                        ]
                    ),
                    start,
                    end,
                )

            env["make_layers"] = make_layers
            model = env["HYV3ForCausalLM"](config())
            self.assertEqual(built, [rank * 2, rank * 2 + 1])
            self.assertEqual(isinstance(model.model.embed_tokens, Missing), rank == 1)
            self.assertEqual(isinstance(model.lm_head, Missing), rank == 0)
            output = model(
                None,
                None,
                None,
                input_embeds=torch.zeros(2, 16),
                pp_proxy_tensors={
                    "hidden_states": torch.zeros(2, 16),
                    "residual": torch.zeros(2, 16),
                },
            )
            self.assertEqual(isinstance(output, dict), rank == 0)
            if rank == 0:
                self.assertEqual(set(output), {"hidden_states", "residual"})
            else:
                self.assertEqual(output, "logits")

    def test_loader_filters_other_stage_and_mtp_and_maps_router(self):
        env, pp = namespace(1, 2)
        model = object.__new__(env["HYV3ForCausalLM"])
        object.__setattr__(model, "config", config())
        object.__setattr__(model, "pp_group", pp)
        object.__setattr__(model, "model", SimpleNamespace(start_layer=2, end_layer=4))
        params = {
            name: Param()
            for name in (
                "model.layers.0.input_layernorm.weight",
                "model.layers.2.mlp.gate.weight",
                "model.layers.4.input_layernorm.weight",
                "model.layers.2.mlp.experts.w13_weight",
            )
        }
        model.named_parameters = lambda: iter(params.items())
        model.load_weights(
            [
                (name, torch.ones(1))
                for name in (
                    "model.layers.0.input_layernorm.weight",
                    "model.layers.2.mlp.router.gate.weight",
                    "model.layers.4.input_layernorm.weight",
                    "model.layers.2.mlp.experts.0.gate_proj.weight",
                )
            ]
        )
        self.assertFalse(params["model.layers.0.input_layernorm.weight"].loaded)
        self.assertFalse(params["model.layers.4.input_layernorm.weight"].loaded)
        self.assertEqual(len(params["model.layers.2.mlp.gate.weight"].loaded), 1)
        self.assertEqual(
            params["model.layers.2.mlp.experts.w13_weight"].loaded[0][1],
            {"expert_id": 0, "shard_id": "w1"},
        )

    def test_tied_embedding_loads_stage_last_head(self):
        env, pp = namespace(1, 2)
        model = object.__new__(env["HYV3ForCausalLM"])
        object.__setattr__(model, "config", config(tie_word_embeddings=True))
        object.__setattr__(model, "pp_group", pp)
        object.__setattr__(model, "model", SimpleNamespace(start_layer=2, end_layer=4))
        head = Param()
        model.named_parameters = lambda: iter([("lm_head.weight", head)])
        model.load_weights(
            [
                ("model.embed_tokens.weight", torch.ones(1)),
                ("lm_head.weight", torch.zeros(1)),
            ]
        )
        self.assertEqual(len(head.loaded), 1)
        torch.testing.assert_close(head.loaded[0][0][1], torch.ones(1))


if __name__ == "__main__":
    unittest.main()
