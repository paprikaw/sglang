"""CPU tests of the BF16 LL bridge; these do not validate CUDA/transport."""

import ast
import importlib.util
import sys
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NamedTuple, Optional, Tuple
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[5]
MOE = ROOT / "python/sglang/srt/layers/moe"


def load_class(path, name, **extra):
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    env = dict(
        torch=torch,
        NamedTuple=NamedTuple,
        Optional=Optional,
        Tuple=Tuple,
        TopKOutput=object,
        DispatchOutputFormat=object,
        CombineInputFormat=object,
        TopKOutputFormat=object,
        **extra
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    return env[name]


def load_bridge():
    spec = importlib.util.spec_from_file_location(
        "bf16_deepep_cpu", MOE / "moe_runner/bf16_deepep.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BRIDGE = load_bridge()
StandardDispatchOutput = load_class(
    MOE / "token_dispatcher/standard.py", "StandardDispatchOutput"
)
StandardTopKOutput = load_class(MOE / "topk.py", "StandardTopKOutput")
DeepEPLLCombineInput = load_class(
    MOE / "token_dispatcher/deepep.py", "DeepEPLLCombineInput"
)


@dataclass
class Config:
    top_k: int = 8
    inplace: bool = True
    no_combine: bool = False
    apply_router_weight_on_input: bool = True
    routed_scaling_factor: float = 2.826


def adapter_env(fused):
    tree = ast.parse((MOE / "moe_runner/triton.py").read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "fused_experts_deepep_ll_to_triton_bf16"
    )
    node.decorator_list = []
    # Annotations are evaluated lazily in the production module.
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    env = dict(torch=torch, replace=replace, fused_experts_none_to_triton=fused)
    exec(compile(module, str(MOE / "moe_runner/triton.py"), "exec"), env)
    modules = {}
    for suffix, attrs in {
        "moe_runner.bf16_deepep": {
            "prepare_bf16_deepep_ll": BRIDGE.prepare_bf16_deepep_ll
        },
        "token_dispatcher": {
            "DispatchOutputChecker": SimpleNamespace(
                format_is_deepep_ll=lambda d: d.format == "ll"
            )
        },
        "token_dispatcher.deepep": {"DeepEPLLCombineInput": DeepEPLLCombineInput},
        "token_dispatcher.standard": {"StandardDispatchOutput": StandardDispatchOutput},
        "topk": {"StandardTopKOutput": StandardTopKOutput},
    }.items():
        name = "sglang.srt.layers.moe." + suffix
        stub = ModuleType(name)
        stub.__dict__.update(attrs)
        modules[name] = stub
    return env["fused_experts_deepep_ll_to_triton_bf16"], modules


def moe_class(mode):
    path = MOE / "ep_moe/layer.py"
    tree = ast.parse(path.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepEPMoE"
    )
    node.body = [
        n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    ]

    class Base:
        def __init__(self, **kwargs):
            self.w13_weight = torch.empty(0, dtype=torch.bfloat16)
            self.dispatcher = SimpleNamespace(set_quant_config=Mock())

    runner = SimpleNamespace(
        is_humming=lambda: False,
        is_auto=lambda: False,
        is_flashinfer_cutedsl=lambda: False,
        is_triton=lambda: True,
        is_deep_gemm=lambda: False,
    )
    env = dict(
        torch=torch,
        FusedMoE=Base,
        get_moe_runner_backend=lambda: runner,
        get_moe_a2a_backend=lambda: SimpleNamespace(is_deepep=lambda: True),
        get_deepep_mode=lambda: SimpleNamespace(is_low_latency=lambda: mode == "ll"),
        _use_aiter=False,
        _is_npu=False,
        _is_hip=False,
        deep_gemm_wrapper=SimpleNamespace(ENABLE_JIT_DEEPGEMM=False),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), env)
    return env["DeepEPMoE"]


class TestBF16DeepEPLL(unittest.TestCase):
    def test_unquantized_triton_low_latency_selects_bf16_dispatch(self):
        layer = moe_class("ll")(
            num_experts=8, top_k=2, hidden_size=16, intermediate_size=8, layer_id=1
        )
        self.assertTrue(layer.deprecate_flag)
        layer.dispatcher.set_quant_config.assert_called_once_with(
            {"dispatcher_output_dtype": "bf16"}
        )

    def test_unquantized_triton_normal_mode_rejected_at_construction(self):
        with self.assertRaisesRegex(NotImplementedError, "low_latency"):
            moe_class("normal")(
                num_experts=8, top_k=2, hidden_size=16, intermediate_size=8, layer_id=1
            )

    def test_local_expert_ids_and_nan_padding(self):
        hidden = torch.full((3, 4, 2), float("nan"), dtype=torch.bfloat16)
        hidden[0, :2] = torch.tensor([[1, 2], [3, 4]])
        hidden[2, :1] = torch.tensor([[5, 6]])
        flat, weights, ids, valid = BRIDGE.prepare_bf16_deepep_ll(
            hidden, torch.tensor([2, 0, 1], dtype=torch.int32)
        )
        self.assertEqual(tuple(flat.shape), (12, 2))
        self.assertEqual(
            ids[:, 0].tolist(), [0, 0, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1]
        )
        self.assertEqual(weights[:, 0].tolist(), [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        self.assertTrue(torch.isfinite(flat).all())
        torch.testing.assert_close(flat.reshape_as(hidden)[valid], hidden[valid])
        self.assertTrue((flat.reshape_as(hidden)[~valid] == 0).all())

    def test_all_empty_and_zero_slots(self):
        for slots in (0, 4):
            hidden = torch.full((2, slots, 3), float("nan"), dtype=torch.bfloat16)
            flat, weights, ids, _ = BRIDGE.prepare_bf16_deepep_ll(
                hidden, torch.zeros(2, dtype=torch.int32)
            )
            self.assertEqual(tuple(flat.shape), (2 * slots, 3))
            self.assertTrue((flat == 0).all())
            self.assertTrue((ids == -1).all())
            self.assertTrue((weights == 0).all())

    def test_invalid_dtype_and_count_shape_rejected(self):
        with self.assertRaises(TypeError):
            BRIDGE.prepare_bf16_deepep_ll(
                torch.zeros(2, 4, 3), torch.zeros(2, dtype=torch.int32)
            )
        with self.assertRaises(ValueError):
            BRIDGE.prepare_bf16_deepep_ll(
                torch.zeros(2, 4, 3, dtype=torch.bfloat16),
                torch.zeros(3, dtype=torch.int32),
            )

    def test_adapter_preserves_slots_and_applies_routing_only_at_combine(self):
        calls = []

        def fake_fused(dispatch, quant, config):
            calls.append(config)
            self.assertIsNone(dispatch.hidden_states_scale)
            self.assertFalse(config.inplace)
            self.assertFalse(config.apply_router_weight_on_input)
            self.assertIsNone(config.routed_scaling_factor)
            self.assertEqual(config.top_k, 1)
            ids = dispatch.topk_output.topk_ids[:, 0]
            # A different simple expert function for each local expert.
            output = dispatch.hidden_states * (ids + 2).clamp(min=0).unsqueeze(1)
            return SimpleNamespace(hidden_states=output)

        function, modules = adapter_env(fake_fused)
        original_ids = torch.tensor([[0, 2]], dtype=torch.int64)
        original_weights = torch.tensor([[0.25, 2.576]])
        hidden = torch.tensor(
            [[[1, 2], [float("nan"), float("nan")]], [[3, 4], [5, 6]]],
            dtype=torch.bfloat16,
        )
        dispatch = SimpleNamespace(
            format="ll",
            hidden_states=hidden,
            hidden_states_scale=None,
            masked_m=torch.tensor([1, 2], dtype=torch.int32),
            topk_ids=original_ids,
            topk_weights=original_weights,
        )
        quant = SimpleNamespace(
            w13_weight=torch.empty(0, dtype=torch.bfloat16),
            w2_weight=torch.empty(0, dtype=torch.bfloat16),
            use_mxfp8=False,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
        )
        config = Config()
        with patch.dict(sys.modules, modules):
            output = function(dispatch, quant, config)
        self.assertEqual(len(calls), 1)
        self.assertEqual(config.top_k, 8)
        self.assertIs(output.topk_ids, original_ids)
        self.assertIs(output.topk_weights, original_weights)
        expected = torch.tensor(
            [[[2, 4], [0, 0]], [[9, 12], [15, 18]]], dtype=torch.bfloat16
        )
        torch.testing.assert_close(output.hidden_states, expected)
        # Native combine can now weight the expert results once, not twice.
        combined = output.hidden_states[0, 0].float() * original_weights[0, 0]
        combined += output.hidden_states[1, 0].float() * original_weights[0, 1]
        torch.testing.assert_close(combined, torch.tensor([23.684, 31.912]))

    def test_adapter_rejects_normal_format(self):
        function, modules = adapter_env(lambda *args: self.fail("expert kernel called"))
        with patch.dict(sys.modules, modules), self.assertRaises(NotImplementedError):
            function(SimpleNamespace(format="normal"), None, None)


if __name__ == "__main__":
    unittest.main()
