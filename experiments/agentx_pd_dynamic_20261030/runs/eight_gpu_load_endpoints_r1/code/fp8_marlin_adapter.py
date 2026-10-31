"""Opt-in, experiment-private standard-FP8 Expert Marlin W8A16 wiring.

Keeps native checkpoint loading, full local experts, StandardDispatcher, Qwen's
post-experts allreduce, and Dense FP8 Linear unchanged. Only this method's MoE
calls select the existing low-level Marlin FP8 scalar type. No torch import,
CUDA check, build, or GPU work happens at module import. ``install`` only wires
Python hooks; actual model construction/qualification belongs to Guard.

Unvalidated use needs both ENABLE_ENV=1 and QUALIFICATION_ENV=1. Formal use
instead needs a passing receipt at RECEIPT_ENV for this exact adapter source.
CPU tests are not a GPU qualification receipt.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import hashlib
import importlib
import json
import os
from pathlib import Path


ENABLE_ENV = "QWEN_A100_FP8_MOE_MARLIN"
QUALIFICATION_ENV = "QWEN_A100_FP8_MOE_MARLIN_QUALIFICATION"
RECEIPT_ENV = "QWEN_A100_FP8_MOE_MARLIN_RECEIPT"
ENGINE_COMMIT = "870269702285d94bdd926a50cc29ad1a20bcf60c"
_FP8_CALL = ContextVar("qwen_a100_fp8_marlin_call", default=False)
_INSTALLED = False


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def qualification_mode(environ=None):
    """Explicitly distinguish a test launch from a GPU-qualified launch."""
    environ = os.environ if environ is None else environ
    if environ.get(QUALIFICATION_ENV) == "1":
        return "qualification"
    path = environ.get(RECEIPT_ENV)
    if not path or not Path(path).is_absolute():
        raise RuntimeError("FP8 Marlin needs an explicit qualification flag or receipt")
    receipt = json.loads(Path(path).read_text())
    checks = receipt.get("checks", {})
    required = ("ep2", "ep4", "decode", "prefill", "changed_inputs")
    if (receipt.get("status") != "passed"
            or receipt.get("adapter_sha256") != source_sha256()
            or receipt.get("engine_commit") != ENGINE_COMMIT
            or any(checks.get(name) is not True for name in required)):
        raise RuntimeError("FP8 Marlin qualification receipt is incomplete or stale")
    return "validated"


@dataclass(frozen=True)
class Geometry:
    global_experts: int
    local_experts: int
    ep_size: int
    hidden_size: int
    intermediate_size: int

    @property
    def checkpoint_shapes(self):
        e, h, n = self.local_experts, self.hidden_size, self.intermediate_size
        return {"w13_weight": (e, 2 * n, h), "w2_weight": (e, h, n),
                "w13_weight_scale_inv": (e, 2 * n // 128, h // 128),
                "w2_weight_scale_inv": (e, h // 128, n // 128)}

    @property
    def packed_shapes(self):
        e, h, n = self.local_experts, self.hidden_size, self.intermediate_size
        return {"w13_weight": (e, h // 16, 2 * n * 4),
                "w2_weight": (e, n // 16, h * 4),
                "w13_weight_scale": (e, h // 128, 2 * n),
                "w2_weight_scale": (e, n // 128, h)}


def geometry_for(layer):
    g = Geometry(int(layer.num_experts), int(layer.num_local_experts),
                 int(layer.moe_ep_size), int(layer.hidden_size),
                 int(layer.intermediate_size_per_partition))
    if (g.ep_size not in (1, 2, 4) or g.local_experts <= 0
            or g.local_experts * g.ep_size != g.global_experts):
        raise ValueError("Expected complete, evenly sharded EP1/EP2/EP4 experts")
    if int(layer.moe_tp_size) != 1:
        raise ValueError("FP8 Marlin requires whole experts (MoE TP1)")
    if int(getattr(layer, "num_fused_shared_experts", 0)) != 0:
        raise ValueError("Use the separate native shared-expert MLP")
    if (g.hidden_size <= 0 or g.intermediate_size <= 0
            or g.hidden_size % 128 or g.intermediate_size % 128):
        raise ValueError("Require aligned block128 hidden/intermediate sizes")
    return g


def validate_config(config):
    if (not config.is_checkpoint_fp8_serialized
            or config.activation_scheme != "dynamic"
            or tuple(config.weight_block_size or ()) != (128, 128)
            or getattr(config, "use_mxfp8", False)
            or getattr(config, "is_fp4_experts", False)):
        raise ValueError("Only serialized E4M3FN FP8 block[128,128] is admitted")


def validate_shapes(layer, expected):
    for name, shape in expected.items():
        value = getattr(layer, name, None)
        if value is None or tuple(value.shape) != shape:
            raise ValueError(f"Unexpected {name} shape; expected {shape}")


class _LocalExpertsView:
    """Native packer iterates num_experts, although EP buffers are local.

    Proxy that count without mutating the model's global expert identity. Every
    parameter replacement/deletion and native dir()-based scale lookup still
    operates on the real nn.Module.
    """
    def __init__(self, layer, local_experts):
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "_local_experts", local_experts)

    def __getattr__(self, name):
        return self._local_experts if name == "num_experts" else getattr(self._layer, name)

    def __setattr__(self, name, value):
        setattr(self._layer, name, value)

    def __delattr__(self, name):
        delattr(self._layer, name)

    def __dir__(self):
        return sorted(set(dir(self._layer)) | {"num_experts"})


def scalar_selector(original, fp8_type):
    """Only adapter MoE calls may reinterpret eight bits as FP8, not INT8."""
    @wraps(original)
    def select(num_bits, has_zp, scales=None, global_scale=None):
        if not _FP8_CALL.get():
            return original(num_bits, has_zp, scales, global_scale)
        if num_bits != 8 or has_zp or global_scale is not None:
            raise ValueError("FP8 Marlin requires 8-bit weights without zero points")
        return fp8_type
    return select


def prepare_layer(layer, config, native_prepare, torch_module, check_support):
    """Keep FP8 bytes and multiplicative checkpoint scales; do not requantize."""
    if getattr(layer, "_qwen_a100_fp8_marlin_prepared", False):
        raise RuntimeError("A packed Marlin layer cannot be repacked")
    validate_config(config)
    geometry = geometry_for(layer)
    validate_shapes(layer, geometry.checkpoint_shapes)
    if layer.params_dtype != torch_module.bfloat16:
        raise ValueError("This adapter admits BF16 activations/scales only")
    if not check_support(layer, 128):
        raise ValueError("Native Marlin rejects this geometry/activation")
    for name in ("w13_weight", "w2_weight"):
        if getattr(layer, name).dtype != torch_module.float8_e4m3fn:
            raise ValueError("Expert checkpoint weights must remain E4M3FN FP8")
    for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
        scale = getattr(layer, name)
        if scale.dtype != torch_module.float32 or getattr(scale, "format_ue8m0", False):
            raise ValueError("Require original FP32 block scales, not UE8M0")
    if getattr(layer, "with_bias", False):
        raise ValueError("This Qwen adapter does not admit expert biases")
    layer.orig_dtype = layer.params_dtype
    layer.weight_block_size = [128, 128]
    # Existing helper: E,Nblocks,Kgroups -> E,Kgroups,Nblocks, repeat each
    # output block 128 times, native Marlin scale permutation + exponent-bias
    # fusion. scale_inv already multiplies dequantized FP8: do not reciprocate.
    native_prepare(_LocalExpertsView(layer, geometry.local_experts), size_k_first=False)
    validate_shapes(layer, geometry.packed_shapes)
    layer._qwen_a100_fp8_marlin_prepared = True
    layer._qwen_a100_fp8_marlin_geometry = geometry
    layer.dispatcher.set_quant_config({"weight_dtype": torch_module.bfloat16})
    return geometry


def build_quant_info(layer, quant_info_class):
    geometry = layer._qwen_a100_fp8_marlin_geometry
    mapping = getattr(layer.dispatcher, "local_expert_mapping", None)
    if geometry.ep_size > 1 and mapping is None:
        raise RuntimeError("EP needs native StandardDispatcher local expert mapping")
    # StandardDispatcher already maps global IDs to local IDs/-1. Native
    # fused_marlin uses expert_map only as an EP mask switch, not to remap IDs.
    return quant_info_class(
        w13_qweight=layer.w13_weight, w2_qweight=layer.w2_weight,
        w13_scales=layer.w13_weight_scale, w2_scales=layer.w2_weight_scale,
        w13_g_idx_sort_indices=None, w2_g_idx_sort_indices=None,
        weight_bits=8, is_k_full=True, expert_map=mapping,
        global_num_experts=geometry.local_experts)


def make_method_class(base_class, dependencies):
    """Native Fp8MoEMethod subclass preserves native weight-loader type tests."""
    class Fp8MarlinMoEMethod(base_class):
        def create_moe_runner(self, layer, moe_runner_config):
            dependencies.check_runtime()
            self.moe_runner_config = moe_runner_config
            self.runner = dependencies.MoeRunner(dependencies.marlin_backend, moe_runner_config)

        def process_weights_after_loading(self, layer):
            prepare_layer(layer, self.quant_config, dependencies.prepare,
                          dependencies.torch, dependencies.check_support)

        def apply(self, layer, dispatch_output):
            if not getattr(layer, "_qwen_a100_fp8_marlin_prepared", False):
                raise RuntimeError("FP8 Marlin layer is not prepared")
            if not dependencies.is_standard(dispatch_output.topk_output):
                raise ValueError("Require StandardTopKOutput before local EP mapping")
            if dispatch_output.hidden_states.dtype != dependencies.torch.bfloat16:
                raise ValueError("FP8 Marlin requires BF16 hidden states")
            info = build_quant_info(layer, dependencies.quant_info_class)
            token = _FP8_CALL.set(True)
            try:
                return self.runner.run(dispatch_output, quant_info=info)
            finally:
                _FP8_CALL.reset(token)
    return Fp8MarlinMoEMethod


def install_on(config_class, base_class, fused_moe_class, method_class):
    """Ignored layers and every Dense Linear retain their original method."""
    if config_class.__dict__.get("_qwen_a100_fp8_marlin_installed", False):
        return False
    original = config_class.get_quant_method

    @wraps(original)
    def get_quant_method(self, layer, prefix):
        method = original(self, layer, prefix)
        if isinstance(layer, fused_moe_class) and isinstance(method, base_class):
            validate_config(self)
            return method_class(self)
        return method

    config_class.get_quant_method = get_quant_method
    config_class._qwen_a100_fp8_marlin_installed = True
    return True


def install():
    """Install before model construction; do not query CUDA before rank setup."""
    global _INSTALLED
    if os.environ.get(ENABLE_ENV) != "1":
        return {"enabled": False, "reason": "explicit opt-in absent"}
    mode = qualification_mode()
    if _INSTALLED:
        return {"enabled": True, "installed": False, "mode": mode}
    from types import SimpleNamespace
    import torch
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.moe_runner import MoeRunner
    from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_moe_a2a_backend, get_moe_runner_backend
    from sglang.srt.layers.quantization.marlin_utils import check_moe_marlin_supports_layer
    from sglang.srt.layers.quantization.marlin_utils_fp8 import prepare_moe_fp8_layer_for_marlin
    from sgl_kernel.scalar_type import scalar_types

    def check_runtime():
        if torch.cuda.get_device_capability() != (8, 0):
            raise RuntimeError("This isolated adapter is restricted to SM80")
        if not get_moe_a2a_backend().is_none() or not get_moe_runner_backend().is_marlin():
            raise RuntimeError("Require none/allreduce EP and Marlin MoE")

    fused = importlib.import_module("sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe")
    dependencies = SimpleNamespace(
        torch=torch, prepare=prepare_moe_fp8_layer_for_marlin,
        check_support=check_moe_marlin_supports_layer, check_runtime=check_runtime,
        MoeRunner=MoeRunner, marlin_backend=MoeRunnerBackend.MARLIN,
        is_standard=TopKOutputChecker.format_is_standard,
        quant_info_class=MarlinMoeQuantInfo)
    method = make_method_class(Fp8MoEMethod, dependencies)
    fused.get_scalar_type = scalar_selector(fused.get_scalar_type, scalar_types.float8_e4m3fn)
    install_on(Fp8Config, Fp8MoEMethod, FusedMoE, method)
    _INSTALLED = True
    return {"enabled": True, "installed": True, "mode": mode,
            "transport": "native_allreduce", "expert_compute": "FP8_E4M3FN_W8A16_Marlin",
            "checkpoint_requantized": False, "scale_compute_dtype": "bfloat16",
            "original_scale_storage_dtype": "float32", "gpu_qualified": mode == "validated"}


def correctness_fixture(*, ep_size, ep_rank, all_reduce=None, device="cuda"):
    """GPU qualification entry, called ONLY by the external Guard-owned worker.

    Requires native runtime/parallel context initialized with TP=EP, DP=1,
    none+Marlin. It uses the real StandardDispatcher and Marlin runner with
    deterministic block-varying FP8 weights/scales and nonlocal expert routes.
    Each actual EP rank computes its full local experts. ``all_reduce`` must
    be the native TP group's sum for EP>1. No full model/weights are loaded.
    Returns evidence; never declares the full Qwen engine GPU-qualified.
    """
    from types import SimpleNamespace
    import torch
    import torch.nn.functional as F
    from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.layers.quantization.marlin_utils import check_moe_marlin_supports_layer
    from sglang.srt.layers.quantization.marlin_utils_fp8 import prepare_moe_fp8_layer_for_marlin
    from sglang.srt.runtime_context import get_parallel

    if os.environ.get(QUALIFICATION_ENV) != "1":
        raise RuntimeError("GPU fixture needs explicit qualification mode")
    if (ep_size not in (1, 2, 4) or not 0 <= ep_rank < ep_size
            or (ep_size > 1 and all_reduce is None)):
        raise ValueError("Actual EP rank and native sum collective are required")
    ps = get_parallel()
    if (ps.moe_ep_size != ep_size or ps.moe_ep_rank != ep_rank
            or ps.moe_tp_size != 1 or ps.tp_size != ep_size):
        raise RuntimeError("Fixture runtime must use actual whole-expert TP=EP, DP1")
    installed = install()
    if not installed["enabled"]:
        raise RuntimeError("FP8 adapter opt-in is required")
    e, h, n, topk = 8, 256, 256, 2
    local = e // ep_size
    generator = torch.Generator(device="cpu").manual_seed(50913)
    # Cast on CPU: SM80 is not asked to execute native FP8 conversion.
    w13 = torch.randint(-4, 5, (e, 2 * n, h), generator=generator).to(torch.float8_e4m3fn)
    w2 = torch.randint(-4, 5, (e, h, n), generator=generator).to(torch.float8_e4m3fn)
    s13 = torch.tensor([2.0 ** (-8 + (i % 5)) for i in range(e * 4 * 2)],
                       dtype=torch.float32).reshape(e, 4, 2)
    s2 = torch.tensor([2.0 ** (-8 + (i % 4)) for i in range(e * 2 * 2)],
                      dtype=torch.float32).reshape(e, 2, 2)
    # Preserve independent, unpacked FP32 references before native repacking.
    ref13 = (w13.float() * s13.repeat_interleave(128, 1).repeat_interleave(128, 2)).to(device)
    ref2 = (w2.float() * s2.repeat_interleave(128, 1).repeat_interleave(128, 2)).to(device)
    config = SimpleNamespace(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic",
                             weight_block_size=[128, 128], use_mxfp8=False, is_fp4_experts=False)
    layer = torch.nn.Module()
    for name, value in dict(num_experts=e, num_local_experts=local, moe_ep_size=ep_size,
                            moe_ep_rank=ep_rank, moe_tp_size=1, hidden_size=h,
                            intermediate_size_per_partition=n, num_fused_shared_experts=0,
                            params_dtype=torch.bfloat16, with_bias=False).items():
        setattr(layer, name, value)
    start, end = ep_rank * local, (ep_rank + 1) * local
    for name, value in (("w13_weight", w13), ("w2_weight", w2),
                         ("w13_weight_scale_inv", s13), ("w2_weight_scale_inv", s2)):
        layer.register_parameter(name, torch.nn.Parameter(value[start:end].to(device), requires_grad=False))
    runner_config = MoeRunnerConfig(num_experts=e, num_local_experts=local,
        hidden_size=h, intermediate_size_per_partition=n, top_k=topk,
        num_fused_shared_experts=0, params_dtype=torch.bfloat16, inplace=False,
        gate_up_interleaved=False)
    layer.moe_runner_config = runner_config
    layer.dispatcher = StandardDispatcher(runner_config)
    prepare_layer(layer, config, prepare_moe_fp8_layer_for_marlin, torch,
                  check_moe_marlin_supports_layer)
    runner = MoeRunner(MoeRunnerBackend.MARLIN, runner_config)
    rows = []
    for batch in (1, 16):
        for changed_input in (False, True):
            x = (torch.randn(batch, h, generator=generator) * .1
                 + (.07 if changed_input else 0)).to(device, torch.bfloat16)
            ids = torch.stack((torch.arange(batch) % e,
                               (torch.arange(batch) + e // 2) % e), 1).to(device, torch.int32)
            weights = torch.tensor([.625, .375], device=device).expand(batch, topk).contiguous()
            logits = torch.zeros(batch, e, device=device)
            expected = torch.zeros(batch, h, device=device, dtype=torch.float32)
            for token_idx in range(batch):
                for slot in range(topk):
                    expert = int(ids[token_idx, slot].item())
                    gu = x[token_idx].float() @ ref13[expert].T
                    expert_out = (F.silu(gu[:n]) * gu[n:]) @ ref2[expert].T
                    expected[token_idx] += weights[token_idx, slot] * expert_out
            dispatch = layer.dispatcher.dispatch(x, StandardTopKOutput(weights, ids, logits))
            info = build_quant_info(layer, MarlinMoeQuantInfo)
            token = _FP8_CALL.set(True)
            try:
                actual = runner.run(dispatch, quant_info=info).hidden_states
            finally:
                _FP8_CALL.reset(token)
            if all_reduce is not None:
                actual = all_reduce(actual)
            torch.cuda.synchronize()
            # BF16 intermediates versus independent FP32 reference; pointwise
            # atol handles true near-zero cancellation, not route mismatches.
            torch.testing.assert_close(actual.float(), expected, rtol=.03, atol=.003)
            error_rms = (actual.float() - expected).square().mean().sqrt()
            reference_rms = expected.square().mean().sqrt()
            relative_rms = float((error_rms / reference_rms.clamp_min(1e-12)).item())
            if relative_rms > .02:
                raise AssertionError(f"FP8 Marlin fixture relative RMS error {relative_rms} > .02")
            rows.append(dict(batch=batch, changed_input=changed_input, status="passed",
                             relative_rms_error=relative_rms,
                             max_abs_error=float((actual.float() - expected).abs().max().item())))
    return dict(status="passed", ep_size=ep_size, ep_rank=ep_rank, checks=rows,
                adapter_sha256=source_sha256(), engine_commit=ENGINE_COMMIT,
                fixture_only=True, full_model_qualified=False)
