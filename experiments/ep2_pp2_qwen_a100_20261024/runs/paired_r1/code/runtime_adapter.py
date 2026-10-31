"""Experiment-private FP8 Marlin/DeepEP-normal bridge; no model truncation.

Both arms use the same existing W8A16 Marlin compute. DeepEP-normal already
returns token-major rows with LOCAL expert IDs and routing weights. Marlin
reduces the locally selected experts; DeepEP combine sums destination-rank
contributions back to the source. No routing weight is applied a second time.
CUDA Graph is off in both arms. This is not the CE or low-latency protocol.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import importlib
import json
import os
from pathlib import Path

LAYER = ContextVar('ep2_pp2_layer', default=-1)
_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import torch
    import fp8_marlin_adapter as base
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.moe_runner.marlin import (
        MarlinMoeQuantInfo, fused_experts_none_to_marlin)
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.token_dispatcher import deepep as de
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend
    from sglang.srt.layers.quantization.marlin_utils import check_moe_marlin_supports_layer
    from sglang.srt.layers.quantization.marlin_utils_fp8 import prepare_moe_fp8_layer_for_marlin
    from sgl_kernel.scalar_type import scalar_types

    fused = importlib.import_module('sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe')
    fused.get_scalar_type = base.scalar_selector(fused.get_scalar_type, scalar_types.float8_e4m3fn)

    def quant_info(layer):
        g = layer._qwen_a100_fp8_marlin_geometry
        mapping = getattr(layer.dispatcher, 'local_expert_mapping', None)
        if g.ep_size > 1 and mapping is None:
            # Native fused_marlin uses a non-None map to admit -1 route slots.
            # IDs returned by DeepEP are already local: no second remapping.
            mapping = layer._bridge_mask_marker
        return MarlinMoeQuantInfo(w13_qweight=layer.w13_weight,
            w2_qweight=layer.w2_weight, w13_scales=layer.w13_weight_scale,
            w2_scales=layer.w2_weight_scale, w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None, weight_bits=8, is_k_full=True,
            expert_map=mapping, global_num_experts=g.local_experts)

    def compute(layer, dispatch):
        if isinstance(dispatch, de.DeepEPNormalDispatchOutput):
            if dispatch.hidden_states_scale is not None:
                raise RuntimeError('Bridge requires native BF16 transport')
            ids = dispatch.topk_ids.to(torch.int32)
            topk = StandardTopKOutput(dispatch.topk_weights, ids,
                dispatch.hidden_states.new_empty((ids.shape[0], 0)))
            standard = StandardDispatchOutput(dispatch.hidden_states, None, topk)
        elif isinstance(dispatch, StandardDispatchOutput):
            standard = dispatch
        else:
            raise TypeError('Only Standard and DeepEP NORMAL are qualified')
        if standard.hidden_states.dtype != torch.bfloat16:
            raise RuntimeError('Both arms require BF16 activations')
        token = base._FP8_CALL.set(True)
        try:
            output = fused_experts_none_to_marlin(standard, quant_info(layer),
                                                 layer._bridge_runner_config)
        finally:
            base._FP8_CALL.reset(token)
        if isinstance(dispatch, de.DeepEPNormalDispatchOutput):
            return de.DeepEPNormalCombineInput(output.hidden_states,
                                               dispatch.topk_ids, dispatch.topk_weights)
        return output

    class BridgeMethod(Fp8MoEMethod):
        def create_moe_runner(self, layer, config):
            if not get_moe_runner_backend().is_marlin():
                raise RuntimeError('Both arms must use Marlin')
            if not (get_moe_a2a_backend().is_none() or get_moe_a2a_backend().is_deepep()):
                raise RuntimeError('Unqualified expert transport')
            self.moe_runner_config = config
            layer._bridge_runner_config = config

        def process_weights_after_loading(self, layer):
            base.prepare_layer(layer, self.quant_config, prepare_moe_fp8_layer_for_marlin,
                               torch, check_moe_marlin_supports_layer)
            layer._bridge_mask_marker = torch.empty(0, device=layer.w13_weight.device,
                                                    dtype=torch.int32)

        def apply(self, layer, dispatch):
            with scope(torch, f'component.expert.routed.{layer.layer_id}'):
                return compute(layer, dispatch)

    base.install_on(Fp8Config, Fp8MoEMethod, FusedMoE, BridgeMethod)
    # The original normal dispatcher only admits its old DeepGEMM/AITER
    # callers here. Marlin above has already reduced weighted local experts,
    # so the native combine core can consume the same token-major result.
    def combine_a(self, hidden_states, topk_ids, topk_weights):
        previous = de.Buffer.capture() if self.async_finish else None
        return hidden_states, previous
    de._DeepEPDispatcherImplNormal.combine_a = combine_a

    for name, category in [('dispatch', 'dispatch'), ('combine', 'combine')]:
        original = getattr(de.DeepEPDispatcher, name)
        def make_wrapper(fn, label):
            @wraps(fn)
            def wrapped(self, *args, **kwargs):
                with scope(torch, f'component.communication.{label}.{LAYER.get()}'):
                    return fn(self, *args, **kwargs)
            return wrapped
        setattr(de.DeepEPDispatcher, name, make_wrapper(original, category))

    from sglang.srt.model_executor.model_runner import ModelRunner
    old_load = ModelRunner.load_model
    @wraps(old_load)
    def load(self, *args, **kwargs):
        result = old_load(self, *args, **kwargs)
        qualify_model(self, compute, quant_info)
        attach_scopes(self.model, torch)
        return result
    ModelRunner.load_model = load
    _INSTALLED = True


def scope(torch, name):
    from contextlib import nullcontext
    # No per-layer annotation work in the unprofiled performance cohorts.
    return (torch.profiler.record_function(name)
            if torch.autograd._profiler_enabled() else nullcontext())


def attach_scopes(model, torch):
    for name, module in model.named_modules():
        parts = name.split('.')
        if 'layers' not in parts:
            continue
        i = parts.index('layers')
        if i + 1 >= len(parts) or not parts[i + 1].isdigit():
            continue
        layer = int(parts[i + 1])
        suffix = '.'.join(parts[i + 2:])
        label = None
        if not suffix:
            label = f'layer.{layer}'
        elif suffix == 'self_attn':
            label = f'component.attention.full.{layer}'
        elif suffix == 'linear_attn':
            label = f'component.attention.linear.{layer}'
        elif suffix == 'mlp.shared_expert':
            label = f'component.expert.shared.{layer}'
        if label is None:
            continue
        original = module.forward
        def make_wrapper(fn, selected_layer, selected_label):
            @wraps(fn)
            def wrapped(*args, **kwargs):
                token = LAYER.set(selected_layer)
                try:
                    with scope(torch, selected_label):
                        return fn(*args, **kwargs)
                finally:
                    LAYER.reset(token)
            return wrapped
        module.forward = make_wrapper(original, layer, label)


def qualify_model(runner, compute, quant_info):
    """Before KV allocation, compare actual routed weights with an
    independent all-gather/local-experts/all-reduce reference. Both use the
    unchanged Marlin math; this checks the new routing/weight/combine bridge.
    """
    import torch
    import torch.distributed as dist
    import fp8_marlin_adapter as base
    from sglang.srt.distributed import get_moe_ep_group
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.moe_runner.marlin import fused_experts_none_to_marlin
    ps = runner.ps
    layers = [m for m in runner.model.modules()
              if hasattr(m, '_qwen_a100_fp8_marlin_geometry')]
    expected = 48 // ps.pp_size
    if len(layers) != expected:
        raise RuntimeError(f'Full layer coverage differs: {len(layers)} != {expected}')
    if ps.moe_ep_size == 2:
        layer = layers[0]
        group = get_moe_ep_group().device_group
        g = layer._qwen_a100_fp8_marlin_geometry
        rows = []
        for count, delta in [(1, 0), (4, 0), (4, .07), (16, .02)]:
            gen = torch.Generator(device='cpu').manual_seed(791 + ps.moe_ep_rank)
            x = (torch.randn(count, g.hidden_size, generator=gen) * .1 + delta).to('cuda', torch.bfloat16)
            ids = ((torch.arange(count)[:, None] * 17 + torch.arange(8)[None, :] * 31
                    + ps.moe_ep_rank * 13) % g.global_experts).to('cuda', torch.int32)
            weights = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.], device='cuda')
            weights = (weights / weights.sum()).expand(count, 8).contiguous()
            all_x, all_ids, all_w = [], [], []
            for tensor, output in [(x, all_x), (ids, all_ids), (weights, all_w)]:
                output.extend(torch.empty_like(tensor) for _ in range(2))
                dist.all_gather(output, tensor, group=group)
            global_ids = torch.cat(all_ids)
            start = ps.moe_ep_rank * g.local_experts
            local_ids = torch.where((global_ids >= start) & (global_ids < start + g.local_experts),
                                    global_ids - start, -1)
            standard = StandardDispatchOutput(torch.cat(all_x), None,
                StandardTopKOutput(torch.cat(all_w), local_ids, x.new_empty((2 * count, 0))))
            token = base._FP8_CALL.set(True)
            try:
                expected_value = fused_experts_none_to_marlin(standard, quant_info(layer),
                                      layer._bridge_runner_config).hidden_states
            finally:
                base._FP8_CALL.reset(token)
            dist.all_reduce(expected_value, group=group)
            expected_value = expected_value[ps.moe_ep_rank * count:(ps.moe_ep_rank + 1) * count]
            dispatch = layer.dispatcher.dispatch(x, StandardTopKOutput(weights, ids, x.new_empty((count, 0))))
            result = layer.dispatcher.combine(compute(layer, dispatch))
            torch.cuda.synchronize()
            torch.testing.assert_close(result.float(), expected_value.float(), rtol=.04, atol=.02)
            rel = float(((result.float() - expected_value.float()).square().mean().sqrt()
                         / expected_value.float().square().mean().sqrt().clamp_min(1e-12)).item())
            if rel > .02:
                raise AssertionError(f'Bridge numerical RMS error {rel}')
            rows.append(dict(tokens_per_source=count, input_delta=delta, relative_rms=rel))
    else:
        if ps.moe_ep_size != 1 or ps.pp_size != 2:
            raise RuntimeError('Expected full EP2 or PP2/EP1 model')
        rows = [dict(local_ep1=True, new_transport_bridge_not_used=True)]
    root = Path(os.environ['EP2_PP2_RUN']) / os.environ['EP2_PP2_ARM'] / 'qualification'
    root.mkdir(parents=True, exist_ok=True)
    data = dict(passed=True, tp_rank=ps.tp_rank, pp_rank=ps.pp_rank,
        dp_rank=ps.dp_rank, expert_parallel=ps.moe_ep_size, layers=len(layers),
        experts_per_layer=layers[0]._qwen_a100_fp8_marlin_geometry.local_experts,
        dtype='FP8_W8A16_Marlin', checks=rows,
        geometry=[vars(m._qwen_a100_fp8_marlin_geometry) for m in layers])
    path = root / f'pp{ps.pp_rank}.tp{ps.tp_rank}.json'
    path.write_text(json.dumps(data, indent=2) + '\n')
    print('EP2_PP2_MODEL_QUALIFIED', json.dumps({k:v for k,v in data.items() if k not in ('geometry','checks')}), flush=True)
    torch.cuda.empty_cache()
