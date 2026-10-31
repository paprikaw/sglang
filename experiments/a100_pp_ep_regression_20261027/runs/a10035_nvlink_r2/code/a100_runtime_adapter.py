"""Private BF16 Triton/DeepEP-normal interface; full native weights and routing.

DeepEP returns token-major rows and local expert IDs. The SAME native Triton
Expert function executes both EP and PP. Local weighted sums are then returned
by native DeepEP combine. No quantization, weight conversion, or second route
weight is introduced. An independent allgather reference validates this bridge
on loaded weights before serving, including a destination receiving no rows.
"""
from contextvars import ContextVar
from functools import wraps
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
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo, fused_experts_none_to_triton
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput, StandardCombineInput
    from sglang.srt.layers.moe.token_dispatcher import deepep as de
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    def quant_info(layer):
        return TritonMoeQuantInfo(w13_weight=layer.w13_weight,w2_weight=layer.w2_weight,
            fuse_swiglu_interleaved=layer.quant_method.w13_swiglu_interleaved)

    def compute(layer, dispatch):
        deep = isinstance(dispatch, de.DeepEPNormalDispatchOutput)
        if deep:
            layer._bridge_last_received_rows = dispatch.hidden_states.shape[0]
            if dispatch.hidden_states_scale is not None:
                raise RuntimeError('Require BF16 transport, without activation quantization')
            ids=dispatch.topk_ids.to(torch.int32)
            standard=StandardDispatchOutput(dispatch.hidden_states,None,
                StandardTopKOutput(dispatch.topk_weights,ids,dispatch.hidden_states.new_empty((ids.shape[0],0))))
        elif isinstance(dispatch,StandardDispatchOutput):
            standard=dispatch
        else:
            raise TypeError('Only Standard/DeepEP-normal are qualified')
        if standard.hidden_states.dtype != torch.bfloat16:
            raise RuntimeError('Require BF16 activations')
        if standard.hidden_states.shape[0] == 0:
            output=StandardCombineInput(standard.hidden_states)
        else:
            output=fused_experts_none_to_triton(standard,quant_info(layer),layer.moe_runner_config)
        if deep:
            return de.DeepEPNormalCombineInput(output.hidden_states,dispatch.topk_ids,dispatch.topk_weights)
        return output

    def apply(self,layer,dispatch_output):
        with scope(torch,f'component.expert.routed.{layer.layer_id}'):
            return compute(layer,dispatch_output)
    UnquantizedFusedMoEMethod.apply=apply
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    old_core=DeepEPMoE.run_moe_core
    def run_moe_core(self,dispatch_output):
        if isinstance(self.quant_method,UnquantizedFusedMoEMethod):
            return self.quant_method.apply(self,dispatch_output)
        return old_core(self,dispatch_output)
    DeepEPMoE.run_moe_core=run_moe_core
    def combine_a(self,hidden_states,topk_ids,topk_weights):
        return hidden_states,de.Buffer.capture() if self.async_finish else None
    de._DeepEPDispatcherImplNormal.combine_a=combine_a
    for name in ('dispatch','combine'):
        def wrap(fn,label):
            @wraps(fn)
            def call(self,*args,**kwargs):
                with scope(torch,f'component.communication.{label}.{LAYER.get()}'):
                    return fn(self,*args,**kwargs)
            return call
        setattr(de.DeepEPDispatcher,name,wrap(getattr(de.DeepEPDispatcher,name),name))
    from sglang.srt.model_executor.model_runner import ModelRunner
    previous=ModelRunner.load_model
    @wraps(previous)
    def load(self,*args,**kwargs):
        import logging
        ps=self.ps
        class RankIdentity(logging.Filter):
            def filter(self,record):
                if ps.pp_size>1 and ps.tp_size==1 and str(record.msg).startswith('CPU weight cache load:'):
                    record.msg=f'[PP{ps.pp_rank} TP{ps.tp_rank}] '+record.msg
                return True
        logger=logging.getLogger('sglang.srt.weight_cache.cpu_canonical_loader')
        rank_filter=RankIdentity();logger.addFilter(rank_filter)
        try:
            result=previous(self,*args,**kwargs)
        finally:
            logger.removeFilter(rank_filter)
        qualify_model(self,compute,quant_info)
        attach_scopes(self.model,torch)
        return result
    ModelRunner.load_model=load
    _INSTALLED=True


def scope(torch, name):
    from contextlib import nullcontext
    # No per-layer annotation work in the unprofiled performance cohorts.
    return (torch.profiler.record_function(name)
            if torch.autograd._profiler_enabled() else nullcontext())

def attach_scopes(model, torch):
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
            # Qwen3.5's Full Attention is a METHOD on the decoder layer,
            # not a self_attn child module. Include QKV/RoPE/core/output
            # projection, matching the linear_attn module's scope boundary.
            if hasattr(module, 'self_attention'):
                module.self_attention = make_wrapper(module.self_attention, layer,
                    f'component.attention.full.{layer}')
        elif suffix == 'self_attn':
            label = f'component.attention.full.{layer}'
        elif suffix == 'linear_attn':
            label = f'component.attention.linear.{layer}'
        elif suffix == 'mlp.shared_expert':
            label = f'component.expert.shared.{layer}'
        if label is None:
            continue
        original = module.forward
        module.forward = make_wrapper(original, layer, label)

def qualify_model(runner, compute, quant_info):
    """Before KV allocation, compare actual routed weights with an
    independent all-gather/local-experts/all-reduce reference. Both use the
    native BF16 Triton math; this checks the new routing/weight/combine bridge.
    """
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace
    from sglang.srt.distributed import get_moe_ep_group
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.moe_runner.triton import fused_experts_none_to_triton
    ps = runner.ps
    layers = [m for m in runner.model.modules()
              if hasattr(m, 'w13_weight') and hasattr(m, 'dispatcher')]
    expected = 40 // ps.pp_size
    if torch.cuda.get_device_capability()[0] != 8 or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("Expected an A100 GPU")
    for m in layers:
        if m.w13_weight.dtype != torch.bfloat16 or m.w2_weight.dtype != torch.bfloat16:
            raise RuntimeError("Require native BF16 expert weights")
        if not m.quant_method.runner.runner_backend.is_triton():
            raise RuntimeError("Both arms require the same Triton Expert backend")
    def geometry(m):
        return SimpleNamespace(global_experts=m.num_experts,local_experts=m.num_local_experts,
            hidden_size=m.hidden_size,intermediate_size=m.intermediate_size_per_partition)
    if len(layers) != expected:
        raise RuntimeError(f'Full layer coverage differs: {len(layers)} != {expected}')
    if ps.moe_ep_size == 2:
        layer = layers[0]
        group = get_moe_ep_group().device_group
        g = geometry(layer)
        rows = []
        for count, delta, route in [(1, 0, 'spread'), (4, 0, 'spread'),
                                   (4, .07, 'spread'), (16, .02, 'spread'),
                                   (1, .03, 'destination0'), (4, .04, 'destination0')]:
            gen = torch.Generator(device='cpu').manual_seed(791 + ps.moe_ep_rank)
            x = (torch.randn(count, g.hidden_size, generator=gen) * .1 + delta).to('cuda', torch.bfloat16)
            ids = ((torch.arange(count)[:, None] * 17 + torch.arange(8)[None, :] * 31
                    + ps.moe_ep_rank * 13) % g.global_experts).to('cuda', torch.int32)
            if route == 'destination0':
                ids = torch.arange(8, device='cuda', dtype=torch.int32).expand(count,8).contiguous()
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
            expected_value = fused_experts_none_to_triton(standard, quant_info(layer),
                                      layer.moe_runner_config).hidden_states
            dist.all_reduce(expected_value, group=group)
            expected_value = expected_value[ps.moe_ep_rank * count:(ps.moe_ep_rank + 1) * count]
            # Exercise the ACTUAL EP module forward, including its legacy
            # subclass entrypoint, not just a hand-assembled dispatcher call.
            result = layer(hidden_states=x,
                topk_output=StandardTopKOutput(weights, ids, x.new_empty((count, 0))))
            torch.cuda.synchronize()
            torch.testing.assert_close(result.float(), expected_value.float(), rtol=.04, atol=.02)
            rel = float(((result.float() - expected_value.float()).square().mean().sqrt()
                         / expected_value.float().square().mean().sqrt().clamp_min(1e-12)).item())
            if rel > .02:
                raise AssertionError(f'Bridge numerical RMS error {rel}')
            rows.append(dict(tokens_per_source=count, input_delta=delta, route=route,
                             received_rows=layer._bridge_last_received_rows, relative_rms=rel))
    else:
        if ps.moe_ep_size != 1 or ps.pp_size != 2:
            raise RuntimeError('Expected full EP2 or PP2/EP1 model')
        layer = layers[0]
        g = geometry(layer)
        rows = []
        for count in (1, 4, 16):
            gen = torch.Generator(device='cpu').manual_seed(791 + ps.pp_rank)
            x = (torch.randn(count, g.hidden_size, generator=gen) * .1).to('cuda', torch.bfloat16)
            ids = ((torch.arange(count)[:, None] * 17 + torch.arange(8)[None, :] * 31)
                   % g.global_experts).to('cuda', torch.int32)
            weights = torch.arange(1., 9., device='cuda')
            weights = (weights / weights.sum()).expand(count, 8).contiguous()
            topk = StandardTopKOutput(weights, ids, x.new_empty((count, 0)))
            standard = StandardDispatchOutput(x, None, topk)
            expected_value = fused_experts_none_to_triton(standard, quant_info(layer),
                                      layer.moe_runner_config).hidden_states
            # Traverse the native EP1 dispatcher/container, including its
            # keyword-based quant-method interface, before starting serving.
            result = layer(hidden_states=x, topk_output=topk)
            torch.cuda.synchronize()
            torch.testing.assert_close(result.float(), expected_value.float(), rtol=.04, atol=.02)
            rows.append(dict(local_ep1=True, tokens_per_source=count,
                             actual_layer_forward=True))
    root = Path(os.environ['EP2_PP2_RUN']) / os.environ['EP2_PP2_ARM'] / 'qualification'
    root.mkdir(parents=True, exist_ok=True)
    data = dict(passed=True, tp_rank=ps.tp_rank, pp_rank=ps.pp_rank,
        dp_rank=ps.dp_rank, expert_parallel=ps.moe_ep_size, layers=len(layers),
        experts_per_layer=layers[0].num_local_experts,
        dtype='BF16_Triton', device=torch.cuda.get_device_name(), checks=rows,
        geometry=[vars(geometry(m)) for m in layers])
    path = root / f'pp{ps.pp_rank}.tp{ps.tp_rank}.json'
    path.write_text(json.dumps(data, indent=2) + '\n')
    print('EP2_PP2_MODEL_QUALIFIED', json.dumps({k:v for k,v in data.items() if k not in ('geometry','checks')}), flush=True)
    torch.cuda.empty_cache()
