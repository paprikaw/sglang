"""Private annotations and loader rank labels; native Hopper mathematics.

No Marlin bridge, weight conversion, routing change, or fabricated cache hit.
The native DeepEP buffers are charged before the KV allocator runs.
"""
from contextlib import nullcontext
from contextvars import ContextVar
from functools import wraps
import json
import logging
import os
from pathlib import Path

_DONE = False
_LAYER = ContextVar('hopper_component_layer', default=-1)


def install():
    global _DONE
    if _DONE:
        return
    import torch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher
    import runtime_setup
    import pp_ready_send
    runtime_setup.install_on(ModelRunner)
    pp_ready_send.install(torch)
    import coalesce
    coalesce.install()

    def scope(name):
        return torch.profiler.record_function(name) if torch.autograd._profiler_enabled() else nullcontext()

    def instrument(original, name, *, layer=None):
        @wraps(original)
        def call(*args, **kwargs):
            token = _LAYER.set(layer) if layer is not None else None
            try:
                with scope(name + '.' + str(_LAYER.get())):
                    return original(*args, **kwargs)
            finally:
                if token is not None:
                    _LAYER.reset(token)
        return call

    UnquantizedFusedMoEMethod.apply = instrument(UnquantizedFusedMoEMethod.apply, 'component.expert.routed')
    for name in ('dispatch', 'combine'):
        setattr(DeepEPDispatcher, name, instrument(getattr(DeepEPDispatcher, name),
                                                  'component.communication.' + name))
    previous = ModelRunner.load_model

    @wraps(previous)
    def load(self, *args, **kwargs):
        ps = self.ps
        class ActualRank(logging.Filter):
            def filter(self, record):
                if ps.pp_size > 1 and ps.tp_size == 1 and str(record.msg).startswith('CPU weight cache load:'):
                    record.msg = f'[PP{ps.pp_rank} TP{ps.tp_rank}] ' + record.msg
                return True
        logger = logging.getLogger('sglang.srt.weight_cache.cpu_canonical_loader')
        rank_filter = ActualRank()
        logger.addFilter(rank_filter)
        try:
            result = previous(self, *args, **kwargs)
        finally:
            logger.removeFilter(rank_filter)
        experts = [m for m in self.model.modules() if hasattr(m, 'w13_weight') and hasattr(m, 'dispatcher')]
        if len(experts) != 40 // ps.pp_size:
            raise RuntimeError('The complete model layer count differs')
        if any(m.w13_weight.dtype != torch.bfloat16 for m in experts):
            raise RuntimeError('Unquantized BF16 expert weights were not retained')
        if torch.cuda.get_device_capability()[0] != 9:
            raise RuntimeError('This protocol requires an H100')
        for name, module in self.model.named_modules():
            parts = name.split('.')
            if 'layers' not in parts:
                continue
            index = parts.index('layers')
            if index + 1 >= len(parts) or not parts[index + 1].isdigit():
                continue
            layer = int(parts[index + 1])
            suffix = '.'.join(parts[index + 2:])
            label = {'': 'component.layer', 'self_attn': 'component.attention.full',
                     'linear_attn': 'component.attention.linear',
                     'mlp.shared_expert': 'component.expert.shared'}.get(suffix)
            if label:
                module.forward = instrument(module.forward, label, layer=layer)
            if suffix == '' and hasattr(module, 'self_attention'):
                module.self_attention = instrument(module.self_attention, 'component.attention.full', layer=layer)
        root = Path(os.environ['EP2_PP2_RUN']) / os.environ['EP2_PP2_ARM'] / 'qualification'
        root.mkdir(parents=True, exist_ok=True)
        row = dict(passed=True, layers=len(experts), pp_rank=ps.pp_rank, tp_rank=ps.tp_rank,
                   ep_size=ps.moe_ep_size, weight_dtype=str(experts[0].w13_weight.dtype),
                   expert_shape=list(experts[0].w13_weight.shape), native_kernel=True,
                   actual_moe_runner=str(experts[0].quant_method.runner.runner_backend),
                   loaded_gpu_allocated_bytes=torch.cuda.memory_allocated(),
                   device=torch.cuda.get_device_name(), compute_capability=list(torch.cuda.get_device_capability()))
        (root / f'pp{ps.pp_rank}.tp{ps.tp_rank}.json').write_text(json.dumps(row, indent=2))
        print('H100_FULL_MODEL_QUALIFIED', json.dumps(row), flush=True)
        return result
    ModelRunner.load_model = load
    _DONE = True
