"""Charge native DeepEP memory before the native KV capacity allocator.

Only initialization order changes. Reuse the real model dispatchers and the
native process-local buffer singleton; no dummy forward, alternative group,
dispatch mode change, transport change, or replacement kernel is introduced.
Imports remain CPU-safe until the installed method actually runs.
"""
from __future__ import annotations

from functools import wraps
import importlib
import json
import logging


FIELDS = ('group', 'hidden_size', 'params_bytes', 'deepep_mode',
          'num_max_dispatch_tokens_per_rank', 'num_experts')


def dispatcher_parameters(model, dispatcher_class, expected_ep_size):
    """Find real dispatchers behind MaybeTbo, and reject mixed singleton users."""
    implementations = []
    seen = set()

    def visit(dispatcher):
        if dispatcher is None or id(dispatcher) in seen:
            return
        seen.add(id(dispatcher))
        if isinstance(dispatcher, dispatcher_class):
            for name in ('_normal_dispatcher', '_low_latency_dispatcher'):
                value = getattr(dispatcher, name, None)
                if value is not None:
                    implementations.append(value)
        else:
            for inner in getattr(dispatcher, '_inners', ()):
                visit(inner)

    for module in model.modules():
        visit(getattr(module, 'dispatcher', None))
    if not implementations:
        if expected_ep_size != 1:
            raise RuntimeError('EP layout has no native DeepEP dispatcher to initialize')
        return None
    first = implementations[0]
    values = tuple(getattr(first, field) for field in FIELDS)
    if first.group.size() != expected_ep_size or expected_ep_size <= 1:
        raise RuntimeError('Native DeepEP group differs from the planned EP layout')
    for other in implementations[1:]:
        observed = tuple(getattr(other, field) for field in FIELDS)
        if observed[0] is not values[0] or observed[1:] != values[1:]:
            raise RuntimeError('Model dispatchers disagree on the shared DeepEP buffer')
    return values


def preinitialize(runner):
    import torch
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer, DeepEPDispatcher

    values = dispatcher_parameters(runner.model, DeepEPDispatcher, runner.ps.moe_ep_size)
    if values is None:
        runner._qwen_pre_kv_deepep = {'initialized': False, 'reason': 'EP1'}
        return
    if DeepEPBuffer._state().buffer is not None:
        raise RuntimeError('Unexpected preexisting native DeepEP buffer before KV qualification')

    def memory():
        free, total = torch.cuda.mem_get_info()
        return dict(free_bytes=free, total_bytes=total,
                    torch_allocated_bytes=torch.cuda.memory_allocated(),
                    torch_reserved_bytes=torch.cuda.memory_reserved())

    logger = logging.getLogger(__name__)
    before = memory()
    # Return cached loading slabs before native cuMemCreate; native KV profiling
    # also empties this cache later, but that would be too late for DeepEP.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    reclaimed = memory()
    logger.info('QWEN_PRE_KV_DEEPEP_BEGIN %s', json.dumps(dict(
        ep_size=values[0].size(), hidden_size=values[1], param_bytes=values[2],
        mode=str(values[3]), max_dispatch_tokens_per_rank=values[4],
        num_experts=values[5], before=before, after_empty_cache=reclaimed), sort_keys=True))
    buffer = DeepEPBuffer.get_deepep_buffer(*values)
    torch.cuda.synchronize()
    runner._qwen_pre_kv_deepep = dict(initialized=True, ep_size=values[0].size(),
        num_nvl_bytes=buffer.num_nvl_bytes, num_rdma_bytes=buffer.num_rdma_bytes,
        before=before, after_empty_cache=reclaimed, after_initialize=memory())
    logger.info('QWEN_PRE_KV_DEEPEP_READY %s',
                json.dumps(runner._qwen_pre_kv_deepep, sort_keys=True))


def install_on(model_runner_class, initialize=preinitialize):
    if model_runner_class.__dict__.get('_qwen_pre_kv_deepep_installed', False):
        return
    original = model_runner_class.alloc_memory_pool

    @wraps(original)
    def alloc_memory_pool(self, *args, **kwargs):
        if not hasattr(self, '_qwen_pre_kv_deepep'):
            initialize(self)
        return original(self, *args, **kwargs)

    model_runner_class.alloc_memory_pool = alloc_memory_pool
    model_runner_class._qwen_pre_kv_deepep_installed = True


def install():
    native = importlib.import_module('sglang.srt.model_executor.model_runner')
    install_on(native.ModelRunner)
