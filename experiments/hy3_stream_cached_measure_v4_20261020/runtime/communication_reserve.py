"""Allocate the actual reusable DeepEP buffer before native KV sizing.

Only startup order changes. No model forward, synthetic computation, capacity
override, or alternative transport is introduced.
"""
import os
from pathlib import Path
import time
from stream_common import config, save


def reserve(runner):
    candidates = []
    for module in runner.model.modules():
        dispatcher = getattr(module, 'dispatcher', None)
        inners = getattr(dispatcher, '_inners', [dispatcher])
        for inner in inners:
            low = getattr(inner, '_low_latency_dispatcher', None)
            if low is not None:
                candidates.append(low)
    if not candidates:
        raise RuntimeError('Full Hy3 DeepEP dispatcher missing before KV allocation')
    first = candidates[0]
    shape = (first.hidden_size, first.params_bytes,
             first.num_max_dispatch_tokens_per_rank, first.num_experts)
    if shape != (4096, 2, 128, 192):
        raise RuntimeError('Unexpected admitted Hy3 communication buffer shape')
    if any((x.hidden_size, x.params_bytes,
            x.num_max_dispatch_tokens_per_rank, x.num_experts) != shape
           for x in candidates):
        raise RuntimeError('Multiple incompatible layer communication shapes')
    return first._get_buffer()


def install():
    from sglang.srt.model_executor.model_runner import ModelRunner
    if getattr(ModelRunner, '_hy3_pre_kv_comm', False):
        return
    original = ModelRunner.alloc_memory_pool

    def allocate(self, *args, **kwargs):
        import torch
        from sglang.srt.runtime_context import get_parallel
        before = torch.cuda.mem_get_info()[0]
        buffer = reserve(self)
        torch.cuda.synchronize()
        after = torch.cuda.mem_get_info()[0]
        run = Path(os.environ['HY3_RUN_DIR'])
        cfg = config(run)
        save(run / cfg['layout'] / 'capacity' /
             f'communication_rank{get_parallel().world_rank}.json',
             dict(timestamp=time.time(), ready=True, free_before_bytes=before,
                  free_after_bytes=after, consumed_bytes=before-after,
                  rdma_buffer_bytes=buffer.num_rdma_bytes,
                  max_source_tokens_per_rank=128,
                  initialized_before_native_kv_sizing=True))
        return original(self, *args, **kwargs)

    ModelRunner.alloc_memory_pool = allocate
    ModelRunner._hy3_pre_kv_comm = True
