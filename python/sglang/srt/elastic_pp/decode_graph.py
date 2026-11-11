"""Decode Graph lifecycle at an already drained elastic-PP boundary.

Native KV/GDN pools change addresses on commit. Reusing a source graph is
therefore unsafe even when a previously visited layer partition returns.
Capture uses SGLang's reserved request/token/state slot zero, never a live
request, and runs only after every rank has installed the target generation.
"""

from __future__ import annotations

import time


class ElasticPPDecodeGraphState:
    def __init__(self):
        self.history = []

    def begin(self, generation):
        if self.history and generation <= self.history[-1]["generation"]:
            raise RuntimeError("elastic Decode Graph generation must increase")
        entry = dict(
            generation=generation,
            status="building",
            capture_seconds=None,
            replays=0,
            eager_decode=0,
        )
        self.history.append(entry)
        return entry

    def invalidate(self):
        if self.history:
            self.history[-1]["status"] = "retired"

    def observe(self, graph):
        if not self.history:
            raise RuntimeError("elastic Decode Graph lifecycle is uninitialized")
        entry = self.history[-1]
        if graph and entry["status"] != "ready":
            raise RuntimeError("cannot replay a retired/unready elastic Decode Graph")
        entry["replays" if graph else "eager_decode"] += 1

    def snapshot(self):
        return [dict(entry) for entry in self.history]


def retire_decode_graph(model_runner):
    """Retire all captures at the drained boundary, including their pool ID.

    A process-wide pool handle is not an owning reference in PyTorch. Once
    the last graph releases a pool, sharing its cached ID can hit the
    allocator's use_count assertion even if some output tensors remain live.
    Elastic PP gates prefill/speculative/compile graphs, so no other capture
    in this process is allowed to keep using the shared pool.
    """
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        set_graph_pool_id,
    )
    from sglang.srt.model_executor.runner_utils.pool import (
        graph_pool_capture_scope,
        set_global_graph_memory_pool,
    )

    # The scope drains any borrow arena before its backing graph is retired.
    with graph_pool_capture_scope():
        for name in ("decode_cuda_graph_runner", "prefill_cuda_graph_runner"):
            runner = getattr(model_runner, name, None)
            backend = getattr(runner, "backend", None)
            if backend is not None:
                backend.cleanup()
            setattr(model_runner, name, None)
        set_global_graph_memory_pool(None)
        set_graph_pool_id(None)
    state = getattr(model_runner, "_elastic_pp_decode_graph_state", None)
    if state is not None:
        state.invalidate()


def rebuild_decode_graph(model_runner, *, generation, active):
    """Called on the scheduler thread, after the global commit consensus.

    Capture failure is fatal to this transition. It must never be hidden as
    an eager success or followed by resuming an already retired generation.
    CPU spill, speculative, compile and prefill Graph combinations stay gated
    separately by ServerArgs until their state and workspace are qualified.
    """
    state = model_runner._elastic_pp_decode_graph_state
    entry = state.begin(generation)
    model_runner.decode_cuda_graph_runner = None
    model_runner.prefill_cuda_graph_runner = None
    if model_runner.server_args.disable_cuda_graph:
        entry.update(status="disabled", capture_seconds=0.0)
        return
    if not active:
        entry.update(status="inactive", capture_seconds=0.0)
        return

    started = time.monotonic()
    try:
        # Index 0 is reserved by ReqToTokenPool. Capture's dummy batch resets
        # req_pool_indices/out_cache_loc to zero (DecodeCudaGraphRunner.capture).
        # Fail closed if an unsupported pool violates that contract.
        pool = model_runner.req_to_token_pool
        if bool(pool.req_to_token[0].count_nonzero().item()):
            raise RuntimeError("Decode Graph capture requires reserved request row 0")
        get_mamba = getattr(pool, "get_mamba_indices", None)
        if get_mamba is not None:
            if bool(get_mamba(pool.req_to_token.new_zeros(1)).count_nonzero().item()):
                raise RuntimeError("Decode Graph capture requires reserved GDN slot 0")
        model_runner.init_decode_cuda_graph()
        if model_runner.decode_cuda_graph_runner is None:
            raise RuntimeError("elastic PP Decode Graph capture produced no runner")
        entry["status"] = "ready"
    except BaseException:
        entry["status"] = "failed"
        model_runner.decode_cuda_graph_runner = None
        raise
    finally:
        entry["capture_seconds"] = time.monotonic() - started
