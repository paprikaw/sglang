"""Ownership of GDN state when Decode reuses KV received from Prefill."""
from __future__ import annotations

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.runtime_context import mamba_checkpoint_grid


def reserve_decode_mamba_slots(tree_cache, req) -> bool:
    """Reclaim cached states before allocating a PD request and its checkpoint.

    Request-row availability alone does not imply recurrent-state availability:
    radix checkpoints share the state allocator with live and preallocated reqs.
    Keep one slot for publishing the received state without consuming its owner.
    """
    if not tree_cache.supports_mamba():
        return True
    pool = tree_cache.req_to_token_pool
    needed = int(req.mamba_pool_idx is None)
    if pool.enable_mamba_extra_buffer and req.mamba_ping_pong_track_buffer is None:
        needed += (
            1
            if pool.enable_mamba_extra_buffer_lazy
            else pool.mamba_ping_pong_track_buffer_size
        )
    if not tree_cache.disable:
        needed += 1
    available = pool.mamba_allocator.available_size()
    if available < needed:
        tree_cache.evict(EvictParams(num_tokens=0, mamba_num=needed - available))
    return pool.mamba_allocator.available_size() >= needed


def prepare_transferred_mamba_state(req, tree_cache) -> None:
    """Publish a real prompt-boundary checkpoint after successful state transfer.

    PREBUILT does not execute Prefill, so its normal state tracking never runs.
    The transferred main slot represents exactly kv_committed_len, not an earlier
    rounded-down boundary. Snapshot it only when that depth is cache-aligned.
    The main slot remains request-owned and is advanced by subsequent Decode.
    """
    if not tree_cache.supports_mamba():
        return
    req.mamba_cow_src_index = None
    req.mamba_needs_clear = False
    req.mamba_branching_seqlen = None
    req.mamba_last_track_seqlen = None
    if tree_cache.disable or not tree_cache.enable_mamba_extra_buffer:
        return
    depth = req.kv_committed_len
    grid = mamba_checkpoint_grid(tree_cache.page_size)
    if depth <= 0 or depth % grid:
        return
    pool = tree_cache.req_to_token_pool
    track_idx = req.mamba_next_track_idx
    src = req.mamba_pool_idx.reshape(1)
    dst = req.mamba_ping_pong_track_buffer[track_idx].reshape(1)
    translate = pool.translate_mamba_indices
    pool.mamba_pool.copy_from(translate(src), translate(dst))
    req.mamba_last_track_idx = track_idx
    if not pool.enable_mamba_extra_buffer_lazy:
        req.mamba_next_track_idx = pool.get_mamba_ping_pong_other_idx(track_idx)
    req.mamba_last_track_seqlen = depth
