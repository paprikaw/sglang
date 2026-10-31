"""Use the existing PP Triton GEMMs on DeepEP's BF16 expert-major buffers.

Only the GPU indexing adapter is new. Routing, weights and DeepEP's weighted
combine are unchanged. The padded-row upper bound includes every assignment
allowed by the original DeepEP dispatch capacity, rather than observed routes.
"""
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _metadata(Counts, Sorted, Experts, Padded, E: tl.constexpr, M: tl.constexpr,
              BM: tl.constexpr, LIMIT: tl.constexpr, EB: tl.constexpr,
              MB: tl.constexpr):
    e = tl.arange(0, EB)
    count = tl.load(Counts + e, e < E, 0)
    blocks = tl.cdiv(count, BM)
    end = tl.cumsum(blocks, 0)
    start = (end - blocks) * BM
    r = tl.arange(0, MB)
    dst = start[:, None] + r[None, :]
    used = (e[:, None] < E) & (r[None, :] < blocks[:, None] * BM)
    token = tl.where(r[None, :] < count[:, None], e[:, None] * M + r[None, :], E * M)
    tl.store(Sorted + dst, token, used & (dst < LIMIT))
    tl.store(Experts + dst // BM, tl.broadcast_to(e[:, None], (EB, MB)),
             used & (dst < LIMIT) & (r[None, :] % BM == 0))
    tl.store(Padded, tl.sum(blocks, 0) * BM)


def tile_config(w13, w2, local_tokens, topk):
    # Resolve the same shape-aware Triton policy as PP, using logical input
    # tokens rather than the capacity of DeepEP's sparse expert-major buffer.
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import try_get_optimal_moe_config
    up, (down, _) = try_get_optimal_moe_config(
        w13.shape, w2.shape, topk, None, local_tokens,
        block_shape=None, per_channel_quant=False, return_down_config=True)
    up, down = dict(up), dict(down) if down is not None else None
    if up.pop('USE_TMA', False) or (down is not None and down.pop('USE_TMA', False)):
        raise RuntimeError('This comparison requires the existing non-TMA small-batch Triton kernel')
    if down is not None and down['BLOCK_SIZE_M'] != up['BLOCK_SIZE_M']:
        raise RuntimeError('Up/down must share a routed-row index block size')
    return up, down


def run(dispatch, w13, w2, *, return_metadata=False):
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel
    from sglang.kernels.ops.moe.ep_moe_kernels import silu_and_mul_masked_fwd
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.distributed.device_communicators.pynccl_allocator import use_symmetric_memory

    x = dispatch.hidden_states
    assert x.dtype == w13.dtype == w2.dtype == torch.bfloat16
    assert dispatch.hidden_states_scale is None and x.is_contiguous()
    e, m, k = x.shape
    local_tokens, topk = dispatch.topk_ids.shape
    up_config, down_config = tile_config(w13, w2, local_tokens, topk)
    bm = up_config['BLOCK_SIZE_M']
    # Each rank can dispatch at most C unique tokens with topk assignments.
    # Thus sum(masked_m) <= C * world * topk, including skewed routing.
    capacity = int(os.environ['SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK'])
    world = get_tp_group().world_size
    assignments_bound = capacity * world * topk
    full_padded = e * triton.cdiv(m, bm) * bm
    padded_bound = min(full_padded, triton.cdiv(assignments_bound + e * (bm - 1), bm) * bm)
    sorted_ids = torch.empty((padded_bound,), device=x.device, dtype=torch.int32)
    expert_ids = torch.empty((padded_bound // bm,), device=x.device, dtype=torch.int32)
    padded = torch.empty((), device=x.device, dtype=torch.int32)
    _metadata[(1,)](dispatch.masked_m, sorted_ids, expert_ids, padded,
        e, m, bm, padded_bound, triton.next_power_of_2(e),
        triton.next_power_of_2(triton.cdiv(m, bm) * bm), num_warps=8)

    flat = x.view(e * m, k)
    # The shared GEMM uses only the number of IDs when router multiplication
    # is disabled. DeepEP combine applies the original top-k weights later.
    dummy_ids = torch.empty((e * m, 1), device=x.device, dtype=torch.int32)
    dummy_weights = torch.empty((e * m, 1), device=x.device, dtype=torch.float32)
    gateup = torch.empty((e, m, w13.shape[1]), device=x.device, dtype=x.dtype)
    activated = torch.empty((e, m, w13.shape[1] // 2), device=x.device, dtype=x.dtype)
    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        output = torch.empty_like(x)

    def gemm(a, w, c, config):
        invoke_fused_moe_kernel(a, w, None, c, None, None, None,
            dummy_weights, dummy_ids, sorted_ids, expert_ids, padded,
            False, 1, config, compute_type=tl.bfloat16,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False, filter_expert=False)

    gemm(flat, w13, gateup.view(e * m, -1), up_config)
    silu_and_mul_masked_fwd(gateup, activated, dispatch.masked_m)
    gemm(activated.view(e * m, -1), w2, output.view(e * m, 1, k), down_config or up_config)
    if return_metadata:
        return output, dict(sorted_ids=sorted_ids, expert_ids=expert_ids, padded=padded,
                            block_m=bm, padded_bound=padded_bound,
                            assignments_bound=assignments_bound,
                            up_config=up_config, down_config=down_config)
    return output
