"""Opt-in causal Split-KV Verify for qualified SM80 GQA shapes.

Top-k-one verification is a causal chain. Each query can use the already
qualified Decode kernel with a different visible KV length. Fixed 256-token
splits make each query's reduction independent of the other requests.

This prototype deliberately rebuilds metadata inside each layer invocation.
That cost is included in its qualified timing. It never caches metadata based
only on a Python object's identity: CUDA Graph replays change tensor contents.
"""


def supported(
    *,
    sm,
    deterministic,
    topk,
    draft_tokens,
    is_verify,
    causal,
    q_heads,
    kv_heads,
    head_dim,
    value_dim,
    page_size,
    q_dtype,
    kv_dtype,
    sliding,
    logit_cap,
    sinks,
    score_mod,
    temperature
):
    return (
        sm == (8, 0)
        and deterministic
        and topk == 1
        and draft_tokens == 3
        and is_verify
        and causal
        and (q_heads, kv_heads, head_dim, value_dim) == (24, 2, 128, 128)
        and page_size == 64
        and q_dtype == kv_dtype == "torch.bfloat16"
        and sliding in (None, -1)
        and not logit_cap
        and sinks is None
        and score_mod is None
        and temperature in (None, -1, 0)
    )


_KERNEL = None


def metadata_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def expand(
        PrefixPtr,
        PrefixIDs,
        ExtendIDs,
        Ptr,
        IDs,
        Splits,
        PREFIX_STRIDE: tl.constexpr,
        EXTEND_STRIDE: tl.constexpr,
        Q: tl.constexpr,
        BS: tl.constexpr,
        TILE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        request, query = row // Q, row % Q
        prefix_start = tl.load(PrefixPtr + request)
        prefix_len = tl.load(PrefixPtr + request + 1) - prefix_start
        # Sum lengths of all preceding expanded rows, without a separate scan.
        start = Q * prefix_start + request * (Q * (Q + 1) // 2)
        start += query * (prefix_len + 1) + query * (query - 1) // 2
        length = prefix_len + query + 1
        tl.store(Ptr + row, start)
        if row == BS * Q - 1:
            tl.store(Ptr + row + 1, start + length)
        tl.store(Splits + row, tl.cdiv(length, TILE))
        lane = tl.arange(0, BLOCK)
        for block in range(tl.cdiv(length, BLOCK)):
            offset = block * BLOCK + lane
            old = tl.load(
                PrefixIDs + (prefix_start + offset) * PREFIX_STRIDE,
                mask=offset < prefix_len,
                other=0,
            )
            fresh = tl.load(
                ExtendIDs + (request * Q + offset - prefix_len) * EXTEND_STRIDE,
                mask=(offset >= prefix_len) & (offset < length),
                other=0,
            )
            tl.store(
                IDs + start + offset,
                tl.where(offset < prefix_len, old, fresh),
                mask=offset < length,
            )

    _KERNEL = expand
    return expand


class CausalDecodeState:
    def __init__(self, batch, queries, heads, dim, context_limit, split_tile, device):
        import torch

        self.batch, self.queries = batch, queries
        self.limit, self.tile = context_limit, split_tile
        rows = batch * queries
        self.maximum = (context_limit + split_tile - 1) // split_tile
        self.ptr = torch.empty(rows + 1, device=device, dtype=torch.int32)
        self.ids = torch.empty(rows * context_limit, device=device, dtype=torch.int64)
        self.splits = torch.empty(rows, device=device, dtype=torch.int32)
        self.logits = torch.empty(
            rows, heads, self.maximum, dim, device=device, dtype=torch.float32
        )
        self.lse = torch.empty(
            rows, heads, self.maximum, device=device, dtype=torch.float32
        )

    def run(self, q, out, kb, vb, prefix_ptr, prefix_ids, extend_ids, scale, page_size):
        from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd

        metadata_kernel()[(self.batch * self.queries,)](
            prefix_ptr,
            prefix_ids,
            extend_ids,
            self.ptr,
            self.ids,
            self.splits,
            PREFIX_STRIDE=prefix_ids.stride(0),
            EXTEND_STRIDE=extend_ids.stride(0),
            Q=self.queries,
            BS=self.batch,
            TILE=self.tile,
            BLOCK=1024,
            num_warps=4,
        )
        decode_attention_fwd(
            q,
            kb,
            vb,
            out,
            self.ptr,
            self.ids,
            self.logits,
            self.lse,
            self.splits,
            self.maximum,
            scale,
            1.0,
            1.0,
            page_size=page_size,
        )
        return out


def try_verify_as_decode(
    backend, q, out, layer, batch, causal, logit_cap, sinks, score_mod
):
    """Return None for unsupported cases so the original path remains intact."""
    if not batch.forward_mode.is_target_verify():
        return None
    import torch

    kb = backend.token_to_kv_pool.get_key_buffer(layer.layer_id)
    vb = backend.token_to_kv_pool.get_value_buffer(layer.layer_id)
    queries = getattr(batch.spec_info, "draft_token_num", None)
    if (
        layer.k_scale is not None
        or layer.v_scale is not None
        or kb.dtype != vb.dtype
        or backend.split_tile_size != 256
    ):
        return None
    if not supported(
        sm=tuple(torch.cuda.get_device_capability(q.device)),
        deterministic=backend.enable_deterministic,
        topk=backend.topk,
        draft_tokens=queries,
        is_verify=True,
        causal=causal,
        q_heads=layer.tp_q_head_num,
        kv_heads=kb.shape[-2],
        head_dim=layer.qk_head_dim,
        value_dim=layer.v_head_dim,
        page_size=backend.page_size,
        q_dtype=str(q.dtype),
        kv_dtype=str(kb.dtype),
        sliding=layer.sliding_window_size,
        logit_cap=logit_cap,
        sinks=sinks,
        score_mod=score_mod,
        temperature=layer.xai_temperature_len,
    ):
        return None
    bs = batch.batch_size
    if q.numel() != bs * queries * layer.tp_q_head_num * layer.qk_head_dim:
        return None
    metadata = backend.forward_metadata
    physical = metadata.out_cache_loc_full_physical
    if physical is None:
        physical = batch.out_cache_loc
    # The write above this method has already translated unified-pool slots.
    # Reuse those physical IDs, never reinterpret the virtual request IDs.
    key = (bs, queries, backend.max_context_len, backend.split_tile_size)
    states = getattr(backend, "_sm80_causal_verify_states", None)
    if states is None:
        states = backend._sm80_causal_verify_states = {}
    state = states.get(key)
    if state is None:
        state = CausalDecodeState(
            bs, queries, 24, 128, backend.max_context_len, 256, q.device
        )
        states[key] = state
    state.run(
        q.view(-1, 24, 128),
        out.view(-1, 24, 128),
        kb,
        vb,
        metadata.kv_indptr,
        metadata.kv_indices,
        physical,
        layer.scaling,
        backend.page_size,
    )
    return out
