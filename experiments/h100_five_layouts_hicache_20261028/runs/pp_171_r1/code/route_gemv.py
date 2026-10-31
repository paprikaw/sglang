"""Experimental route-indexed BF16 GEMV, avoiding expert-row padding.

For tiny PP microbatches only. Weights retain their original halves layout;
no expert pruning, quantization, or duplicated weight bank is used.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def up(X, W, IDS, A, H:tl.constexpr, I:tl.constexpr, TOP:tl.constexpr, B:tl.constexpr):
    route, block = tl.program_id(0), tl.program_id(1)
    token = route // TOP
    expert = tl.load(IDS + route).to(tl.int64)
    cols = block * B + tl.arange(0, B)
    kh = tl.arange(0, H)
    x = tl.load(X + token * H + kh).to(tl.float32)
    gate = tl.load(W + expert * (2 * I * H) + cols[:,None] * H + kh[None,:],
                   (expert >= 0) & (cols[:,None] < I), 0).to(tl.float32)
    value = tl.load(W + expert * (2 * I * H) + (cols[:,None] + I) * H + kh[None,:],
                    (expert >= 0) & (cols[:,None] < I), 0).to(tl.float32)
    g = tl.sum(gate * x[None,:], 1).to(tl.bfloat16).to(tl.float32)
    u = tl.sum(value * x[None,:], 1).to(tl.bfloat16).to(tl.float32)
    y = g / (1. + tl.exp(-g)) * u
    tl.store(A + route * I + cols, y, cols < I)


@tr.jit
def down(A, W, IDS, Y, H:tl.constexpr, I:tl.constexpr, B:tl.constexpr):
    route, block = tl.program_id(0), tl.program_id(1)
    expert = tl.load(IDS + route).to(tl.int64)
    cols = block * B + tl.arange(0, B)
    ki = tl.arange(0, I)
    x = tl.load(A + route * I + ki).to(tl.float32)
    weight = tl.load(W + expert * H * I + cols[:,None] * I + ki[None,:],
                     (expert >= 0) & (cols[:,None] < H), 0).to(tl.float32)
    y = tl.sum(weight * x[None,:], 1)
    tl.store(Y + route * H + cols, y, cols < H)


@tr.jit
def combine(Y, WEIGHT, O, H:tl.constexpr, TOP:tl.constexpr, B:tl.constexpr):
    token, block = tl.program_id(0), tl.program_id(1)
    kk = tl.arange(0, TOP)
    hh = block * B + tl.arange(0, B)
    y = tl.load(Y + (token * TOP + kk[:,None]) * H + hh[None,:], hh[None,:] < H, 0).to(tl.float32)
    weight = tl.load(WEIGHT + token * TOP + kk).to(tl.float32)
    # Standard BF16 MoE rounds each weighted route before its top-k reduction.
    weighted = (y * weight[:,None]).to(tl.bfloat16).to(tl.float32)
    tl.store(O + token * H + hh, tl.sum(weighted, 0), hh < H)


def run(dispatch, w13, w2, width=8):
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    x = dispatch.hidden_states
    topw, topids, _ = dispatch.topk_output
    tokens, hidden = x.shape
    experts, two_intermediate, _ = w13.shape
    intermediate = two_intermediate // 2
    top = topids.shape[1]
    assert top & (top - 1) == 0
    assert hidden & (hidden - 1) == 0 and intermediate & (intermediate - 1) == 0
    act = torch.empty((tokens * top, intermediate), device=x.device, dtype=torch.bfloat16)
    output_routes = torch.empty((tokens * top, hidden), device=x.device, dtype=torch.bfloat16)
    result = torch.empty_like(x)
    up[(tokens * top, tr.cdiv(intermediate, width))](x, w13, topids, act, hidden, intermediate, top, width, num_warps=4)
    down[(tokens * top, tr.cdiv(hidden, width * 2))](act, w2, topids, output_routes, hidden, intermediate, width * 2, num_warps=4)
    combine[(tokens, tr.cdiv(hidden, 128))](output_routes, topw, result, hidden, top, 128, num_warps=4)
    return StandardCombineInput(result)
