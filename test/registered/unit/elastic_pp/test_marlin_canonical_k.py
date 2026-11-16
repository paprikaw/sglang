"""CPU dispatch checks; native arithmetic is qualified separately on A100."""

import importlib
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("canonical", [False, True])
def test_canonical_kernel_has_a_distinct_specialization(monkeypatch, canonical):
    module = importlib.import_module("sglang.kernels.ops.moe.moe_wna16_marlin")
    selections, calls = [], []

    def select(*args):
        selections.append(args)
        return SimpleNamespace(moe_wna16_marlin_gemm=lambda *args: calls.append(args))

    monkeypatch.setattr(module, "_jit_moe_wna16_marlin_module", select)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=108),
    )
    output = module.moe_wna16_marlin_gemm(
        a=torch.ones((2, 16), dtype=torch.bfloat16),
        c_or_none=None,
        b_q_weight=torch.empty((4, 1, 64), dtype=torch.int32),
        b_bias_or_none=None,
        b_scales=torch.ones((4, 1, 16), dtype=torch.bfloat16),
        global_scale_or_none=None,
        b_zeros_or_none=None,
        g_idx_or_none=None,
        perm_or_none=None,
        workspace=torch.zeros(16, dtype=torch.int32),
        sorted_token_ids=torch.arange(16, dtype=torch.int32),
        expert_ids=torch.zeros(2, dtype=torch.int32),
        num_tokens_post_padded=torch.tensor([16], dtype=torch.int32),
        topk_weights=torch.ones((2, 2)),
        moe_block_size=8,
        top_k=2,
        mul_topk_weights=False,
        is_ep=False,
        b_q_type=SimpleNamespace(id=1),
        size_m=2,
        size_n=16,
        size_k=16,
        use_fp32_reduce=True,
        canonical_k=canonical,
    )
    assert selections == [(torch.bfloat16, False, False, canonical)]
    assert len(calls) == 1 and calls[0][1] is output
    assert output.shape == (4, 16)


def test_canonical_rejects_atomic_reduction_before_native_launch():
    module = importlib.import_module("sglang.kernels.ops.moe.moe_wna16_marlin")
    with pytest.raises(ValueError, match="forbid atomic"):
        module.moe_wna16_marlin_gemm(
            a=torch.ones((1, 16)),
            c_or_none=None,
            b_q_weight=None,
            b_bias_or_none=None,
            b_scales=None,
            global_scale_or_none=None,
            b_zeros_or_none=None,
            g_idx_or_none=None,
            perm_or_none=None,
            workspace=None,
            sorted_token_ids=None,
            expert_ids=None,
            num_tokens_post_padded=None,
            topk_weights=None,
            moe_block_size=8,
            top_k=1,
            mul_topk_weights=False,
            is_ep=False,
            b_q_type=None,
            size_m=1,
            size_n=16,
            size_k=16,
            use_atomic_add=True,
            canonical_k=True,
        )
