"""Correctness-first BF16 DeepEP-LL layout bridge for the Triton MoE runner.

No quantization or CUDA-specific operations are used here. The static expert
slots retain their original positions for DeepEP combine. Padding is explicitly
zeroed, since native dispatch does not initialize unused receive slots.
"""

import torch


def prepare_bf16_deepep_ll(hidden_states: torch.Tensor, masked_m: torch.Tensor):
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError("Triton DeepEP LL bridge requires BF16 hidden states")
    if hidden_states.ndim != 3:
        raise ValueError(
            "DeepEP LL hidden states must be [local_experts, slots, hidden]"
        )
    experts, slots, hidden = hidden_states.shape
    if masked_m.shape != (experts,) or masked_m.dtype not in (torch.int32, torch.int64):
        raise ValueError("DeepEP LL masked_m must contain one integer count per expert")
    if masked_m.device != hidden_states.device:
        raise ValueError(
            "DeepEP LL hidden states and counts must be on the same device"
        )
    positions = torch.arange(slots, device=hidden_states.device)
    valid = positions.unsqueeze(0) < masked_m.unsqueeze(1)
    expert_ids = torch.arange(experts, dtype=torch.int32, device=hidden_states.device)
    expert_ids = expert_ids.unsqueeze(1).expand(experts, slots)
    expert_ids = torch.where(valid, expert_ids, -1).reshape(-1, 1).contiguous()
    weights = valid.reshape(-1, 1).to(torch.float32)
    flat_hidden = torch.where(valid.unsqueeze(-1), hidden_states, 0)
    return flat_hidden.reshape(experts * slots, hidden), weights, expert_ids, valid
