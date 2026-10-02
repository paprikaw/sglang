"""Activation for the fixed-capacity expert buffers used by DeepEP LL.

Only ``expert_num_tokens[e]`` rows contain GEMM output for expert ``e``.
A persistent grid skips the unused rows without copying GPU counts to the CPU
or launching one empty CTA for every padding row. The grid and all pointers
remain constant when counts change during CUDA Graph replay.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _masked_silu_and_mul_kernel(
    inputs,
    outputs,
    expert_num_tokens,
    CAPACITY: tl.constexpr,
    N: tl.constexpr,
    PROGRAMS_PER_EXPERT: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    expert = tl.program_id(0)
    program = tl.program_id(1)
    count = tl.minimum(tl.maximum(tl.load(expert_num_tokens + expert), 0), CAPACITY)
    columns = tl.arange(0, BLOCK_N)
    row_offsets = tl.arange(0, ROWS_PER_PROGRAM)
    first_row = program * ROWS_PER_PROGRAM
    step = PROGRAMS_PER_EXPERT * ROWS_PER_PROGRAM

    for start_row in range(first_row, count, step):
        rows = start_row + row_offsets
        mask = (rows[:, None] < count) & (columns[None, :] < N)
        in_offsets = (expert * CAPACITY + rows[:, None]) * (2 * N) + columns[None, :]
        gate = tl.load(inputs + in_offsets, mask=mask, other=0).to(tl.float32)
        up = tl.load(inputs + in_offsets + N, mask=mask, other=0).to(tl.float32)
        # sgl_kernel.silu_and_mul evaluates both activation and multiplication
        # in float32 before the final output cast.
        result = (gate / (1.0 + tl.exp(-gate))) * up
        out_offsets = (expert * CAPACITY + rows[:, None]) * N + columns[None, :]
        tl.store(outputs + out_offsets, result, mask=mask)


def masked_silu_and_mul(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    *,
    programs_per_expert: int = 16,
    rows_per_program: int = 4,
) -> None:
    """Write valid rows of contiguous ``[E * capacity, 2N]`` input.

    Padding is deliberately untouched; both neighboring grouped masked GEMMs
    use the same GPU counts and never consume those rows. No allocation or
    device-to-host synchronization occurs here.
    """
    assert inputs.ndim == outputs.ndim == 2
    assert inputs.is_contiguous() and outputs.is_contiguous()
    assert inputs.shape[0] == outputs.shape[0]
    assert inputs.shape[1] == outputs.shape[1] * 2
    assert expert_num_tokens.is_contiguous()
    assert expert_num_tokens.dtype in (torch.int32, torch.int64)
    assert inputs.dtype == outputs.dtype
    assert inputs.device == outputs.device == expert_num_tokens.device
    experts = expert_num_tokens.numel()
    assert experts > 0 and inputs.shape[0] % experts == 0
    capacity = inputs.shape[0] // experts
    if capacity == 0:
        return
    assert programs_per_expert > 0
    assert rows_per_program > 0 and rows_per_program & (rows_per_program - 1) == 0
    programs_per_expert = min(
        programs_per_expert, triton.cdiv(capacity, rows_per_program)
    )
    n = outputs.shape[1]
    assert n > 0
    block_n = triton.next_power_of_2(n)
    num_warps = 4 if rows_per_program * block_n <= 4096 else 8
    _masked_silu_and_mul_kernel[(experts, programs_per_expert)](
        inputs,
        outputs,
        expert_num_tokens,
        CAPACITY=capacity,
        N=n,
        PROGRAMS_PER_EXPERT=programs_per_expert,
        ROWS_PER_PROGRAM=rows_per_program,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
