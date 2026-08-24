"""KV capacity calculations backed by exact elastic-PP weight profiles."""

from __future__ import annotations

import math
from dataclasses import dataclass

from sglang.srt.elastic_pp.weight_profile import ElasticPPWeightProfile


@dataclass(frozen=True)
class ElasticPPCapacityEstimate:
    """Auditable memory terms for one candidate PP layer range."""

    start_layer: int
    end_layer: int
    usable_memory_bytes: int
    target_weight_bytes: int
    runtime_overhead_bytes: int
    weight_load_workspace_bytes: int
    kv_budget_bytes: int
    kv_bytes_per_block: int
    raw_num_blocks: int
    aligned_num_blocks: int


def estimate_kv_capacity(
    *,
    profile: ElasticPPWeightProfile,
    start_layer: int,
    end_layer: int,
    total_device_memory_bytes: int,
    device_memory_utilization: float,
    kv_block_bytes_by_layer: tuple[int, ...] | list[int],
    runtime_overhead_bytes: int = 0,
    weight_load_workspace_bytes: int = 0,
    block_multiple: int = 1,
) -> ElasticPPCapacityEstimate:
    """Estimate equal per-layer KV blocks for a half-open layer range.

    One logical block consumes the sum of the selected layers' physical KV
    page sizes.  Unlike average-layer formulas, both weights and KV page sizes
    may vary by layer.
    """

    if total_device_memory_bytes <= 0:
        raise ValueError("total_device_memory_bytes must be positive")
    if not math.isfinite(device_memory_utilization) or not (
        0 < device_memory_utilization <= 1
    ):
        raise ValueError("device_memory_utilization must be in (0, 1]")
    if runtime_overhead_bytes < 0 or weight_load_workspace_bytes < 0:
        raise ValueError("memory overhead terms must be non-negative")
    if block_multiple <= 0:
        raise ValueError("block_multiple must be positive")
    if start_layer == end_layer:
        raise ValueError("KV capacity requires a non-empty layer range")

    block_sizes = tuple(int(size) for size in kv_block_bytes_by_layer)
    if len(block_sizes) != len(profile.layer_weight_sizes):
        raise ValueError(
            "KV block profile must contain one entry per model layer: "
            f"got={len(block_sizes)}, expected={len(profile.layer_weight_sizes)}"
        )
    if any(size <= 0 for size in block_sizes):
        raise ValueError("every per-layer KV block size must be positive")

    target_weight_bytes = profile.target_weight_size(start_layer, end_layer)
    usable_memory_bytes = int(total_device_memory_bytes * device_memory_utilization)
    kv_budget_bytes = (
        usable_memory_bytes
        - target_weight_bytes
        - runtime_overhead_bytes
        - weight_load_workspace_bytes
    )
    kv_bytes_per_block = sum(block_sizes[start_layer:end_layer])
    raw_num_blocks = max(0, kv_budget_bytes // kv_bytes_per_block)
    aligned_num_blocks = raw_num_blocks - raw_num_blocks % block_multiple
    return ElasticPPCapacityEstimate(
        start_layer=start_layer,
        end_layer=end_layer,
        usable_memory_bytes=usable_memory_bytes,
        target_weight_bytes=target_weight_bytes,
        runtime_overhead_bytes=runtime_overhead_bytes,
        weight_load_workspace_bytes=weight_load_workspace_bytes,
        kv_budget_bytes=kv_budget_bytes,
        kv_bytes_per_block=kv_bytes_per_block,
        raw_num_blocks=raw_num_blocks,
        aligned_num_blocks=aligned_num_blocks,
    )
