# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Topology helpers for pipeline stages with different TP widths.

The first supported layouts are the ones needed by the elastic-PP experiments::

    stage 0: global ranks [0, 1, 2, 3] (TP4)
    stage 1: global ranks [4] or [4, 5] (TP1 or TP2)

Only stage leaders participate in the PP transport group.  Other ranks remain
members of their stage-local TP group and use singleton physical PP groups so
every distributed rank constructs process groups in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import Iterable, Sequence, Tuple


def _parse_positive_ints(raw: str, *, flag: str) -> Tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(","))
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            f"{flag} must contain comma-separated integers: {raw!r}"
        ) from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{flag} values must all be positive: {raw!r}")
    return values


@dataclass(frozen=True, slots=True)
class HeterogeneousPPPlacement:
    pp_rank: int
    tp_rank: int
    tp_size: int
    global_rank: int
    local_gpu_index: int


@dataclass(frozen=True, slots=True)
class HeterogeneousPPTopology:
    """A contiguous global-rank layout for non-uniform PP stage widths."""

    stage_tp_sizes: Tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.stage_tp_sizes) < 2:
            raise ValueError("heterogeneous PP requires at least two stages")
        if any(size <= 0 for size in self.stage_tp_sizes):
            raise ValueError(
                f"heterogeneous PP stage TP sizes must be positive: {self.stage_tp_sizes}"
            )

    @classmethod
    def parse(cls, raw: str) -> "HeterogeneousPPTopology":
        return cls(_parse_positive_ints(raw, flag="--pp-stage-tp-sizes"))

    @property
    def pp_size(self) -> int:
        return len(self.stage_tp_sizes)

    @property
    def world_size(self) -> int:
        return sum(self.stage_tp_sizes)

    @property
    def stage_offsets(self) -> Tuple[int, ...]:
        return (0, *tuple(accumulate(self.stage_tp_sizes[:-1])))

    @property
    def stage_leader_ranks(self) -> Tuple[int, ...]:
        return self.stage_offsets

    @property
    def tp_groups(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(
            tuple(range(offset, offset + size))
            for offset, size in zip(self.stage_offsets, self.stage_tp_sizes)
        )

    @property
    def pp_transport_groups(self) -> Tuple[Tuple[int, ...], ...]:
        """Return one leader PP group plus singleton groups for non-leaders."""

        leaders = self.stage_leader_ranks
        leader_set = set(leaders)
        singletons = tuple(
            (rank,) for rank in range(self.world_size) if rank not in leader_set
        )
        return (leaders, *singletons)

    @property
    def singleton_groups(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple((rank,) for rank in range(self.world_size))

    def global_rank(self, pp_rank: int, tp_rank: int) -> int:
        if pp_rank < 0 or pp_rank >= self.pp_size:
            raise ValueError(f"invalid PP rank {pp_rank} for {self.pp_size} stages")
        tp_size = self.stage_tp_sizes[pp_rank]
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(
                f"invalid TP rank {tp_rank} for stage {pp_rank} with TP size {tp_size}"
            )
        return self.stage_offsets[pp_rank] + tp_rank

    def stage_for_global_rank(self, global_rank: int) -> Tuple[int, int]:
        if global_rank < 0 or global_rank >= self.world_size:
            raise ValueError(
                f"invalid global rank {global_rank} for world size {self.world_size}"
            )
        for pp_rank, (offset, size) in enumerate(
            zip(self.stage_offsets, self.stage_tp_sizes)
        ):
            if global_rank < offset + size:
                return pp_rank, global_rank - offset
        raise AssertionError("validated global rank was not assigned to a stage")

    def placements_for_node(
        self, *, nnodes: int, node_rank: int
    ) -> Tuple[HeterogeneousPPPlacement, ...]:
        """Map stages to local GPUs for the initially supported placements.

        A single node hosts every stage on consecutive GPUs.  In a multi-node
        launch, one complete stage is placed on each node.  Splitting one TP
        group across nodes is deliberately excluded from the first version.
        """

        if nnodes == 1:
            stage_ranks: Iterable[int] = range(self.pp_size)
        elif nnodes == self.pp_size:
            if node_rank < 0 or node_rank >= nnodes:
                raise ValueError(f"invalid node rank {node_rank} for {nnodes} nodes")
            stage_ranks = (node_rank,)
        else:
            raise ValueError(
                "--pp-stage-tp-sizes currently supports either one node for all "
                f"stages or one complete stage per node; got nnodes={nnodes}, "
                f"pp_size={self.pp_size}"
            )

        placements = []
        next_local_gpu = 0
        for pp_rank in stage_ranks:
            tp_size = self.stage_tp_sizes[pp_rank]
            for tp_rank in range(tp_size):
                local_gpu_index = (
                    self.global_rank(pp_rank, tp_rank)
                    if nnodes == 1
                    else next_local_gpu
                )
                placements.append(
                    HeterogeneousPPPlacement(
                        pp_rank=pp_rank,
                        tp_rank=tp_rank,
                        tp_size=tp_size,
                        global_rank=self.global_rank(pp_rank, tp_rank),
                        local_gpu_index=local_gpu_index,
                    )
                )
                next_local_gpu += 1
        return tuple(placements)

    def previous_stage_leader(self, pp_rank: int) -> int:
        return self.stage_leader_ranks[(pp_rank - 1) % self.pp_size]

    def next_stage_leader(self, pp_rank: int) -> int:
        return self.stage_leader_ranks[(pp_rank + 1) % self.pp_size]


def parse_pp_layer_partition(
    raw: str, *, pp_size: int, num_hidden_layers: int | None = None
) -> Tuple[int, ...]:
    partition = _parse_positive_ints(raw, flag="--pp-layer-partition")
    if len(partition) != pp_size:
        raise ValueError(
            "--pp-layer-partition must contain one layer count per stage: "
            f"got {len(partition)}, expected {pp_size}"
        )
    if num_hidden_layers is not None and sum(partition) != num_hidden_layers:
        raise ValueError(
            "--pp-layer-partition must cover every transformer layer: "
            f"got {sum(partition)}, expected {num_hidden_layers}"
        )
    return partition


def incompatible_attention_tp_sizes(
    stage_tp_sizes: Sequence[int],
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> Tuple[int, ...]:
    """Return TP widths that cannot shard this model's Q/KV heads."""

    invalid = []
    for tp_size in stage_tp_sizes:
        query_heads_divisible = num_attention_heads % tp_size == 0
        kv_heads_divisible = (
            num_key_value_heads % tp_size == 0
            if num_key_value_heads >= tp_size
            else tp_size % num_key_value_heads == 0
        )
        if not query_heads_divisible or not kv_heads_divisible:
            invalid.append(tp_size)
    return tuple(invalid)
