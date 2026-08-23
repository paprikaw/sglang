"""Generation-stamped topology state for elastic pipeline parallelism.

The maximum resident PP group is process-static.  A :class:`PPTopology`
describes the ordered subset of those ranks that owns model layers for one
logical generation.  Reconfiguration is deliberately split into prepare,
fence, commit, and cleanup phases so that model/KV preparation can overlap
serving while the route change itself remains an atomic scheduler cut.

Layer ranges use SGLang's half-open ``[start, end)`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True, order=True)
class LayerRange:
    """A half-open model-layer interval."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"layer range start must be non-negative: {self}")
        if self.end < self.start:
            raise ValueError(f"layer range must be half-open and ordered: {self}")

    @property
    def empty(self) -> bool:
        return self.start == self.end

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PPTopology:
    """One immutable logical PP placement over a fixed resident rank set."""

    generation: int
    max_pp_size: int
    active_ranks: tuple[int, ...]
    layer_ranges: tuple[LayerRange, ...]

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("topology generation must be non-negative")
        if self.max_pp_size <= 0:
            raise ValueError("max_pp_size must be positive")
        if len(self.layer_ranges) != self.max_pp_size:
            raise ValueError(
                "layer_ranges must contain one entry per resident rank: "
                f"got {len(self.layer_ranges)} for max_pp_size={self.max_pp_size}"
            )
        if not self.active_ranks:
            raise ValueError("a topology must have at least one active PP rank")
        if len(set(self.active_ranks)) != len(self.active_ranks):
            raise ValueError("active PP ranks must be unique")
        if any(rank < 0 or rank >= self.max_pp_size for rank in self.active_ranks):
            raise ValueError(
                f"active ranks must be in [0, {self.max_pp_size}): "
                f"{self.active_ranks}"
            )

        nonempty_ranks = {
            rank
            for rank, layer_range in enumerate(self.layer_ranges)
            if not layer_range.empty
        }
        if nonempty_ranks != set(self.active_ranks):
            raise ValueError(
                "active_ranks must exactly match ranks with non-empty layer "
                f"ranges: active={self.active_ranks}, nonempty={sorted(nonempty_ranks)}"
            )

        cursor = 0
        for rank in self.active_ranks:
            layer_range = self.layer_ranges[rank]
            if layer_range.start != cursor:
                raise ValueError(
                    "active ranks must own a contiguous, non-overlapping layer "
                    f"partition in pipeline order; rank {rank} starts at "
                    f"{layer_range.start}, expected {cursor}"
                )
            cursor = layer_range.end

    @classmethod
    def from_layer_counts(
        cls,
        layer_counts: tuple[int, ...] | list[int],
        *,
        generation: int = 0,
    ) -> PPTopology:
        """Build a topology from per-resident-rank layer counts.

        Zero-layer ranks stay resident in the maximal process group but do not
        participate in the logical PP route.
        """

        counts = tuple(int(count) for count in layer_counts)
        if not counts:
            raise ValueError("layer_counts must not be empty")
        if any(count < 0 for count in counts):
            raise ValueError(f"layer counts must be non-negative: {counts}")

        cursor = 0
        ranges: list[LayerRange] = []
        active_ranks: list[int] = []
        for rank, count in enumerate(counts):
            ranges.append(LayerRange(cursor, cursor + count))
            if count:
                active_ranks.append(rank)
                cursor += count
        return cls(
            generation=generation,
            max_pp_size=len(counts),
            active_ranks=tuple(active_ranks),
            layer_ranges=tuple(ranges),
        )

    @property
    def num_layers(self) -> int:
        return self.layer_ranges[self.active_ranks[-1]].end

    @property
    def first_rank(self) -> int:
        return self.active_ranks[0]

    @property
    def last_rank(self) -> int:
        return self.active_ranks[-1]

    def layer_range(self, rank: int) -> LayerRange:
        return self.layer_ranges[rank]

    def prev_rank(self, rank: int) -> int | None:
        index = self.active_ranks.index(rank)
        return None if index == 0 else self.active_ranks[index - 1]

    def next_rank(self, rank: int) -> int | None:
        index = self.active_ranks.index(rank)
        return (
            None
            if index == len(self.active_ranks) - 1
            else self.active_ranks[index + 1]
        )

    def owner(self, layer_id: int) -> int:
        if layer_id < 0 or layer_id >= self.num_layers:
            raise IndexError(f"layer {layer_id} is outside [0, {self.num_layers})")
        for rank in self.active_ranks:
            layer_range = self.layer_ranges[rank]
            if layer_range.start <= layer_id < layer_range.end:
                return rank
        raise RuntimeError(f"topology has no owner for layer {layer_id}")


@dataclass(frozen=True, order=True)
class LayerTransfer:
    """A contiguous layer range whose KV ownership moves between ranks."""

    src_rank: int
    dst_rank: int
    layer_range: LayerRange

    def __post_init__(self) -> None:
        if self.src_rank == self.dst_rank:
            raise ValueError("a layer transfer must change ownership")
        if self.layer_range.empty:
            raise ValueError("a layer transfer cannot be empty")


def _range_difference(
    target: LayerRange, current: LayerRange
) -> tuple[LayerRange, ...]:
    """Return the at-most-two pieces in ``target - current``."""

    pieces: list[LayerRange] = []
    if target.start < current.start:
        end = min(target.end, current.start)
        if target.start < end:
            pieces.append(LayerRange(target.start, end))
    if target.end > current.end:
        start = max(target.start, current.end)
        if start < target.end:
            pieces.append(LayerRange(start, target.end))
    return tuple(pieces)


@dataclass(frozen=True)
class DynamicPPTransition:
    """The deterministic preparation and KV-copy plan for one route change."""

    source: PPTopology
    target: PPTopology
    kv_transfers: tuple[LayerTransfer, ...]
    weight_loads: tuple[tuple[int, LayerRange], ...]

    @classmethod
    def build(cls, source: PPTopology, target: PPTopology) -> DynamicPPTransition:
        if source.max_pp_size != target.max_pp_size:
            raise ValueError("a transition cannot resize the resident PP group")
        if source.num_layers != target.num_layers:
            raise ValueError(
                "source and target must cover the same model layers: "
                f"{source.num_layers} != {target.num_layers}"
            )
        if target.generation <= source.generation:
            raise ValueError(
                "target generation must be newer than source generation: "
                f"{target.generation} <= {source.generation}"
            )

        transfers: list[LayerTransfer] = []
        segment_start = 0
        segment_src = source.owner(0)
        segment_dst = target.owner(0)
        for layer_id in range(1, source.num_layers + 1):
            at_end = layer_id == source.num_layers
            src = None if at_end else source.owner(layer_id)
            dst = None if at_end else target.owner(layer_id)
            if at_end or src != segment_src or dst != segment_dst:
                if segment_src != segment_dst:
                    transfers.append(
                        LayerTransfer(
                            segment_src,
                            segment_dst,
                            LayerRange(segment_start, layer_id),
                        )
                    )
                segment_start = layer_id
                segment_src = src
                segment_dst = dst

        weight_loads: list[tuple[int, LayerRange]] = []
        for rank in target.active_ranks:
            target_range = target.layer_range(rank)
            current_range = source.layer_range(rank)
            for missing_range in _range_difference(target_range, current_range):
                weight_loads.append((rank, missing_range))

        return cls(
            source=source,
            target=target,
            kv_transfers=tuple(transfers),
            weight_loads=tuple(weight_loads),
        )

    @property
    def ranks_to_activate(self) -> frozenset[int]:
        return frozenset(set(self.target.active_ranks) - set(self.source.active_ranks))

    @property
    def ranks_to_deactivate(self) -> frozenset[int]:
        return frozenset(set(self.source.active_ranks) - set(self.target.active_ranks))

    @property
    def required_weight_ranks(self) -> frozenset[int]:
        return frozenset(rank for rank, _ in self.weight_loads)

    @property
    def required_kv_ranks(self) -> frozenset[int]:
        return frozenset(transfer.dst_rank for transfer in self.kv_transfers)

    @property
    def required_cleanup_ranks(self) -> frozenset[int]:
        return frozenset(transfer.src_rank for transfer in self.kv_transfers)


class TransitionPhase(str, Enum):
    PREPARING = "preparing"
    READY_TO_FENCE = "ready_to_fence"
    FENCING = "fencing"
    COMMITTED = "committed"
    CLEANING = "cleaning"
    DONE = "done"
    ABORTED = "aborted"


@dataclass
class DynamicPPTransitionState:
    """Mutable acknowledgements around an immutable transition plan.

    The commit fence includes every resident rank, including currently idle
    ranks.  This mirrors the fixed maximal communicator: no process may retain
    an old generation when the next batch is published.
    """

    transition: DynamicPPTransition
    phase: TransitionPhase = TransitionPhase.PREPARING
    weight_ready_ranks: set[int] = field(default_factory=set)
    kv_ready_ranks: set[int] = field(default_factory=set)
    fence_ready_ranks: set[int] = field(default_factory=set)
    cleanup_ready_ranks: set[int] = field(default_factory=set)

    @property
    def resident_ranks(self) -> frozenset[int]:
        return frozenset(range(self.transition.source.max_pp_size))

    def mark_weights_ready(self, rank: int) -> None:
        self._require_phase(TransitionPhase.PREPARING)
        self._require_rank(rank, self.transition.required_weight_ranks, "weight")
        self.weight_ready_ranks.add(rank)
        self._maybe_ready_to_fence()

    def mark_kv_ready(self, rank: int) -> None:
        self._require_phase(TransitionPhase.PREPARING)
        self._require_rank(rank, self.transition.required_kv_ranks, "KV")
        self.kv_ready_ranks.add(rank)
        self._maybe_ready_to_fence()

    def begin_fence(self) -> None:
        self._require_phase(TransitionPhase.READY_TO_FENCE)
        self.phase = TransitionPhase.FENCING

    def mark_fence_ready(self, rank: int) -> None:
        self._require_phase(TransitionPhase.FENCING)
        self._require_rank(rank, self.resident_ranks, "fence")
        self.fence_ready_ranks.add(rank)

    def commit(self) -> PPTopology:
        self._require_phase(TransitionPhase.FENCING)
        missing = self.resident_ranks - self.fence_ready_ranks
        if missing:
            raise RuntimeError(
                f"cannot commit before every resident rank reaches the fence: {sorted(missing)}"
            )
        self.phase = TransitionPhase.COMMITTED
        if self.transition.required_cleanup_ranks:
            self.phase = TransitionPhase.CLEANING
        else:
            self.phase = TransitionPhase.DONE
        return self.transition.target

    def mark_cleanup_ready(self, rank: int) -> None:
        self._require_phase(TransitionPhase.CLEANING)
        self._require_rank(rank, self.transition.required_cleanup_ranks, "cleanup")
        self.cleanup_ready_ranks.add(rank)
        if self.cleanup_ready_ranks >= self.transition.required_cleanup_ranks:
            self.phase = TransitionPhase.DONE

    def abort(self) -> None:
        if self.phase in {
            TransitionPhase.COMMITTED,
            TransitionPhase.CLEANING,
            TransitionPhase.DONE,
        }:
            raise RuntimeError("a committed PP transition cannot be aborted")
        self.phase = TransitionPhase.ABORTED

    def _maybe_ready_to_fence(self) -> None:
        if (
            self.weight_ready_ranks >= self.transition.required_weight_ranks
            and self.kv_ready_ranks >= self.transition.required_kv_ranks
        ):
            self.phase = TransitionPhase.READY_TO_FENCE

    def _require_phase(self, expected: TransitionPhase) -> None:
        if self.phase != expected:
            raise RuntimeError(
                f"operation requires phase {expected.value}, current phase is {self.phase.value}"
            )

    @staticmethod
    def _require_rank(
        rank: int, expected_ranks: frozenset[int], operation: str
    ) -> None:
        if rank not in expected_ranks:
            raise ValueError(
                f"rank {rank} has no pending {operation} acknowledgement; "
                f"expected one of {sorted(expected_ranks)}"
            )

    def __post_init__(self) -> None:
        self._maybe_ready_to_fence()
