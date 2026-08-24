"""Generation-safe model-layer residency for elastic pipeline parallelism."""

from __future__ import annotations

from dataclasses import dataclass

from sglang.srt.elastic_pp.topology import LayerRange


@dataclass(frozen=True)
class LayerPreparation:
    generation: int
    target: LayerRange
    new_layer_ids: frozenset[int]


class ElasticLayerResidency:
    """Track prepared, ready, scheduled, and reclaimable model layers.

    Module construction and checkpoint loading happen outside this class.  It
    enforces the control-plane invariant that a generation cannot become the
    scheduled range until every newly materialized layer is weight-ready.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        initial_generation: int,
        initial_range: LayerRange,
    ) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if initial_generation < 0:
            raise ValueError("initial_generation must be non-negative")
        self.num_layers = int(num_layers)
        self._validate_range(initial_range)
        self.generation = int(initial_generation)
        self.scheduled_range = initial_range
        self.pending: LayerPreparation | None = None
        self._loaded_new_layer_ids: set[int] = set()
        self._cleanup_generation: int | None = None
        self._cleanup_layer_ids: frozenset[int] = frozenset()

    def _validate_range(self, layer_range: LayerRange) -> None:
        if layer_range.end > self.num_layers:
            raise ValueError(
                "layer range exceeds model: "
                f"range={layer_range}, num_layers={self.num_layers}"
            )

    def begin_prepare(
        self,
        *,
        generation: int,
        target: LayerRange,
        resident_layer_ids: set[int] | frozenset[int],
    ) -> LayerPreparation:
        self._validate_range(target)
        if generation <= self.generation:
            raise ValueError(
                "prepared generation must be newer than the committed "
                f"generation: {generation} <= {self.generation}"
            )
        if self._cleanup_generation is not None:
            raise RuntimeError(
                "the previous elastic layer generation still has deferred "
                f"cleanup: generation={self._cleanup_generation}"
            )
        if self.pending is not None:
            if self.pending.generation == generation and self.pending.target == target:
                return self.pending
            raise RuntimeError(
                "another elastic layer generation is already preparing: "
                f"pending={self.pending.generation}, requested={generation}"
            )

        resident = {int(layer_id) for layer_id in resident_layer_ids}
        invalid = {
            layer_id
            for layer_id in resident
            if layer_id < 0 or layer_id >= self.num_layers
        }
        if invalid:
            raise ValueError(f"resident layer ids are outside the model: {invalid}")
        target_ids = set(range(target.start, target.end))
        preparation = LayerPreparation(
            generation=int(generation),
            target=target,
            new_layer_ids=frozenset(target_ids - resident),
        )
        self.pending = preparation
        self._loaded_new_layer_ids.clear()
        return preparation

    def mark_loaded(self, generation: int, layer_ids: set[int]) -> None:
        pending = self._require_pending(generation)
        unexpected = set(layer_ids) - pending.new_layer_ids
        if unexpected:
            raise ValueError(
                "cannot mark layers outside the preparation as loaded: "
                f"{sorted(unexpected)}"
            )
        self._loaded_new_layer_ids.update(layer_ids)

    def ready_to_commit(self, generation: int) -> bool:
        pending = self._require_pending(generation)
        return pending.new_layer_ids <= self._loaded_new_layer_ids

    def commit(self, generation: int) -> LayerRange:
        pending = self._require_pending(generation)
        missing = pending.new_layer_ids - self._loaded_new_layer_ids
        if missing:
            raise RuntimeError(
                "cannot commit before all new layers are weight-ready: "
                f"missing={sorted(missing)}"
            )
        old_active = set(range(self.scheduled_range.start, self.scheduled_range.end))
        new_active = set(range(pending.target.start, pending.target.end))
        self._cleanup_generation = pending.generation
        self._cleanup_layer_ids = frozenset(old_active - new_active)
        self.generation = pending.generation
        self.scheduled_range = pending.target
        self.pending = None
        self._loaded_new_layer_ids.clear()
        return self.scheduled_range

    def abort(self, generation: int) -> frozenset[int]:
        pending = self._require_pending(generation)
        new_layer_ids = pending.new_layer_ids
        self.pending = None
        self._loaded_new_layer_ids.clear()
        return new_layer_ids

    def cleanup_candidates(
        self,
        generation: int,
        resident_layer_ids: set[int] | frozenset[int],
    ) -> frozenset[int]:
        if self._cleanup_generation != generation:
            raise ValueError(
                "elastic layer cleanup generation mismatch: "
                f"pending={self._cleanup_generation}, requested={generation}"
            )
        return frozenset(set(resident_layer_ids) & self._cleanup_layer_ids)

    def finish_cleanup(self, generation: int, removed_layer_ids: set[int]) -> None:
        expected = self.cleanup_candidates(generation, removed_layer_ids)
        if expected != self._cleanup_layer_ids:
            missing = self._cleanup_layer_ids - set(removed_layer_ids)
            raise RuntimeError(
                "elastic layer cleanup did not remove every retired layer: "
                f"missing={sorted(missing)}"
            )
        self._cleanup_generation = None
        self._cleanup_layer_ids = frozenset()

    def _require_pending(self, generation: int) -> LayerPreparation:
        if self.pending is None:
            raise RuntimeError("no elastic layer generation is preparing")
        if self.pending.generation != generation:
            raise ValueError(
                "elastic layer generation mismatch: "
                f"pending={self.pending.generation}, requested={generation}"
            )
        return self.pending
