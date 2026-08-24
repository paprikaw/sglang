"""Sparse live-slot tracking for elastic PP KV migration."""

from __future__ import annotations

import torch


class ElasticKVSlotTracker:
    """Record slots allocated after a bulk-copy snapshot.

    Autoregressive KV rows are immutable after their write.  Therefore the
    final fence only needs rows allocated after the initial snapshot that are
    still live.  A freed-and-reused old slot is included because its reuse is a
    new allocation.  Tracking is enabled only while a transition is preparing,
    so the steady-state allocator pays no extra kernel or clone.
    """

    def __init__(
        self, *, generation: int, size: int, initial_live_slots: torch.Tensor
    ) -> None:
        if generation < 0:
            raise ValueError("KV slot-tracker generation must be non-negative")
        if size <= 0:
            raise ValueError("KV slot-tracker size must be positive")
        self.generation = int(generation)
        self.size = int(size)
        self.initial_live_slots = initial_live_slots.detach().clone()
        self._allocated_chunks: list[torch.Tensor] = []

    def record_alloc(self, slot_ids: torch.Tensor) -> None:
        if slot_ids.numel():
            self._allocated_chunks.append(slot_ids.detach().clone())

    def changed_live_slots(self, current_live_slots: torch.Tensor) -> torch.Tensor:
        if not self._allocated_chunks:
            return torch.empty(
                0,
                dtype=current_live_slots.dtype,
                device=current_live_slots.device,
            )
        allocated = torch.unique(torch.cat(self._allocated_chunks))
        live_mask = torch.zeros(
            self.size + 1, dtype=torch.bool, device=current_live_slots.device
        )
        if current_live_slots.numel():
            live_mask[current_live_slots] = True
        return allocated[live_mask[allocated]]
