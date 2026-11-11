"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.elastic_pp.kv_slots import ElasticKVSlotTracker
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        self._elastic_pp_slot_tracker: ElasticKVSlotTracker | None = None
        self._elastic_pp_live_patch = None
        self.clear()

    def clear(self):
        if getattr(self, "_elastic_pp_live_patch", None) is not None:
            raise RuntimeError("cannot clear the KV allocator during online KV patch")
        if getattr(self, "_elastic_pp_slot_tracker", None) is not None:
            raise RuntimeError(
                "cannot clear the KV allocator while an elastic PP snapshot "
                "generation is being tracked"
            )
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def available_size(self):
        # To avoid minor "len(free_pages) * 1" overhead
        return len(self.free_pages) + len(self.release_pages)

    def resize(self, config):
        # Base.resize mutates size before clear; reject before that mutation.
        if (
            self._elastic_pp_live_patch is not None
            or self._elastic_pp_slot_tracker is not None
        ):
            raise RuntimeError("cannot resize the KV allocator during elastic tracking")
        super().resize(config)

    def alloc(self, need_size: int):
        if self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()

        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        if self._elastic_pp_slot_tracker is not None:
            self._elastic_pp_slot_tracker.record_alloc(select_index)
        patch = self._elastic_pp_live_patch
        if patch is not None:
            patch.allocated(select_index)
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            patch = self._elastic_pp_live_patch
            if patch is not None:
                # A grouped free reaches this hook only when physically reusable.
                patch.freeing(free_index)
            if self.need_sort:
                self.release_pages = torch.cat((self.release_pages, free_index))
            else:
                self.free_pages = torch.cat((self.free_pages, free_index))
        else:
            self.free_group.append(self._copy_for_free_group(free_index))

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def attach_elastic_pp_live_patch(self, patch):
        if (
            self._elastic_pp_live_patch is not None
            or self._elastic_pp_slot_tracker is not None
        ):
            raise RuntimeError("another elastic KV tracker is already attached")
        if not self.is_not_in_free_group or self.free_group:
            raise RuntimeError("online KV patch requires a closed allocator free group")
        self._elastic_pp_live_patch = patch

    def detach_elastic_pp_live_patch(self, patch):
        if self._elastic_pp_live_patch is patch:
            self._elastic_pp_live_patch = None

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )

    def elastic_pp_live_slots(self) -> torch.Tensor:
        """Return allocated slot ids, including radix-cache-resident rows."""

        if not self.is_not_in_free_group or self.free_group:
            raise RuntimeError(
                "elastic PP live-slot snapshots require a closed allocator "
                "free group"
            )
        live_mask = torch.ones(self.size + 1, dtype=torch.bool, device=self.device)
        live_mask[0] = False
        if self.free_pages.numel():
            live_mask[self.free_pages] = False
        if self.release_pages.numel():
            live_mask[self.release_pages] = False
        return torch.nonzero(live_mask, as_tuple=False).flatten()

    def begin_elastic_pp_slot_tracking(self, *, generation: int) -> torch.Tensor:
        if self._elastic_pp_live_patch is not None:
            raise RuntimeError("online KV patch is already attached")
        if self._elastic_pp_slot_tracker is not None:
            if self._elastic_pp_slot_tracker.generation != generation:
                raise RuntimeError(
                    "another elastic PP KV slot generation is already tracked: "
                    f"pending={self._elastic_pp_slot_tracker.generation}, "
                    f"requested={generation}"
                )
            return self._elastic_pp_slot_tracker.initial_live_slots
        live_slots = self.elastic_pp_live_slots()
        self._elastic_pp_slot_tracker = ElasticKVSlotTracker(
            generation=generation,
            size=self.size,
            initial_live_slots=live_slots,
        )
        return self._elastic_pp_slot_tracker.initial_live_slots

    def elastic_pp_delta_slots(self, *, generation: int) -> torch.Tensor:
        tracker = self._elastic_pp_slot_tracker
        if tracker is None or tracker.generation != generation:
            raise RuntimeError(
                f"no elastic PP KV slot tracking exists for generation {generation}"
            )
        return tracker.changed_live_slots(self.elastic_pp_live_slots())

    def finish_elastic_pp_slot_tracking(self, *, generation: int) -> None:
        tracker = self._elastic_pp_slot_tracker
        if tracker is None or tracker.generation != generation:
            raise RuntimeError(
                f"no elastic PP KV slot tracking exists for generation {generation}"
            )
        self._elastic_pp_slot_tracker = None
