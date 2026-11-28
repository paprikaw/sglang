"""Move occupied KV rows out of the address range a Prefill shrink will retire."""

import hashlib
from array import array

import torch

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType


def relocation_rows(live, capacity):
    high = sorted(row for row in live if row > capacity)
    if not high:
        return [], []
    if len(live) > capacity:
        raise RuntimeError(
            f"defer Prefill shrink: {len(live)} live KV rows exceed {capacity} "
            "after HiCache demotion"
        )
    free_low = []
    for row in range(1, capacity + 1):
        if row not in live:
            free_low.append(row)
            if len(free_low) == len(high):
                break
    return high, free_low


class PrefillKVCompaction:
    """Stage an attention-only row rename after P/D and HiCache I/O drain."""

    def __init__(self, scheduler, pool, capacity):
        self.pool = pool
        self.allocator = allocator = pool._logical_allocator
        live = allocator._live_page_ids
        high, low = relocation_rows(live, capacity)
        self.signature = (
            len(live),
            len(high),
            hashlib.sha256(array("q", sorted(live)).tobytes()).hexdigest(),
        )
        self.rows = high
        self.updates = []
        self.cpu_bytes = 0
        self.request_rows = 0
        if not high:
            return

        self.source = torch.tensor(high, device=allocator.device, dtype=torch.int64)
        self.target = torch.tensor(low, device=allocator.device, dtype=torch.int64)
        self.live = (live - set(high)) | set(low)
        # Keep the source capacity valid until the source registrations retire.
        # If no physical change follows the rename, the source may still serve.
        used = torch.zeros(
            allocator.serving_capacity + 1, dtype=torch.bool, device=allocator.device
        )
        used[0] = True
        used[torch.tensor(sorted(self.live), device=allocator.device)] = True
        self.free = (~used).nonzero(as_tuple=True)[0]
        self.empty = self.free[:0]

        owners = set()
        active = set()

        def rebind(view, owner_set):
            # The unified radix root uses an empty list as its slot sentinel.
            if view is None or not len(view):
                return
            ids = view.to(device=allocator.device, dtype=torch.int64)
            pos = torch.searchsorted(self.source, ids).clamp(max=len(high) - 1)
            affected = self.source[pos] == ids
            found = ids[affected].tolist()
            if found:
                owner_set.update(found)
                values = torch.where(affected, self.target[pos], ids)
                self.updates.append((view, values.to(device=view.device, dtype=view.dtype)))

        for node in scheduler.tree_cache.tree_core._node_arena.values():
            rebind(node.component_data[ComponentType.FULL].value, owners)
        for req in scheduler._elastic_pp_live_requests():
            rebind(req.prefix_indices, active)
            if req.req_pool_idx is not None and req.kv is not None:
                rebind(
                    scheduler.req_to_token_pool.req_to_token[
                        req.req_pool_idx, :req.kv.kv_allocated_len
                    ],
                    active,
                )
        if set(high) != owners | active:
            raise RuntimeError("defer Prefill shrink: an occupied KV row has no known owner")
        self.request_rows = len(active)

    def stage_on_free_rows(self):
        if not self.rows or not self.pool.k_buffer:
            return
        from sglang.srt.elastic_pp.backing_spill import _require_cpu_space

        chunk = self.pool.cpu_offloading_chunk_size
        row_bytes = sum(
            tensor[0].nbytes for tensor in (*self.pool.k_buffer, *self.pool.v_buffer)
        )
        _require_cpu_space(min(chunk, len(self.rows)) * row_bytes)
        # Only the scratch destinations are written. The original high rows
        # remain authoritative until every rank has acknowledged the CPU copy.
        for start in range(0, len(self.rows), chunk):
            end = start + chunk
            with torch.no_grad():
                saved = self.pool.get_cpu_copy(self.source[start:end])
                self.pool.load_cpu_copy(saved, self.target[start:end])
            del saved
        self.cpu_bytes = len(self.rows) * row_bytes

    def commit(self):
        if not self.rows:
            return
        for view, values in self.updates:
            view.copy_(values)
        self.allocator.free_pages = self.free
        self.allocator.release_pages = self.empty
        self.allocator._live_page_ids = self.live
        if self.source.is_cuda:
            torch.cuda.current_stream(self.source.device).synchronize()
