"""Own Mooncake registrations while stable KVCacheD pages change ownership.

New, already mapped pages may be registered during preparation. No address is
published then. Retired registrations are removed only after P/D writers drain;
retained pages remain registered. Abort must retire every attempted new region
before the caller can unmap prepared backing.
"""

from __future__ import annotations

import time


def regions(values):
    result = set()
    for pointer, length in values:
        pointer, length = int(pointer), int(length)
        if length == 0:
            continue
        if pointer <= 0 or length < 0:
            raise ValueError("invalid P/D registration region")
        result.add((pointer, length))
    return result


class PreparedDecodeRegions:
    def __init__(
        self, engine, source, target_kv, *, target_aux=(), deregister_batch_limit=None
    ):
        self.engine = engine
        self.deregister_batch_limit = deregister_batch_limit
        self.source, self.target_kv = regions(source), regions(target_kv)
        self.target_prepared = self.target_kv | regions(target_aux)
        # Sweep once. Source-only registrations need no revalidation, and
        # exact aliases appearing in both sets are tested only once.
        source_end = target_end = 0
        for region in sorted(self.source | self.target_prepared):
            ptr, size = region
            in_source = region in self.source
            in_target = region in self.target_prepared
            if (in_target and ptr < max(source_end, target_end)) or (
                in_source and ptr < target_end
            ):
                raise ValueError("P/D prepared regions overlap a different registration")
            if in_source:
                source_end = max(source_end, ptr + size)
            if in_target:
                target_end = max(target_end, ptr + size)
        self.registered = set(self.source)
        self.attempted_added = set()
        self.phase = "new"
        self.metrics = dict(
            source_regions=len(self.source),
            target_kv_regions=len(self.target_kv),
            source_bytes=sum(n for _, n in self.source),
            target_kv_bytes=sum(n for _, n in self.target_kv),
            target_prepared_regions=len(self.target_prepared),
            target_prepared_bytes=sum(n for _, n in self.target_prepared),
            target_aux_regions=len(self.target_prepared - self.target_kv),
        )

    def _register(self, values, phase):
        values = sorted(values)
        started = time.monotonic()
        if values:
            pointers, lengths = zip(*values)
            status = self.engine.batch_register(list(pointers), list(lengths))
            if status != 0:
                raise RuntimeError(f"P/D {phase} RDMA registration failed: {status}")
        self.metrics[phase + "_seconds"] = time.monotonic() - started
        self.metrics[phase + "_regions"] = len(values)
        self.metrics[phase + "_bytes"] = sum(n for _, n in values)

    def _deregister(self, values, phase):
        values = sorted(values)
        started = time.monotonic()
        if values:
            batch_size = self.deregister_batch_limit or len(values)
            for start in range(0, len(values), batch_size):
                batch = values[start : start + batch_size]
                status = self.engine.batch_deregister([ptr for ptr, _ in batch])
                if status != 0:
                    raise RuntimeError(f"P/D {phase} RDMA deregistration failed: {status}")
        self.metrics[phase + "_seconds"] = time.monotonic() - started
        self.metrics[phase + "_regions"] = len(values)
        self.metrics[phase + "_bytes"] = sum(n for _, n in values)

    def prepare(self):
        if self.phase != "new":
            raise RuntimeError("P/D region preparation already attempted")
        # Record attempts before entering the engine: failures may be partial.
        self.attempted_added = self.target_prepared - self.source
        self.phase = "preparing"
        self._register(self.attempted_added, "prepare")
        self.registered.update(self.attempted_added)
        self.phase = "prepared"

    def suspend(self):
        if self.phase != "prepared":
            raise RuntimeError("P/D regions have no complete preparation")
        retired = self.source - self.target_prepared
        self.phase = "suspending"
        self._deregister(retired, "suspend")
        self.registered.difference_update(retired)
        self.phase = "suspended"

    def commit(self, target):
        if self.phase != "suspended":
            raise RuntimeError("P/D region commit requires suspended writers")
        target = regions(target)
        if not self.target_prepared <= target or not self.registered <= target:
            raise RuntimeError("committed P/D addresses differ from prepared regions")
        self.phase = "committing"
        self._register(target - self.registered, "refresh")
        self.registered = target
        self.phase = "committed"

    def abort(self):
        if self.phase in {"aborted", "restored"}:
            return
        if self.phase in {"committed", "committing"}:
            raise RuntimeError("cannot abort published P/D registrations")
        if self.phase == "suspending":
            raise RuntimeError(
                "unknown partial P/D deregistration; backing must stay mapped"
            )
        self._deregister(self.attempted_added, "abort")
        self.registered.difference_update(self.attempted_added)
        self.attempted_added.clear()
        self.phase = "aborted"

    def restore_source(self):
        if self.phase != "aborted":
            raise RuntimeError("P/D source restore requires prepared backing abort")
        # Only a successfully completed suspend has a known retained subset.
        self._register(self.source - self.registered, "restore")
        self.registered = set(self.source)
        self.phase = "restored"
