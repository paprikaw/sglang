"""CPU-only state for experimental, request-sticky PP ring slots.

This stores references to existing requests, never KV tensors or new requests.
It is intentionally independent of the model/distributed runtime so its ownership
and abort bookkeeping can be tested without initializing CUDA.
"""

from __future__ import annotations


class PPSlotAffinityState:
    def __init__(self, num_slots: int, max_batch_size: int):
        if num_slots < 2 or max_batch_size < 1:
            raise ValueError("PP slot affinity requires >=2 slots and a positive cap")
        self.chunks = [None] * num_slots
        self.pending_aborts = [None] * num_slots
        self.max_batch_size = max_batch_size

    @staticmethod
    def validate_supported(**unsupported):
        enabled = [name for name, value in unsupported.items() if value]
        if enabled:
            raise ValueError(
                "SGLANG_PP_SLOT_AFFINITY does not support: " + ", ".join(enabled)
            )

    def load(self, scheduler, slot: int):
        scheduler._pp_current_slot = slot
        scheduler.chunked_req = self.chunks[slot]
        scheduler._pending_chunked_abort_req = self.pending_aborts[slot]

    def save(self, scheduler):
        slot = scheduler._pp_current_slot
        self.chunks[slot] = scheduler.chunked_req
        self.pending_aborts[slot] = scheduler._pending_chunked_abort_req

    def can_admit(self, req, slot: int) -> bool:
        return getattr(req, "_pp_slot_affinity_owner", slot) == slot

    def check_plan(self, slot: int, running_batch, batch_to_run, chunked_req):
        # Include the unfinished chunk in the slot's capacity reservation even
        # when it is absent from running_batch. Deduplicate the same Req viewed
        # through both the running and launch batch.
        reqs = {}
        for batch in (running_batch, batch_to_run):
            if batch is not None:
                reqs.update((id(req), req) for req in batch.reqs if not req.finished())
        if chunked_req is not None and not chunked_req.finished():
            reqs[id(chunked_req)] = chunked_req
        if len(reqs) > self.max_batch_size:
            raise RuntimeError(
                f"PP slot {slot} has {len(reqs)} requests, exceeding "
                f"pp_max_micro_batch_size={self.max_batch_size}; refusing launch"
            )
        for req in reqs.values():
            if not self.can_admit(req, slot):
                raise RuntimeError(
                    f"PP request {req.rid} moved from slot "
                    f"{req._pp_slot_affinity_owner} to {slot}; refusing launch"
                )
            req._pp_slot_affinity_owner = slot

    def queue_abort(self, scheduler, abort_request):
        for slot, req in enumerate(self.chunks):
            if req is not None and (
                abort_request.abort_all or req.rid.startswith(abort_request.rid)
            ):
                # Each owner processes its marker only after its previous
                # forward has drained, through process_pending_chunked_abort.
                self.pending_aborts[slot] = req
        scheduler._pending_chunked_abort_req = self.pending_aborts[
            scheduler._pp_current_slot
        ]

    def is_drained(self) -> bool:
        return not any(self.chunks) and not any(self.pending_aborts)

    def reset(self):
        if not self.is_drained():
            raise RuntimeError("Cannot reset PP slot affinity with live requests")
        self.chunks[:] = [None] * len(self.chunks)
        self.pending_aborts[:] = [None] * len(self.pending_aborts)
