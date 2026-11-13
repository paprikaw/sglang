"""Quiescent P/D registration transaction for the initial Decode qualification.

No transport restart, in-flight writes, or stale registered CUDA addresses are
permitted.  Admission at the external router must be held during migration.
Existing decode requests retain their slots and continue after the PP fence.
"""

from __future__ import annotations


def validate_decode_args(args):
    required = dict(
        elastic_pp_disagg_decode=True,
        disaggregation_mode="decode",
        disaggregation_transfer_backend="mooncake",
        elastic_pp_transition_strategy="sync",
        elastic_pp_transition_backing="cpu_spill",
        disable_cuda_graph=True,
        enable_hierarchical_cache=True,
        pp_async_batch_depth=0,
    )
    mismatches = [
        f"{name}={getattr(args, name, None)!r} (requires {value!r})"
        for name, value in required.items()
        if getattr(args, name, None) != value
    ]
    if mismatches:
        raise ValueError("elastic P/D Decode qualification: " + "; ".join(mismatches))


def pending_pd_work(scheduler):
    """Do not confuse running decode requests with outstanding RDMA writers."""
    prealloc = scheduler.disagg_decode_prealloc_queue
    transfer = scheduler.disagg_decode_transfer_queue
    return (
        len(prealloc.queue)
        + len(getattr(prealloc, "pending_reqs", ()))
        + len(getattr(prealloc, "retracted_queue", ()))
        + len(transfer.queue)
        + int(transfer.has_pending_deferred_releases())
    )


class DecodeRegistrationTransaction:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.suspended = False

    def suspend(self):
        if self.suspended:
            raise RuntimeError("P/D registration already suspended")
        if pending_pd_work(self.scheduler):
            raise RuntimeError("P/D transfers must finish before backing release")
        manager = self.scheduler.disagg_decode_prealloc_queue.kv_manager
        self.suspended = True
        manager.deregister_buffer_to_engine()

    def refresh(self):
        if not self.suspended:
            return
        queue = self.scheduler.disagg_decode_prealloc_queue
        manager = queue.kv_manager
        # The pool object survives migration; its layer storage and addresses
        # do not.  Rebuild all KV/GDN/aux registrations from the committed pool.
        manager.kv_args = queue._build_kv_args()
        manager.register_buffer_to_engine()
        # Deregistration clears cached bootstrap routes; each subsequent room
        # therefore republishes the current per-layer addresses to every P rank.
        self.suspended = False
