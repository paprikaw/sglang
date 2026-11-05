"""Untimed control of cohort identity; its admission wait stays in client latency."""
from functools import wraps
import os
import time

from pp_schedule_opt import open_policy

_INSTALLED = False


def ordered_local(requests, size, dp_rank):
    expected = set(range(5000 + int(os.environ.get("MATCHED_RANK_OFFSET", "0")) + dp_rank, 5000 + size, 4))
    markers = [r.origin_input_ids[0] for r in requests]
    if len(markers) != len(set(markers)) or not set(markers).issubset(expected):
        raise RuntimeError("EP local cohort contains unexpected or duplicate requests")
    return sorted(requests, key=lambda r:r.origin_input_ids[0]) if set(markers) == expected else None


def instrument(cls, read_epoch, read_size, clock=time.monotonic):
    original, internal = cls.get_next_batch_to_run, cls.get_internal_state

    @wraps(original)
    def get_next(self, *args, **kwargs):
        epoch = read_epoch()
        if epoch and self.ps.pp_size == 1 and getattr(self, "_prefill_admitted_epoch", None) != epoch:
            if self.waiting_queue:
                if getattr(self, "_prefill_wait_epoch", None) != epoch:
                    self._prefill_wait_epoch, self._prefill_wait_since = epoch, clock()
                ordered = ordered_local(self.waiting_queue, read_size(), self.ps.attn_dp_rank)
                if ordered is None:
                    if clock() - self._prefill_wait_since > 5:
                        raise TimeoutError("EP declared local cohort did not arrive")
                    waiting, self.waiting_queue = self.waiting_queue, []
                    try: return original(self, *args, **kwargs)
                    finally: self.waiting_queue = waiting + self.waiting_queue
                self.waiting_queue[:] = ordered
                self._prefill_admitted_epoch = epoch
                self._prefill_admission_receipt = dict(epoch=epoch, dp_rank=self.ps.attn_dp_rank,
                    ordered_markers=[r.origin_input_ids[0] for r in ordered])
        return original(self, *args, **kwargs)

    @wraps(internal)
    def state(self, *args, **kwargs):
        result = internal(self, *args, **kwargs)
        if hasattr(self, "_prefill_admission_receipt"):
            result.internal_state["experiment_prefill_ep_receipt"] = self._prefill_admission_receipt
        return result

    cls.get_next_batch_to_run, cls.get_internal_state = get_next, state


def install():
    global _INSTALLED
    if not _INSTALLED and os.environ["EP2_PP2_ARM"] == "ep2":
        from sglang.srt.managers.scheduler import Scheduler
        epoch = open_policy(os.environ["PP_FIXED_COHORT_CONTROL"])
        size = open_policy(os.environ["PP_FIXED_COHORT_SIZE"])
        instrument(Scheduler, lambda:epoch[0], lambda:size[0])
        _INSTALLED = True
