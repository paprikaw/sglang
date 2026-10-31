"""PP2/MTP one-way feedback experiment; no model math or batch changes.

The tail already owns the result that the head otherwise echoes back. Retain
that *owned* result until its ring slot retires. The activation group and the
tail->head feedback group are unchanged. Only the redundant head->tail echo
is removed. This is deliberately restricted to the qualified two-slot path.
"""
from dataclasses import dataclass


@dataclass
class PendingFeedback:
    ready: object
    payload: object
    request_ids: tuple


class TailFeedbackSlots:
    def __init__(self):
        self.pending = {}
        self.produced = 0
        self.retired = 0

    def put(self, slot, ready, payload, request_ids):
        if slot not in (0, 1) or slot in self.pending:
            raise RuntimeError("PP feedback slot overwritten before retirement")
        self.pending[slot] = PendingFeedback(ready, payload, tuple(request_ids))
        self.produced += 1

    def pop(self, slot, request_ids):
        if slot not in self.pending:
            raise RuntimeError("PP feedback slot retired without a producer")
        pending = self.pending[slot]
        if pending.request_ids != tuple(request_ids):
            raise RuntimeError("PP local feedback request ancestry mismatch")
        del self.pending[slot]
        self.retired += 1
        return pending


def _request_ids(batch, metadata=None):
    snapshot = getattr(metadata, "fwd_batch", None)
    return tuple(req.rid for req in (snapshot or batch).reqs)


def attach(mixin, *, can_skip, current_stream):
    """Install the same small override in real Scheduler and CPU test doubles."""
    old_init = mixin.init_pp_loop_state
    old_send = mixin._pp_send_output_to_next_stage
    old_recv = mixin._pp_recv_dict_from_prev_stage
    old_exchange = mixin._pp_send_recv_and_preprocess_output_tensors

    def init(self, *args, **kwargs):
        result = old_init(self, *args, **kwargs)
        if not self._pp_spec_microbatch or self.ps.pp_size != 2 or self.pp_loop_size != 2:
            raise RuntimeError("Local feedback requires qualified PP2/MTP two-slot scheduler")
        self._pp_local_feedback = TailFeedbackSlots()
        self._pp_local_echo_skipped = 0
        self._pp_local_network_received = 0
        self._pp_local_retiring = None
        return result

    def send(self, next_first_rank_mb_id, mbs, last_rank_comm_queue, pp_outputs):
        if not self.pp_group.is_last_rank:
            if pp_outputs is not None:
                self._pp_local_echo_skipped += 1
            return []
        target = mbs[next_first_rank_mb_id]
        if target is None:
            return []
        event, produced = last_rank_comm_queue.popleft()
        if target.forward_mode.is_prebuilt() or can_skip(target):
            return []
        metadata = self.mb_metadata[next_first_rank_mb_id]
        self._pp_local_feedback.put(
            next_first_rank_mb_id, event, produced, _request_ids(target, metadata)
        )
        current_stream(self).wait_event(event)
        return self._pp_send_dict_to_next_stage(
            produced.tensors, async_send=True, msg_type="output"
        )

    def recv(self):
        if not self.pp_group.is_last_rank:
            self._pp_local_network_received += 1
            return old_recv(self)
        if self._pp_local_retiring is None:
            raise RuntimeError("PP local result consumed outside a retirement slot")
        slot, rids = self._pp_local_retiring
        pending = self._pp_local_feedback.pop(slot, rids)
        current_stream(self).wait_event(pending.ready)
        return pending.payload.tensors

    def exchange(self, next_first_rank_mb_id, next_mb_id, mbs, mb_metadata,
                 last_rank_comm_queue, pp_outputs):
        if self._pp_requires_immediate_output_relay():
            raise RuntimeError("Local feedback is not qualified for elastic PP")
        target = mbs[next_mb_id]
        self._pp_local_retiring = (
            (next_mb_id, _request_ids(target, mb_metadata[next_mb_id]))
            if target is not None else None
        )
        try:
            return old_exchange(self, next_first_rank_mb_id, next_mb_id, mbs,
                                mb_metadata, last_rank_comm_queue, pp_outputs)
        finally:
            self._pp_local_retiring = None

    mixin.init_pp_loop_state = init
    mixin._pp_send_output_to_next_stage = send
    mixin._pp_recv_dict_from_prev_stage = recv
    mixin._pp_send_recv_and_preprocess_output_tensors = exchange
    # Preserve original functions for diagnosis, never use them to bypass a gate.
    mixin._pp_local_feedback_originals = (old_init, old_send, old_recv, old_exchange)


def receipt(scheduler):
    slots = getattr(scheduler, "_pp_local_feedback", None)
    if slots is None:
        return dict(enabled=False)
    if slots.pending or slots.produced != slots.retired:
        raise RuntimeError("Local PP feedback did not drain at fully-idle fence")
    return dict(enabled=True, pending_slots=[], produced=slots.produced,
                retired=slots.retired,
                echo_sends_skipped=scheduler._pp_local_echo_skipped,
                head_feedback_receives=scheduler._pp_local_network_received,
                same_owned_payload_retired_locally=True)


def install():
    from common import config
    import os
    from pathlib import Path
    cfg = config(Path(os.environ["GLM_RUN_DIR"]))
    if not cfg.get("local_tail_feedback"):
        return
    if cfg["layout"] != "pp2ep4" or not cfg["mtp"]:
        raise RuntimeError("Local-tail feedback enabled on unsupported arm")
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin, _pp_can_skip_output_comm
    attach(SchedulerPPMixin, can_skip=_pp_can_skip_output_comm,
           current_stream=lambda scheduler: scheduler.device_module.current_stream())
