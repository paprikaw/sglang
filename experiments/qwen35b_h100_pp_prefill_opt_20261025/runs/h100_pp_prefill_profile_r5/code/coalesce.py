"""Bounded PP input coalescing, before head forwards input to the next stage.

This private experiment adapter changes only admission timing. It never waits
for a named cohort or a future request count, and has a five millisecond cap.
Control messages pass through immediately. No model/kernel math is changed.
"""
from functools import wraps
import os
import time

_DONE = False
_POLICY = None


def gather(first, poll, is_work, *, clock=time.monotonic, sleep=time.sleep,
           target=8, timeout_s=.005):
    """Coalesce one bounded receive; preserve every object and input order."""
    result = list(first)
    if not result or any(not is_work(x) for x in result):
        return result
    if len(result) >= target:
        return result
    deadline = clock() + timeout_s
    while len(result) < target and clock() < deadline:
        sleep(min(.0002, max(0., deadline - clock())))
        incoming = list(poll())
        result.extend(incoming)
        if any(not is_work(x) for x in incoming):
            break
    return result


def install():
    global _DONE, _POLICY
    if _DONE:
        return
    import pp_ready_send
    from sglang.srt.managers.scheduler_components.request_receiver import SchedulerRequestReceiver
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    _POLICY = pp_ready_send.open_policy(os.environ['PP_COALESCE_POLICY_PATH'])
    original = SchedulerRequestReceiver.recv_requests

    @wraps(original)
    def receive(self):
        first = original(self)
        mode = _POLICY[0]
        if mode not in (ord('0'), ord('1')):
            raise RuntimeError('Invalid coalescing policy')
        if mode == ord('0') or self.ps.pp_rank != 0 or self.ps.pp_size != 2:
            return first
        # Head TP1 only: polling again cannot introduce unmatched collectives.
        if self.ps.tp_size != 1 or self.ps.attn_cp_size != 1:
            raise RuntimeError('This coalescer is qualified only for PP2/TP1/CP1')
        def eligible(req):
            return (isinstance(req, TokenizedGenerateReqInput)
                    and req.input_embeds is None and req.mm_inputs is None)
        return gather(first, lambda: original(self), eligible)

    SchedulerRequestReceiver.recv_requests = receive
    _DONE = True
