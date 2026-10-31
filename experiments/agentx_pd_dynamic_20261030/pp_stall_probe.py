"""Bounded stack dumps for the reproduced PP Decode startup stall."""
import faulthandler
from functools import wraps


def install():
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
    original = SchedulerPPMixin.event_loop_pp_disagg_decode

    @wraps(original)
    def loop(self):
        faulthandler.dump_traceback_later(45, repeat=True)
        try:
            return original(self)
        finally:
            faulthandler.cancel_dump_traceback_later()

    SchedulerPPMixin.event_loop_pp_disagg_decode = loop
