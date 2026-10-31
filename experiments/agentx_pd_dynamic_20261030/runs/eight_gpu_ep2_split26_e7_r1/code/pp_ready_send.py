"""Experiment-local PP send readiness; no model or kernel mathematics change.

One node-local shared byte switches the already-loaded PP workers between the
original path and producer-ready metadata publication. Switch only at idle
cohort boundaries. This is intentionally not installed into the shared engine.
"""
from functools import wraps
import mmap
import os
from pathlib import Path
import stat

_MAPPING = None
_INSTALLED = False


def create_policy(path):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        os.write(fd, b"0")
    finally:
        os.close(fd)


def open_policy(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        s = os.fstat(fd)
        if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_mode & 0o077 or s.st_size != 1:
            raise RuntimeError("Policy must be a private UID-owned one-byte file")
        return mmap.mmap(fd, 1, access=mmap.ACCESS_READ)
    finally:
        os.close(fd)


def set_policy(path, enabled):
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        s = os.fstat(fd)
        if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_mode & 0o077 or s.st_size != 1:
            raise RuntimeError("Invalid policy file")
        os.pwrite(fd, b"1" if enabled else b"0", 0)
    finally:
        os.close(fd)


def wrap_send(original, read_policy, annotation):
    @wraps(original)
    def send(self, tensor_dict, async_send=True, msg_type="default", dst=None):
        mode = read_policy()
        if mode not in (ord("0"), ord("1")):
            raise RuntimeError("Unknown PP readiness mode")
        if (mode == ord("1") and msg_type in ("proxy", "output")
                and any(getattr(t, "is_cuda", False) for t in tensor_dict.values())):
            if self.ps.pp_size != 2:
                raise RuntimeError("This experiment only qualifies static PP2")
            # The caller has already inserted the producer event dependency on
            # this stream. Complete it BEFORE native send_tensor_dict publishes
            # CPU metadata; do not ask the receiver to launch a GPU wait early.
            with annotation("pp.ready_send.wait." + msg_type):
                event = self.device_module.Event(enable_timing=False, blocking=True)
                event.record(self.device_module.current_stream())
                event.synchronize()
        return original(self, tensor_dict, async_send=async_send, msg_type=msg_type, dst=dst)
    return send


def install(torch):
    global _MAPPING, _INSTALLED
    if _INSTALLED:
        return
    from contextlib import nullcontext
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
    _MAPPING = open_policy(os.environ["PP_READY_POLICY_PATH"])
    def annotation(name):
        return (torch.profiler.record_function(name)
                if torch.autograd._profiler_enabled() else nullcontext())
    SchedulerPPMixin._pp_send_dict_to_next_stage = wrap_send(
        SchedulerPPMixin._pp_send_dict_to_next_stage, lambda: _MAPPING[0], annotation)
    _INSTALLED = True
