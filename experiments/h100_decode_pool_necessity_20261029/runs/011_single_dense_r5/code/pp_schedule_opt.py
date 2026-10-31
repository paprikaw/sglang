"""Experiment-local, static PP2 activation scheduling; no output-ring changes."""
from functools import wraps
import mmap
import os
import stat

_INSTALLED = False
_POLICY = None


def open_policy(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        s = os.fstat(fd)
        if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_mode & 0o077 or s.st_size != 1:
            raise RuntimeError('Expected a private one-byte scheduling policy')
        return mmap.mmap(fd, 1, access=mmap.ACCESS_READ)
    finally:
        os.close(fd)


def wrap_methods(cls, read_policy):
    launch = cls._pp_launch_or_skip_batch
    retire = cls._pp_commit_send_output_work_and_preprocess_output_tensors
    send = cls._pp_send_launched_proxy

    def enabled(self):
        mode = read_policy()
        if mode not in (ord('0'), ord('1')):
            raise RuntimeError('Unknown scheduling policy')
        if mode == ord('1'):
            if self.ps.pp_size != 2 or self._pp_topology() is not None:
                raise RuntimeError('Candidate is limited to static full-stage PP2')
            return True
        return False

    @wraps(launch)
    def launch_wrapped(self, mb_id, batch, proxy, metadata, queue):
        result, event = launch(self, mb_id, batch, proxy, metadata, queue)
        # Preserve the original output relay/FIFO and last-rank behavior.
        # Only the head's produced activation is eligible for early submission.
        self._decode_opt_pending = ((batch, result)
            if batch is not None and not self._pp_is_last_rank() else None)
        return result, event

    @wraps(retire)
    def retire_wrapped(self, *args, **kwargs):
        pending = getattr(self, '_decode_opt_pending', None)
        if enabled(self) and pending is not None:
            # This is the SAME original send, with its original producer event,
            # stream, tensors and ownership. Do not alter output relay semantics.
            send(self, *pending)
            self._decode_opt_sent = pending
            self._decode_opt_pending = None
        return retire(self, *args, **kwargs)

    @wraps(send)
    def send_wrapped(self, batch, result):
        sent = getattr(self, '_decode_opt_sent', None)
        if sent is not None and sent[0] is batch and sent[1] is result:
            self._decode_opt_sent = None
            return None
        self._decode_opt_pending = None
        return send(self, batch, result)

    cls._pp_launch_or_skip_batch = launch_wrapped
    cls._pp_commit_send_output_work_and_preprocess_output_tensors = retire_wrapped
    cls._pp_send_launched_proxy = send_wrapped


def install():
    global _INSTALLED, _POLICY
    if _INSTALLED:
        return
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
    _POLICY = open_policy(os.environ['PP_DECODE_OPT_POLICY_PATH'])
    wrap_methods(SchedulerPPMixin, lambda: _POLICY[0])
    _INSTALLED = True
