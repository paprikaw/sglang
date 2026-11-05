"""Bounded, CPU-only observations. No CUDA events, tensor copies or fences."""
from collections import deque
from functools import wraps
import json
import mmap
import os
from pathlib import Path
import socket
import threading
import time

_installed = False
_queue = deque()
_dropped = 0
_written = 0
_control = None
_scheduler = None
_next_sample = 0.0
_path = None


def enqueue(row):
    global _dropped
    if len(_queue) >= 32768:
        _dropped += 1
    else:
        _queue.append(row)


def writer():
    global _written
    while True:
        if _path is not None and _queue:
            batch = []
            for _ in range(min(512, len(_queue))):
                batch.append(json.dumps(_queue.popleft(), separators=(',', ':')) + '\n')
            with _path.open('a') as out:
                out.writelines(batch)
            _written += len(batch)
        time.sleep(.25)


def state_row(s):
    cache = s.tree_cache
    result = dict(fully_idle=s.is_fully_idle(), waiting=len(s.waiting_queue),
                  running=len(s.running_batch.reqs), cache_type=type(cache).__name__)
    for attr in ('full_evictable_size', 'mamba_evictable_size'):
        method = getattr(cache, attr, None)
        if method:
            value = method()
            if isinstance(value, (int, float)):
                result[attr] = value
    for attr in ('ongoing_write_through', 'ongoing_load_back', 'ongoing_prefetch'):
        value = getattr(cache, attr, None)
        if value is not None:
            result[attr] = len(value)
    return result


def install():
    global _installed, _control
    if _installed or not os.environ.get('MATRIX_CONTROL'):
        return
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_components.output_streamer import SchedulerOutputStreamer
    fd = os.open(os.environ['MATRIX_CONTROL'], os.O_RDONLY | os.O_NOFOLLOW)
    try:
        _control = mmap.mmap(fd, 1, access=mmap.ACCESS_READ)
    finally:
        os.close(fd)
    original_init = Scheduler.__init__

    @wraps(original_init)
    def init(self, *args, **kwargs):
        global _scheduler, _path
        original_init(self, *args, **kwargs)
        _scheduler = self
        root = Path(os.environ['MATRIX_LOCAL_ROOT']) / Path(os.environ['EP2_PP2_RUN']).name
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _path = root / f'pp{self.ps.pp_rank}.tp{self.ps.tp_rank}.jsonl'
        threading.Thread(target=writer, daemon=True).start()

    Scheduler.__init__ = init
    previous_output = SchedulerOutputStreamer._maybe_log_time_stats

    @wraps(previous_output)
    def output(self, *, req):
        previous_output(self, req=req)
        if not _control[0] or getattr(req, '_matrix_saved', False) or not req.finished():
            return
        req._matrix_saved = True
        # These are timestamps already maintained by the engine on CPU.
        stats = {k: v for k, v in vars(req.time_stats).items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)}
        enqueue(dict(kind='request', request_id=req.rid, monotonic_s=time.perf_counter(),
                     unix_ns=time.time_ns(), pp_rank=self.ps.pp_rank, tp_rank=self.ps.tp_rank,
                     role=os.environ.get('BENCH_INSTANCE_ROLE'), input_tokens=len(req.origin_input_ids),
                     output_tokens=len(req.output_ids), cached_tokens=req.cached_tokens,
                     attempts=req.prefill_attempt_count, times=stats,
                     finish_type=type(req.finished_reason).__name__))

    SchedulerOutputStreamer._maybe_log_time_stats = output
    original_batch = Scheduler.run_batch

    @wraps(original_batch)
    def batch(self, batch, *args, **kwargs):
        global _next_sample
        if _control[0] >= 2:
            now = time.perf_counter()
            if now >= _next_sample:
                _next_sample = now + 2
                reqs = batch.reqs
                # Host request lengths; do not read seq_lens CUDA tensors.
                enqueue(dict(kind='batch', monotonic_s=now, unix_ns=time.time_ns(),
                             pp_rank=self.ps.pp_rank, tp_rank=self.ps.tp_rank,
                             role=os.environ.get('BENCH_INSTANCE_ROLE'), mode=batch.forward_mode.name,
                             batch_size=len(reqs), waiting=len(self.waiting_queue),
                             context_lengths=[len(r.origin_input_ids)+len(r.output_ids) for r in reqs],
                             extend_input_len=getattr(batch, 'extend_num_tokens', None)))
        return original_batch(self, batch, *args, **kwargs)

    Scheduler.run_batch = batch
    old_state = Scheduler.get_internal_state

    @wraps(old_state)
    def get_state(self, *args, **kwargs):
        result = old_state(self, *args, **kwargs)
        result.internal_state['experiment_decode_service'] = dict(
            phase_timing=False, early_proxy=getattr(self, '_pp_pd_early_proxy', False),
            typed_channels=len(getattr(self, '_pp_pd_tensor_channels', [])),
            hicache_poll_steps=getattr(self, '_pp_pd_hicache_poll_steps', 1),
            microsteps=getattr(self, '_pp_pd_microsteps', 0),
            hicache_polls=getattr(self, '_pp_pd_hicache_polls', 0),
            early_proxy_sends=getattr(self, '_pp_pd_early_proxy_sends', 0))
        result.internal_state['matrix_observer'] = dict(
            level=_control[0], dropped=_dropped, pending=len(_queue), written=_written,
            path=str(_path), node=socket.gethostname(), **state_row(self))
        return result

    Scheduler.get_internal_state = get_state
    _installed = True
