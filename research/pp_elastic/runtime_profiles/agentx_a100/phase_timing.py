"""Opt-in asynchronous forward spans and PP control timings for saved experiments.

CUDA event spans measure work on the forward stream, including its waits and
launch gaps. They are not sums of kernel busy time. No tensor values are read,
and the scheduler never synchronizes CUDA for this observer.
"""
from collections import defaultdict, deque
from functools import wraps
import json
import os
from pathlib import Path
import threading
import time

_INSTALLED = False
_PENDING = deque()
_CPU = defaultdict(lambda: [0, 0.0])
_SHAPE = None
_DROPPED = 0
_COMPLETED = 0
_ROOT = None


def install():
    global _INSTALLED
    if _INSTALLED or os.environ.get('BENCH_PHASE_TIMING') != '1':
        return
    import torch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    original = ModelRunner.forward

    @wraps(original)
    def forward(self, forward_batch, *args, **kwargs):
        global _SHAPE, _ROOT, _DROPPED
        if _SHAPE is None:
            _SHAPE = dict(pp_rank=self.ps.pp_rank, tp_rank=self.ps.tp_rank,
                pp_size=self.ps.pp_size, tp_size=self.ps.tp_size,
                ep_size=self.ps.moe_ep_size,
                role=os.environ.get('BENCH_INSTANCE_ROLE', 'unknown'))
            _ROOT = Path(os.environ['EP2_PP2_RUN']) / 'phase_timing'
            _ROOT.mkdir(parents=True, exist_ok=True)
            device = torch.cuda.current_device()
            threading.Thread(target=publish, args=(device,), daemon=True).start()
        if torch.cuda.is_current_stream_capturing() or len(_PENDING) >= 65536:
            _DROPPED += 1
            return original(self, forward_batch, *args, **kwargs)
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        wall_start = time.time_ns()
        cpu_start = time.perf_counter_ns()
        mode = forward_batch.forward_mode.name
        batch_size = forward_batch.batch_size
        input_rows = forward_batch.input_ids.numel()
        begin.record()
        result = original(self, forward_batch, *args, **kwargs)
        end.record()
        row = dict(kind='forward', start_ns=wall_start, end_ns=time.time_ns(),
            forward_mode=mode, batch_size=batch_size, input_rows=input_rows,
            cpu_launch_ms=(time.perf_counter_ns()-cpu_start)/1e6)
        _PENDING.append((begin, end, row))
        return result

    ModelRunner.forward = forward

    def cpu_timed(cls, name):
        previous = getattr(cls, name)

        @wraps(previous)
        def call(self, *args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return previous(self, *args, **kwargs)
            finally:
                key = (time.time_ns()//1_000_000_000, name)
                value = _CPU[key]
                value[0] += 1
                value[1] += (time.perf_counter_ns()-start)/1e6

        setattr(cls, name, call)

    for name in ['_pp_pd_get_retract_ids', '_pp_pd_get_prealloc_ids',
                 '_pp_pd_get_decode_transferred_ids', '_pp_recv_proxy_tensors',
                 '_pp_launch_or_skip_batch', '_pp_commit_comm_work',
                 '_pp_commit_send_output_work_and_preprocess_output_tensors',
                 '_pp_pd_send_consensus_bootstrapped_ids',
                 '_pp_pd_send_consensus_release_ids', '_pp_process_batch_result']:
        cpu_timed(SchedulerPPMixin, name)
    cpu_timed(UnifiedRadixCache, 'check_hicache_events_pp_decode')
    previous_state = Scheduler.get_internal_state

    @wraps(previous_state)
    def state(self, *args, **kwargs):
        result = previous_state(self, *args, **kwargs)
        result.internal_state['experiment_decode_service'] = dict(
            phase_timing=True, pending_spans=len(_PENDING), completed_spans=_COMPLETED,
            skipped_or_dropped_spans=_DROPPED,
            early_proxy=getattr(self, '_pp_pd_early_proxy', False),
            typed_channels=len(getattr(self, '_pp_pd_tensor_channels', [])),
            hicache_poll_steps=getattr(self, '_pp_pd_hicache_poll_steps', 1),
            microsteps=getattr(self, '_pp_pd_microsteps', 0),
            hicache_polls=getattr(self, '_pp_pd_hicache_polls', 0),
            early_proxy_sends=getattr(self, '_pp_pd_early_proxy_sends', 0))
        return result

    Scheduler.get_internal_state = state
    _INSTALLED = True


def publish(device):
    global _COMPLETED
    import torch
    torch.cuda.set_device(device)
    name = f'pp{_SHAPE["pp_rank"]}.tp{_SHAPE["tp_rank"]}'
    path = _ROOT / f'{name}.jsonl'
    status = _ROOT / f'{name}.status.json'
    with path.open('a', buffering=1) as output:
        while True:
            rows = []
            while _PENDING and _PENDING[0][1].query():
                begin, end, row = _PENDING.popleft()
                row['gpu_stream_ms'] = begin.elapsed_time(end)
                rows.append(row)
                _COMPLETED += 1
            now = time.time_ns()//1_000_000_000
            for key in list(_CPU):
                if key[0] < now:
                    calls, elapsed = _CPU.pop(key)
                    rows.append(dict(kind='cpu_control', second=key[0],
                        method=key[1], calls=calls, cpu_inclusive_ms=elapsed))
            if rows:
                output.write(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in rows))
            value = dict(shape=_SHAPE, time_ns=time.time_ns(), pending=len(_PENDING),
                         completed=_COMPLETED, skipped_or_dropped=_DROPPED)
            temp = status.with_suffix('.tmp')
            temp.write_text(json.dumps(value))
            os.replace(temp, status)
            time.sleep(0.25)
