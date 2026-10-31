"""Bounded unfenced PP flow diagnostic, separate from production and A/E profile.

Each actual replay gets unique outer CUDA events. Timestamp reads and the only
device fence happen at the final idle boundary. Gaps are current-stream gaps,
not device-idle time; host communication calls include waiting and metadata.
"""
from contextvars import ContextVar
from functools import wraps
import time

_FRAME = ContextVar('pp_flow_frame', default=None)
_PHASE = ContextVar('pp_flow_phase', default=None)
_DRAFT = ContextVar('pp_flow_draft', default=None)
_STATE = None
MAX_REPLAYS = 256
MAX_HOST = 4096


def status():
    s = _STATE
    return dict(armed=bool(s and s['armed']), finished=bool(s and s['finished']),
                replays=len(s['replays']) if s else 0,
                per_replay_fences=0, per_replay_timestamp_reads=0)


def arm(scheduler):
    global _STATE
    if _STATE is not None or not scheduler.is_fully_idle():
        raise RuntimeError('Unfenced flow must arm once at fully-idle boundary')
    import torch
    # Materialize event handles before the diagnostic cohort, never in replay.
    pool = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(MAX_REPLAYS)]
    for begin, end in pool:
        begin.record(); end.record()
    anchor = torch.cuda.Event(enable_timing=True)
    anchor.record(); anchor.synchronize()
    _STATE = dict(armed=True, finished=False, pool=pool, anchor=anchor,
                  replays=[], host=[], target_forwards=0)
    return status()


def finish(scheduler):
    s = _STATE
    if s is None or not s['armed'] or not scheduler.is_fully_idle():
        raise RuntimeError('Unfenced flow must finish at fully-idle boundary')
    import torch
    torch.cuda.synchronize()
    rows = []
    for row in s['replays']:
        begin, end = s['pool'][row['index']]
        rows.append(dict(row, start_ms=float(s['anchor'].elapsed_time(begin)),
                         end_ms=float(s['anchor'].elapsed_time(end)),
                         elapsed_ms=float(begin.elapsed_time(end))))
    if not rows or s['target_forwards'] < 12:
        raise RuntimeError('Missing actual target replays for unfenced flow')
    from sglang.srt.runtime_context import get_parallel
    from common import save
    import os
    p = get_parallel()
    save(f'{os.environ["GLM_ARM"]}/unfenced_flow_rank{p.world_rank}.json', dict(
        passed=True, rank=p.world_rank, pp_rank=p.pp_rank, replays=rows,
        host_calls=s['host'], target_forwards=s['target_forwards'],
        actual_graphs=True, unique_events_per_replay=True,
        per_replay_fences=0, per_replay_timestamp_reads=0, final_idle_fence=1,
        same_gpu_clock_only=True, no_gpu_idle_fraction_claim=True,
        timing_scope='actual Graph outer current-stream intervals; host PP API wait+enqueue, not pure wire time'))
    s.update(armed=False, finished=True)
    return status()


def install():
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
    from activity_metadata import batch_metadata, target_phase
    import torch

    original_run = Scheduler.run_batch
    @wraps(original_run)
    def run(self, batch, *args, **kwargs):
        if _STATE is None or not _STATE['armed']:
            return original_run(self, batch, *args, **kwargs)
        frame = batch_metadata(batch)
        frame['pp_microstep'] = getattr(self, '_pp_microstep', None)
        token = _FRAME.set(frame)
        try:
            return original_run(self, batch, *args, **kwargs)
        finally:
            _FRAME.reset(token)
    Scheduler.run_batch = run

    original_forward = ModelRunner.forward
    @wraps(original_forward)
    def forward(self, batch, *args, **kwargs):
        if _STATE is None or not _STATE['armed']:
            return original_forward(self, batch, *args, **kwargs)
        phase = (_DRAFT.get() or 'draft_model') if self.is_draft_worker else target_phase(batch)
        token = _PHASE.set(phase)
        if phase in ('target_verify', 'target_decode'):
            _STATE['target_forwards'] += 1
        try:
            return original_forward(self, batch, *args, **kwargs)
        finally:
            _PHASE.reset(token)
    ModelRunner.forward = forward

    original_replay = FullCudaGraphBackend.replay
    @wraps(original_replay)
    def replay(self, shape_key, static_forward_batch, *args, **kwargs):
        s, phase = _STATE, _PHASE.get()
        if s is None or not s['armed'] or phase is None:
            return original_replay(self, shape_key, static_forward_batch, *args, **kwargs)
        index = len(s['replays'])
        if index >= MAX_REPLAYS:
            raise RuntimeError('Unfenced flow replay bound exceeded')
        begin, end = s['pool'][index]
        row = dict(index=index, phase=phase, shape_key=str(shape_key),
                   stream=int(torch.cuda.current_stream().cuda_stream),
                   cpu_start_ns=time.perf_counter_ns(), **(_FRAME.get() or {}))
        s['replays'].append(row)
        begin.record()
        try:
            return original_replay(self, shape_key, static_forward_batch, *args, **kwargs)
        finally:
            end.record()
            row['cpu_end_ns'] = time.perf_counter_ns()
    FullCudaGraphBackend.replay = replay

    def draft_wrapper(original, phase):
        @wraps(original)
        def wrapped(self, *args, **kwargs):
            token = _DRAFT.set(phase)
            try:
                return original(self, *args, **kwargs)
            finally:
                _DRAFT.reset(token)
        return wrapped
    for name, phase in (('draft', 'draft'), ('_draft_extend_for_decode', 'draft_extend')):
        setattr(EagleDraftWorker, name, draft_wrapper(getattr(EagleDraftWorker, name), phase))

    def host_wrapper(original, name):
        @wraps(original)
        def wrapped(self, *args, **kwargs):
            s = _STATE
            if s is None or not s['armed']:
                return original(self, *args, **kwargs)
            if len(s['host']) >= MAX_HOST:
                raise RuntimeError('Unfenced flow host bound exceeded')
            row = dict(name=name, microstep=getattr(self, '_pp_microstep', None),
                       cpu_start_ns=time.perf_counter_ns())
            s['host'].append(row)
            try:
                return original(self, *args, **kwargs)
            finally:
                row['cpu_end_ns'] = time.perf_counter_ns()
        return wrapped
    for name in ('_pp_recv_proxy_tensors', '_pp_send_launched_proxy',
                 '_pp_send_recv_and_preprocess_output_tensors',
                 '_pp_prep_batch_result', '_pp_process_batch_result'):
        setattr(Scheduler, name, host_wrapper(getattr(Scheduler, name), name))
