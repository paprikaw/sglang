"""Separate-cohort CUDA activity for Decode and MTP, never performance timing.

Two existing Graph warmups provide independent CPU-scope/name calibration.
The existing PP P2P warmup supplies actual transport-name witnesses. No extra
model forward is introduced. Bootstrap/arm/stop are legal only at scheduler
idle. Inside model execution the only added GPU operations are paired tiny
marker kernels around coarse target/draft APIs; no component events or fences.
"""
from collections import defaultdict, deque
from contextvars import ContextVar
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import time

_CAL = ContextVar('stage_activity_calibration', default=False)
_SHARED = ContextVar('stage_activity_shared', default=False)
_DEPTH = ContextVar('stage_activity_segment_depth', default=0)
_FRAME = ContextVar('stage_activity_request_frame', default=None)
_STATE = None
MAX_TARGET_FORWARDS = 12
MAX_SEGMENTS = 160


def batch_metadata(batch, frame=None):
    """Read CPU identities/shapes only; never materialize a CUDA tensor."""
    requests = getattr(batch, 'reqs', None)
    ids = [r.rid for r in requests] if requests is not None else getattr(batch, 'rids', None)
    if not isinstance(ids, (tuple, list)) or not all(isinstance(x, str) for x in ids):
        ids = (frame or {}).get('request_ids', [])
    size = len(requests) if requests is not None else getattr(batch, 'batch_size', None)
    if type(size) is not int:
        size = (frame or {}).get('batch_size')
    inputs = getattr(batch, 'input_ids', None)
    shape = list(inputs.shape) if inputs is not None and hasattr(inputs, 'shape') else None
    lengths = getattr(batch, 'seq_lens_cpu', None)
    # A CPU Tensor conversion is deliberately excluded as well: receipts keep
    # only already-Python metadata so this helper is safe on every code path.
    lengths = list(lengths) if isinstance(lengths, (list, tuple)) else None
    return dict(request_ids=list(ids), batch_size=size, input_shape=shape,
                mode=getattr(getattr(batch, 'forward_mode', None), 'name', None),
                seq_lens_cpu=lengths, pp_microstep=(frame or {}).get('pp_microstep'))


def target_phase(batch, is_draft_worker=False):
    if is_draft_worker:
        return 'draft_model'
    mode = getattr(batch, 'forward_mode', None)
    if mode is not None and mode.is_target_verify():
        return 'target_verify'
    if mode is not None and mode.is_decode():
        return 'target_decode'
    return None


def status():
    return dict(installed=_STATE is not None,
                bootstrapped=bool(_STATE and _STATE.get('bootstrapped')),
                armed=bool(_STATE and _STATE.get('armed')),
                active=bool(_STATE and _STATE.get('profiler') is not None and not _STATE['done']),
                done=bool(_STATE and _STATE['done']))


def _config():
    from common import config
    return config(Path(os.environ['GLM_RUN_DIR']))


def _output(name):
    cfg = _config()
    return Path(cfg['run_dir']) / cfg['arm'] / name


def _save(path, value):
    from common import save
    save(path, value)


def _rank():
    if _STATE['rank'] is None:
        from sglang.srt.runtime_context import get_parallel
        p = get_parallel()
        _STATE['rank'] = dict(world_rank=p.world_rank, pp_rank=p.pp_rank,
                              tp_rank=p.tp_rank, attn_dp_rank=p.attn_dp_rank)
    return _STATE['rank']['world_rank']


def _require_idle(scheduler):
    if not scheduler.is_fully_idle():
        raise RuntimeError('CUDA activity control requires confirmed scheduler idle')
    from scheduler_hooks import quiescent
    quiet, counts = quiescent(scheduler.tree_cache)
    if not quiet:
        raise RuntimeError('CUDA activity control requires quiescent HiCache: ' + str(counts))
    cfg = _config()
    if (Path(cfg['run_dir']) / cfg['arm'] / 'TIMED').exists():
        raise RuntimeError('CUPTI cannot overlap the performance timing marker')


def bootstrap(scheduler):
    """Resolve an actual marker and initialize CUPTI at idle, before release."""
    if _STATE is None:
        raise RuntimeError('stage_activity.install must precede bootstrap')
    if _STATE.get('bootstrapped'):
        return status()
    cfg = _config()
    if cfg.get('role') == 'producer':
        return dict(installed=True, skipped_producer=True)
    _require_idle(scheduler)
    import torch
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False, profile_memory=False, with_stack=False) as profiler:
        torch.cuda._sleep(1)
        torch.cuda.synchronize()
    path = _output(f'activity_marker_startup_rank{_rank()}.json')
    profiler.export_chrome_trace(str(path))
    kernels = [x for x in json.loads(path.read_text())['traceEvents'] if x.get('cat') == 'kernel']
    if len(kernels) != 1:
        raise RuntimeError('Startup marker requires exactly one observed GPU kernel')
    _STATE.update(bootstrapped=True, marker_name=kernels[0]['name'])
    _save(_output(f'activity_marker_signature_rank{_rank()}.json'),
          dict(passed=True, rank=_rank(), name=kernels[0]['name'],
               at_confirmed_scheduler_idle=True, before_profile_client_release=True,
               added_cuda_synchronizations=2, inside_model_forward=False))
    return status()


def arm_profile(scheduler, cohort_id):
    if _STATE is None or not _STATE.get('bootstrapped'):
        raise RuntimeError('CUDA profiler must bootstrap at idle before arming')
    if _STATE['armed']:
        raise RuntimeError('Only one bounded activity cohort per consumer')
    _require_idle(scheduler)
    import torch
    # Starting CUPTI can cause initialization work. Complete a NON-marker fill
    # now, not on the first model forward, before any profile request is released.
    torch.cuda.synchronize()
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                                      record_shapes=False, profile_memory=False,
                                      with_stack=False, with_modules=False)
    profiler.start()
    bootstrap_tensor = torch.empty(1, device='cuda', dtype=torch.int32)
    bootstrap_tensor.fill_(0)
    torch.cuda.synchronize()
    _STATE.update(armed=True, profiler=profiler, cohort_id=str(cohort_id),
                  wall_origin_ns=time.time_ns(), host_origin_ns=time.perf_counter_ns(),
                  bootstrap_tensor=bootstrap_tensor,
                  bootstrap=dict(at_confirmed_scheduler_idle=True,
                                 before_profile_client_release=True,
                                 non_marker_fill_kernel=True, cuda_synchronizations=2,
                                 inside_model_forward=False))
    return status()


def finish_profile(scheduler):
    """Stop/export only after the whole separate profile cohort is idle."""
    if _STATE is None or not _STATE['armed']:
        raise RuntimeError('No armed profile to finish')
    if _STATE['done']:
        return status()
    _require_idle(scheduler)
    state = _STATE
    state['profiler'].stop()
    path = _output(f'activity_trace_rank{_rank()}.json')
    state['profiler'].export_chrome_trace(str(path))
    cfg = _config()
    target = [s for s in state['segments'] if s['phase'].startswith('target_')]
    expected = 'target_verify' if cfg['mtp'] else 'target_decode'
    errors = []
    if len(target) < 2:
        errors.append('At least two target markers are needed for a complete steady cycle')
    if len(target) > MAX_TARGET_FORWARDS or any(s['phase'] != expected for s in target):
        errors.append('Unexpected target phase or unbounded target sample')
    if any(s['batch_size'] != 4 or len(s['request_ids']) != 4 for s in target):
        errors.append('Target profile is not actual local B4')
    if state['errors']:
        errors.extend(state['errors'])
    receipt = dict(schema_version=1, passed=not errors, errors=errors,
                   arm=cfg['arm'], layout=cfg['layout'], mtp=cfg['mtp'], rank=_rank(),
                   topology=state['rank'], cohort_id=state['cohort_id'],
                   target_forward_count=len(target), max_target_forwards=MAX_TARGET_FORWARDS,
                   segments=state['segments'], accepted=state['accepted'],
                   trace=str(path), marker_kernel_name=state['marker_name'],
                   profile_only=True, production_timing_affected=False,
                   host_origin_ns=state['host_origin_ns'], wall_origin_ns=state['wall_origin_ns'],
                   inner_phase_synchronizations=0, inner_phase_events=0,
                   bootstrap=state['bootstrap'],
                   stop_only_at_confirmed_scheduler_idle=True,
                   trace_bound='One separate B8/Q1->64 cohort; at most 12 marked target forwards and 160 coarse API segments',
                   attention_scope='Glm4MoeAttention includes QKV/output projections, attention kernel, RoPE and KV store',
                   accepted_scope='CPU request.output_ids advances after native result processing; no added device readbacks',
                   source_bank_fill_or_model_warmup_timed=False)
    _save(_output(f'activity_receipt_rank{_rank()}.json'), receipt)
    state['done'] = True
    if errors:
        raise RuntimeError('Activity receipt failed: ' + '; '.join(errors))
    return receipt


def install():
    global _STATE
    if _STATE is not None:
        return
    import torch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    from sglang.srt.models.glm4_moe import Glm4MoeAttention, Glm4MoeMLP, Glm4MoeSparseMoeBlock
    from sglang.srt.layers.moe.moe_runner.humming import HummingRunnerCore
    from sglang.srt.layers.moe.token_dispatcher.deepep import _DeepEPDispatcherImplLowLatency
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.managers.scheduler import Scheduler
    state = dict(armed=False, bootstrapped=False, profiler=None, done=False,
                 rank=None, target_count=0, segments=[], accepted=[], errors=[],
                 pending=defaultdict(deque), iterations=defaultdict(int), calibrations=0)
    _STATE = state

    def scope(cls, name, label, shared=False):
        descriptor = inspect.getattr_static(cls, name)
        original = descriptor.__func__ if isinstance(descriptor, (staticmethod, classmethod)) else descriptor
        @wraps(original)
        def wrapped(*args, **kwargs):
            if not _CAL.get():
                return original(*args, **kwargs)
            actual = 'expert_shared' if label == 'dense_ffn' and _SHARED.get() else label
            token = _SHARED.set(True) if shared else None
            try:
                with torch.profiler.record_function('stage-cal:' + actual):
                    return original(*args, **kwargs)
            finally:
                if token is not None:
                    _SHARED.reset(token)
        setattr(cls, name, staticmethod(wrapped) if isinstance(descriptor, staticmethod) else
                classmethod(wrapped) if isinstance(descriptor, classmethod) else wrapped)
    for cls, name, label in [
        (Glm4MoeAttention, 'forward', 'attention'),
        (Glm4MoeMLP, 'forward', 'dense_ffn'),
        (HummingRunnerCore, '_run_grouped_masked_gemm', 'expert_routed'),
        (HummingRunnerCore, 'apply_masked_activation', 'expert_routed'),
        (_DeepEPDispatcherImplLowLatency, 'dispatch_a', 'deepep_dispatch'),
        (_DeepEPDispatcherImplLowLatency, 'dispatch_b', 'deepep_dispatch'),
        (_DeepEPDispatcherImplLowLatency, 'combine_a', 'deepep_combine'),
        (_DeepEPDispatcherImplLowLatency, 'combine_b', 'deepep_combine')]:
        scope(cls, name, label)
    scope(Glm4MoeSparseMoeBlock, '_forward_shared_experts', 'expert_shared', shared=True)
    for name in ('all_reduce', 'reduce_scatter_along_dim', 'reduce_scatter_tensor', 'reduce_scatter',
                 'reduce_scatterv', 'all_gather_into_tensor', 'all_gather', 'all_gatherv'):
        if hasattr(GroupCoordinator, name):
            scope(GroupCoordinator, name, 'nccl_tp')
    for name in ('send_tensor_dict', 'recv_tensor_dict', 'recv_tensor_dict_into'):
        if hasattr(GroupCoordinator, name):
            scope(GroupCoordinator, name, 'pp_boundary')
    # Existing PP P2P startup traffic (two opposite directions) is a useful
    # independent calibration witness; no new send/receive is introduced.
    import pp_p2p_warm
    original_pp_warm = pp_p2p_warm.warm
    for name in ('isend', 'irecv'):
        scope(torch.distributed, name, 'pp_boundary')
    @wraps(original_pp_warm)
    def pp_warm(*args, **kwargs):
        if _rank() not in (0, 4) or _config().get('role') == 'producer':
            return original_pp_warm(*args, **kwargs)
        token = _CAL.set(True)
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=False, profile_memory=False, with_stack=False) as prof:
                result = original_pp_warm(*args, **kwargs)
            prof.export_chrome_trace(str(_output(f'calibration_pp_p2p_rank{_rank()}.json')))
            return result
        finally:
            _CAL.reset(token)
    pp_p2p_warm.warm = pp_warm

    original_capture = FullCudaGraphBackend.capture_one
    @wraps(original_capture)
    def capture(self, shape_key, forward_fn, *args, **kwargs):
        if _rank() not in (0, 4) or _config().get('role') == 'producer':
            return original_capture(self, shape_key, forward_fn, *args, **kwargs)
        ordinal = state['calibrations']
        state['calibrations'] += 1
        calls = [0]
        runner = self._cuda_graph_runner.model_runner
        role = 'draft' if runner.is_draft_worker else 'target'
        def calibrated():
            if torch.cuda.is_current_stream_capturing():
                return forward_fn()
            index = calls[0]
            calls[0] += 1
            token = _CAL.set(True)
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=False, profile_memory=False, with_stack=False) as prof:
                    # Explicit complement scope: routing, norms, residuals and
                    # graph bookkeeping are real model GPU work, not attention
                    # or routed expert GEMMs. Inner calibrated scopes override.
                    with torch.profiler.record_function('stage-cal:other_gpu'):
                        result = forward_fn()
                path = _output(f'calibration_rank{_rank()}_{role}_graph{ordinal}_warm{index}.json')
                prof.export_chrome_trace(str(path))
                return result
            finally:
                _CAL.reset(token)
        return original_capture(self, shape_key, calibrated, *args, **kwargs)
    FullCudaGraphBackend.capture_one = capture

    def segment(function, phase, batch):
        active = state['armed'] and not state['done'] and state['profiler'] is not None
        if not active or _DEPTH.get() or state['target_count'] >= MAX_TARGET_FORWARDS:
            return function()
        meta = batch_metadata(batch, _FRAME.get())
        is_target = phase in ('target_decode', 'target_verify')
        if is_target:
            expected = 'target_verify' if _config()['mtp'] else 'target_decode'
            if phase != expected:
                state['errors'].append('Unexpected target phase ' + phase)
            state['target_count'] += 1
        if len(state['segments']) >= MAX_SEGMENTS:
            raise RuntimeError('Bounded activity segment storage exceeded')
        index = len(state['segments'])
        key = tuple(sorted(meta['request_ids']))
        iteration = state['iterations'][key]
        if is_target:
            state['iterations'][key] += 1
            state['pending'][key].append(index)
        token = _DEPTH.set(1)
        row = dict(index=index, phase=phase, target_ordinal=state['target_count'],
                   request_iteration=iteration, **meta,
                   cpu_start_ns=time.perf_counter_ns(),
                   start_stream=int(torch.cuda.current_stream().cuda_stream))
        state['segments'].append(row)
        torch.cuda._sleep(1)
        try:
            return function()
        finally:
            torch.cuda._sleep(1)
            row.update(cpu_end_ns=time.perf_counter_ns(),
                       end_stream=int(torch.cuda.current_stream().cuda_stream))
            _DEPTH.reset(token)

    original_run = Scheduler.run_batch
    @wraps(original_run)
    def run(self, batch, *args, **kwargs):
        if not state['armed'] or state['done']:
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
        phase = target_phase(batch, self.is_draft_worker)
        if phase is None:
            return original_forward(self, batch, *args, **kwargs)
        return segment(lambda: original_forward(self, batch, *args, **kwargs), phase, batch)
    ModelRunner.forward = forward

    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
    def draft_wrapper(method, phase):
        @wraps(method)
        def wrapped(self, batch, *args, **kwargs):
            return segment(lambda: method(self, batch, *args, **kwargs), phase, batch)
        return wrapped
    for name, phase in [('draft', 'draft'), ('_draft_extend_for_decode', 'draft_extend')]:
        setattr(EagleDraftWorker, name, draft_wrapper(getattr(EagleDraftWorker, name), phase))

    original_process = Scheduler.process_batch_result
    @wraps(original_process)
    def process(self, batch, result, *args, **kwargs):
        key = tuple(sorted(r.rid for r in getattr(batch, 'reqs', [])))
        pending = state['pending'].get(key)
        collect = state['armed'] and not state['done'] and bool(pending) and batch.forward_mode.is_decode()
        before = {r.rid: len(r.output_ids) for r in batch.reqs} if collect else {}
        value = original_process(self, batch, result, *args, **kwargs)
        if collect:
            index = pending.popleft()
            advances = {r.rid: len(r.output_ids) - before[r.rid] for r in batch.reqs}
            state['accepted'].append(dict(target_segment_index=index,
                                          output_tokens_by_request=advances,
                                          output_tokens=sum(advances.values()),
                                          cpu_time_ns=time.perf_counter_ns()))
        return value
    Scheduler.process_batch_result = process
