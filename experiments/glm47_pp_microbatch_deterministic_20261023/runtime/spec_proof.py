"""CPU-only counters for actual verify, draft and native CUDA Graph submissions.

No CUDA events, device readbacks, synchronization or log I/O in measured work.
Submission proof is not a numerical-correctness or GPU-completion assertion.
"""
from collections import Counter
from contextvars import ContextVar
import copy
from functools import wraps

_FRAME = ContextVar('glm_spec_proof', default=None)
_STATE = None

def cohort(batch):
    requests = getattr(batch, 'reqs', None)
    ids = [r.rid for r in requests] if requests is not None else getattr(batch, 'rids', None)
    if not isinstance(ids, (list, tuple)) or not ids: return None
    bases = []
    for rid in ids:
        if not isinstance(rid, str) or not rid.startswith(('warm-', 'timed-', 'control-', 'profile-', 'prepare-', 'repeat-')): return None
        base, _, index = rid.rpartition('-')
        if not index.isdecimal() or not 0 <= int(index) < 8: return None
        bases.append(base)
    return bases[0] if len(set(bases)) == 1 else None

def _row(name):
    if name not in _STATE['cohorts']:
        if len(_STATE['cohorts']) >= 64: raise RuntimeError('Spec proof bounded cohort storage exceeded')
        _STATE['cohorts'][name] = dict(forwards=Counter(), graph_flags=Counter(), replays=Counter(),
                                      batch_sizes=Counter(), widths=Counter(), phases=Counter())
    return _STATE['cohorts'][name]

def install():
    global _STATE
    if _STATE is not None: return
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
    from sglang.srt.runtime_context import get_parallel
    _STATE = dict(cohorts={}, rank=None)
    original_run = Scheduler.run_batch
    @wraps(original_run)
    def run(self, batch, *args, **kwargs):
        name = cohort(batch)
        token = _FRAME.set((name, 'target'))
        try: return original_run(self, batch, *args, **kwargs)
        finally: _FRAME.reset(token)
    Scheduler.run_batch = run

    def phase_wrapper(method, phase):
        @wraps(method)
        def wrapped(self, batch, *args, **kwargs):
            name = cohort(batch)
            if name: _row(name)['phases'][phase] += 1
            token = _FRAME.set((name, phase))
            try: return method(self, batch, *args, **kwargs)
            finally: _FRAME.reset(token)
        return wrapped
    for method, phase in [('draft', 'draft'), ('_draft_extend_for_decode', 'draft_extend'),
                          ('_draft_extend_for_prefill', 'draft_prefill')]:
        setattr(EagleDraftWorker, method, phase_wrapper(getattr(EagleDraftWorker, method), phase))

    original_forward = ModelRunner.forward
    @wraps(original_forward)
    def forward(self, batch, *args, **kwargs):
        frame = _FRAME.get()
        name = frame[0] if frame else cohort(batch)
        if not name: return original_forward(self, batch, *args, **kwargs)
        phase = frame[1] if frame else ('draft' if self.is_draft_worker else 'target')
        if _STATE['rank'] is None:
            p = get_parallel()
            _STATE['rank'] = dict(world_rank=p.world_rank, pp_rank=p.pp_rank, tp_rank=p.tp_rank)
        mode = getattr(batch.forward_mode, 'name', str(batch.forward_mode))
        key = phase + ':' + mode
        row = _row(name)
        row['forwards'][key] += 1
        if type(batch.batch_size) is int: row['batch_sizes'][str(batch.batch_size)] += 1
        token = _FRAME.set((name, phase))
        try: result = original_forward(self, batch, *args, **kwargs)
        finally: _FRAME.reset(token)
        row['graph_flags'][key + ':' + str(getattr(result, 'can_run_graph', None))] += 1
        return result
    ModelRunner.forward = forward

    original_replay = FullCudaGraphBackend.replay
    @wraps(original_replay)
    def replay(self, shape_key, batch, **kwargs):
        result = original_replay(self, shape_key, batch, **kwargs)
        frame = _FRAME.get()
        if frame and frame[0]:
            row = _row(frame[0]); phase = frame[1]
            row['replays'][phase] += 1
            size = getattr(shape_key, 'size', None)
            row['widths'][phase + ':' + str(size if type(size) is int else None)] += 1
        return result
    FullCudaGraphBackend.replay = replay

def snapshot(reset=False):
    if _STATE is None: return dict(installed=False)
    result = copy.deepcopy(_STATE)
    for row in result['cohorts'].values():
        for key, value in row.items(): row[key] = dict(value)
    result.update(installed=True, gpu_synchronizations_added=0, cuda_events_added=0,
                  device_tensor_readbacks_added=0,
                  proof_boundary='Successful native Graph replay submission and real request CPU identities; not numerical correctness')
    if reset: _STATE['cohorts'].clear()
    return result

def validate_receipts(receipts, layout, cohort_prefix='timed-'):
    if len(receipts) != 8 or sorted(r['rank'] for r in receipts) != list(range(8)):
        raise RuntimeError('Missing rank proof')
    rows = []
    for receipt in receipts:
        graph = receipt['graph']
        cohorts = [v for k, v in graph.get('cohorts', {}).items() if k.startswith(cohort_prefix)]
        if not graph.get('installed') or len(cohorts) != 1: raise RuntimeError('Missing actual timed MTP cohort')
        row = cohorts[0]
        verify = sum(v for k, v in row['forwards'].items() if k.startswith('target:') and 'TARGET_VERIFY' in k)
        if not verify or not row['replays'].get('target', 0):
            raise RuntimeError('Target verify or actual native Graph replay missing: '+str(row))
        should_draft = layout == 'ep8' or receipt['rank'] >= 4
        draft = row['phases'].get('draft', 0)
        if should_draft != bool(draft): raise RuntimeError('MTP draft placement mismatch')
        rows.append(dict(rank=receipt['rank'], target_verify_forwards=verify,
                         actual_native_replays=row['replays'], draft_calls=draft,
                         batch_sizes=row['batch_sizes'], replay_sizes=row['widths']))
    return dict(passed=True, ranks=rows, numerical_correctness_claim=False)
