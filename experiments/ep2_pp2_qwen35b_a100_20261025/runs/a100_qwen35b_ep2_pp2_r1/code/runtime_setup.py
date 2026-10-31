"""Qualify native EP mode before the native KV capacity allocator.

Only initialization order changes. Reuse the real model dispatchers and the
native process-local buffer singleton; no dummy forward, alternative group,
dispatch mode change, transport change, or replacement kernel is introduced.
Imports remain CPU-safe until the installed method actually runs.
The A100 standard-dispatch/allreduce EP path has no DeepEP allocation; it is
validated without importing torch or the optional DeepEP extension.
"""
from __future__ import annotations

from functools import wraps
import gc
import importlib
import json
import logging
import os
from pathlib import Path
import tempfile


FIELDS = ('group', 'hidden_size', 'params_bytes', 'deepep_mode',
          'num_max_dispatch_tokens_per_rank', 'num_experts')


def dispatcher_parameters(model, dispatcher_class, expected_ep_size):
    """Find real dispatchers behind MaybeTbo, and reject mixed singleton users."""
    implementations = []
    seen = set()

    def visit(dispatcher):
        if dispatcher is None or id(dispatcher) in seen:
            return
        seen.add(id(dispatcher))
        if isinstance(dispatcher, dispatcher_class):
            for name in ('_normal_dispatcher', '_low_latency_dispatcher'):
                value = getattr(dispatcher, name, None)
                if value is not None:
                    implementations.append(value)
        else:
            for inner in getattr(dispatcher, '_inners', ()):
                visit(inner)

    for module in model.modules():
        visit(getattr(module, 'dispatcher', None))
    if not implementations:
        if expected_ep_size != 1:
            raise RuntimeError('EP layout has no native DeepEP dispatcher to initialize')
        return None
    first = implementations[0]
    values = tuple(getattr(first, field) for field in FIELDS)
    if first.group.size() != expected_ep_size or expected_ep_size <= 1:
        raise RuntimeError('Native DeepEP group differs from the planned EP layout')
    for other in implementations[1:]:
        observed = tuple(getattr(other, field) for field in FIELDS)
        if observed[0] is not values[0] or observed[1:] != values[1:]:
            raise RuntimeError('Model dispatchers disagree on the shared DeepEP buffer')
    return values


def standard_ep_parameters(model, expected_ep_size):
    """Prove the standard runner stores distinct experts, not expert TP shards."""
    if type(expected_ep_size) is not int or expected_ep_size < 1:
        raise RuntimeError('A positive native expert parallel size is required')
    rows = []
    for module in model.modules():
        if not hasattr(module, '_num_local_routed'):
            continue
        global_experts = getattr(module, '_num_global_routed', None)
        local_experts = module._num_local_routed
        if (getattr(module, 'moe_ep_size', None) != expected_ep_size
                or getattr(module, 'moe_tp_size', None) != 1
                or type(global_experts) is not int or global_experts <= 0
                or global_experts % expected_ep_size
                or local_experts != global_experts // expected_ep_size):
            raise RuntimeError('Standard EP must preserve native routed-expert partitioning with expert TP1')
        rows.append(dict(global_routed_experts=global_experts,
                         local_routed_experts=local_experts, expert_tp_size=1))
    if not rows:
        raise RuntimeError('Standard EP model has no native sharded MoE layer to qualify')
    return rows


def fp8_model_parameters(model):
    """Attest every actual routed layer, never a toy-only model substitute."""
    routed, dense = [], []
    for name, module in model.named_modules():
        method = getattr(module, 'quant_method', None)
        prepared = getattr(module, '_qwen_a100_fp8_marlin_prepared', False)
        if hasattr(module, '_num_local_routed'):
            if (not prepared or method is None
                    or method.__class__.__name__ != 'Fp8MarlinMoEMethod'
                    or method.__class__.__module__ != 'fp8_marlin_adapter'
                    or getattr(module, 'num_fused_shared_experts', 0) != 0):
                raise RuntimeError('Every native routed expert layer must use the isolated FP8 Marlin method')
            geometry = module._qwen_a100_fp8_marlin_geometry
            routed.append(dict(name=name, ep_size=geometry.ep_size,
                global_experts=geometry.global_experts, local_experts=geometry.local_experts,
                hidden_size=geometry.hidden_size, intermediate_size=geometry.intermediate_size))
        elif method is not None:
            if method.__class__.__module__ == 'fp8_marlin_adapter' or prepared:
                raise RuntimeError('The MoE adapter changed a native dense/shared expert linear')
            dense.append(dict(name=name, quant_method=method.__class__.__name__,
                              method_module=method.__class__.__module__))
    if not routed or not dense:
        raise RuntimeError('FP8 qualification must observe full model routed and native dense layers')
    return routed, dense


def qualify_fp8_model(runner):
    """Run one tiny real kernel fixture per native rank after full model load.

    This hook runs before native KV allocation; temporary fixture buffers are
    released before capacity profiling. It never changes the model backend.
    """
    if os.environ.get('QWEN_A100_FP8_MOE_MARLIN') != '1':
        return None
    adapter = importlib.import_module('fp8_marlin_adapter')
    expected_sha = os.environ.get('QWEN_A100_FP8_ADAPTER_SHA256')
    if expected_sha != adapter.source_sha256():
        raise RuntimeError('The enabled FP8 adapter differs from the frozen source')
    if adapter.qualification_mode() != 'qualification':
        # Formal execution verifies its own receipt and never reruns the toy.
        return None
    owner = json.loads(os.environ.get('QWEN_A100_FP8_QUALIFICATION_OWNER_JSON', '{}'))
    target = Path(os.environ.get('QWEN_A100_FP8_QUALIFICATION_ATTESTATION_DIR', ''))
    if (set(owner) != {'run_id', 'layout', 'replica'} or not target.is_absolute()
            or runner.ps.tp_size != runner.ps.moe_ep_size
            or runner.ps.tp_rank != runner.ps.moe_ep_rank
            or runner.ps.moe_ep_size not in (2, 4)):
        raise RuntimeError('FP8 qualification requires its frozen owner and actual TP=EP2/EP4 context')
    routed, dense = fp8_model_parameters(runner.model)
    expected_layers = int(os.environ.get('QWEN_A100_FP8_MODEL_MOE_LAYERS', '0'))
    if expected_layers != 48 or len(routed) != expected_layers:
        raise RuntimeError('The complete pinned 48-layer Qwen FP8 model must be loaded')
    from sglang.srt.distributed import get_tp_group
    fixture = adapter.correctness_fixture(ep_size=runner.ps.moe_ep_size,
        ep_rank=runner.ps.moe_ep_rank, all_reduce=get_tp_group().all_reduce, device='cuda')
    import torch
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    proof = dict(status='passed', owner=owner, tp_rank=runner.ps.tp_rank,
        ep_rank=runner.ps.moe_ep_rank, ep_size=runner.ps.moe_ep_size,
        adapter_sha256=expected_sha, engine_commit=adapter.ENGINE_COMMIT,
        model_moe_layers=routed, native_dense_layers=dense,
        expected_model_moe_layers=expected_layers,
        all_model_moe_layers_prepared=True, native_dense_unchanged=True,
        fixture=fixture, fixture_temporary_cuda_allocations_released_before_KV=True)
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = target / f'rank{runner.ps.tp_rank}.json'
    with path.open('x') as handle:
        os.chmod(path, 0o600)
        json.dump(proof, handle, indent=2, sort_keys=True)
        handle.write('\n')
    logging.getLogger(__name__).info('QWEN_A100_FP8_FULL_MODEL_FIXTURE_READY %s',
        json.dumps(dict(rank=runner.ps.tp_rank, ep_size=runner.ps.moe_ep_size,
                        layer_count=len(routed), adapter_sha256=expected_sha), sort_keys=True))
    return proof


def preinitialize(runner):
    backend = runner.server_args.moe_a2a_backend
    backend = getattr(backend, 'value', backend)
    if backend == 'none':
        rows = standard_ep_parameters(runner.model, runner.ps.moe_ep_size)
        fp8_proof = qualify_fp8_model(runner)
        runner._qwen_pre_kv_deepep = dict(initialized=False,
            reason='native_standard_dispatch_nccl_allreduce_no_deepep',
            moe_a2a_backend='none', ep_size=runner.ps.moe_ep_size,
            native_expert_layers=rows,
            fp8_marlin_qualification=(dict(status=fp8_proof['status'],
                adapter_sha256=fp8_proof['adapter_sha256'],
                model_moe_layer_count=len(fp8_proof['model_moe_layers']))
                if fp8_proof is not None else None))
        logging.getLogger(__name__).info('QWEN_PRE_KV_STANDARD_EP_READY %s',
            json.dumps(runner._qwen_pre_kv_deepep, sort_keys=True))
        return
    if backend != 'deepep':
        raise RuntimeError('Unknown native EP transport; refusing unqualified KV calibration')
    import torch
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer, DeepEPDispatcher

    values = dispatcher_parameters(runner.model, DeepEPDispatcher, runner.ps.moe_ep_size)
    if values is None:
        runner._qwen_pre_kv_deepep = {'initialized': False, 'reason': 'EP1'}
        return
    if DeepEPBuffer._state().buffer is not None:
        raise RuntimeError('Unexpected preexisting native DeepEP buffer before KV qualification')

    def memory():
        free, total = torch.cuda.mem_get_info()
        return dict(free_bytes=free, total_bytes=total,
                    torch_allocated_bytes=torch.cuda.memory_allocated(),
                    torch_reserved_bytes=torch.cuda.memory_reserved())

    logger = logging.getLogger(__name__)
    before = memory()
    # Return cached loading slabs before native cuMemCreate; native KV profiling
    # also empties this cache later, but that would be too late for DeepEP.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    reclaimed = memory()
    logger.info('QWEN_PRE_KV_DEEPEP_BEGIN %s', json.dumps(dict(
        ep_size=values[0].size(), hidden_size=values[1], param_bytes=values[2],
        mode=str(values[3]), max_dispatch_tokens_per_rank=values[4],
        num_experts=values[5], before=before, after_empty_cache=reclaimed), sort_keys=True))
    buffer = DeepEPBuffer.get_deepep_buffer(*values)
    torch.cuda.synchronize()
    runner._qwen_pre_kv_deepep = dict(initialized=True, ep_size=values[0].size(),
        num_nvl_bytes=buffer.num_nvl_bytes, num_rdma_bytes=buffer.num_rdma_bytes,
        before=before, after_empty_cache=reclaimed, after_initialize=memory())
    logger.info('QWEN_PRE_KV_DEEPEP_READY %s',
                json.dumps(runner._qwen_pre_kv_deepep, sort_keys=True))


def install_on(model_runner_class, initialize=preinitialize):
    if model_runner_class.__dict__.get('_qwen_pre_kv_deepep_installed', False):
        return
    original = model_runner_class.alloc_memory_pool

    @wraps(original)
    def alloc_memory_pool(self, *args, **kwargs):
        if not hasattr(self, '_qwen_pre_kv_deepep'):
            initialize(self)
        return original(self, *args, **kwargs)

    model_runner_class.alloc_memory_pool = alloc_memory_pool
    model_runner_class._qwen_pre_kv_deepep_installed = True


def _graph_evidence_enabled():
    return (os.environ.get('QWEN_A100_FP8_MOE_MARLIN') == '1'
            and os.environ.get('QWEN_A100_FP8_MOE_MARLIN_QUALIFICATION') == '1')


def _graph_shape(key):
    """Read only host-side ShapeKey fields; never inspect a CUDA tensor."""
    size = getattr(key, 'size', None)
    if type(size) is not int or size <= 0:
        raise RuntimeError('Native Decode Graph registry has an invalid shape key')
    result = dict(size=size)
    for name in ('stream_idx', 'variant_label', 'dsa_variant'):
        value = getattr(key, name, None)
        expected = int if name == 'stream_idx' else str
        if value is not None and type(value) is not expected:
            raise RuntimeError('Native Decode Graph key must contain only host metadata')
        result[name] = value
    return result


def _write_graph_evidence(state):
    """Atomic CPU-only evidence publication, outside the native graph call."""
    target = state['path']
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + '.', dir=target.parent)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(state['proof'], handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _record_graph_capture(model_runner, decode_runner_class):
    """Attest actual completed native captures, not requested CLI settings."""
    decode = getattr(model_runner, 'decode_cuda_graph_runner', None)
    if type(decode) is not decode_runner_class:
        raise RuntimeError('FP8 qualification requires the native DecodeCudaGraphRunner')
    backend = getattr(decode, 'backend', None)
    registry = getattr(backend, '_graphs', None)
    if (not isinstance(registry, dict) or not registry
            or any(not callable(getattr(graph, 'replay', None)) for graph in registry.values())):
        raise RuntimeError('FP8 qualification requires nonempty actual native Decode Graph captures')
    capture_bs = list(getattr(decode, 'capture_bs', ()))
    if not capture_bs or any(type(value) is not int or value <= 0 for value in capture_bs):
        raise RuntimeError('Native Decode Graph capture batch sizes are invalid')
    owner = json.loads(os.environ.get('QWEN_A100_FP8_QUALIFICATION_OWNER_JSON', '{}'))
    target = Path(os.environ.get('QWEN_A100_FP8_QUALIFICATION_ATTESTATION_DIR', ''))
    ps = model_runner.ps
    adapter = importlib.import_module('fp8_marlin_adapter')
    expected_sha = os.environ.get('QWEN_A100_FP8_ADAPTER_SHA256')
    pre_kv = getattr(model_runner, '_qwen_pre_kv_deepep', {}).get('fp8_marlin_qualification', {})
    if (set(owner) != {'run_id', 'layout', 'replica'} or not target.is_absolute()
            or ps.tp_size != ps.moe_ep_size or ps.tp_rank != ps.moe_ep_rank
            or ps.moe_ep_size not in (2, 4) or expected_sha != adapter.source_sha256()
            or pre_kv.get('status') != 'passed'
            or pre_kv.get('adapter_sha256') != expected_sha
            or pre_kv.get('model_moe_layer_count') != 48):
        raise RuntimeError('Native Decode Graph evidence lacks its full-model qualification owner')
    shapes = [_graph_shape(key) for key in registry]
    shapes.sort(key=lambda row: json.dumps(row, sort_keys=True))
    proof = dict(version=1, status='captured', owner=owner, tp_rank=ps.tp_rank,
        ep_rank=ps.moe_ep_rank, ep_size=ps.moe_ep_size, adapter_sha256=expected_sha,
        engine_commit=adapter.ENGINE_COMMIT, native_runner_class=type(decode).__name__,
        native_runner_module=type(decode).__module__,
        native_backend_class=type(backend).__name__, native_backend_module=type(backend).__module__,
        capture_batch_sizes=capture_bs, actual_capture_shapes=shapes,
        actual_capture_count=len(registry), actual_captures_nonempty=True,
        successful_native_decode_replays=0, replay_count=0,
        actual_model_decode_graph_replay_verified=False, numerical_equivalence_checked=False,
        scope='actual_model_decode_graph_execution_only')
    state = dict(path=target / f'rank{ps.tp_rank}.graph.json', proof=proof,
                 backend=backend, registry=registry)
    if state['path'].exists():
        raise RuntimeError('Refusing to reuse preexisting native Decode Graph evidence')
    decode._qwen_fp8_decode_graph_evidence = state
    _write_graph_evidence(state)


def install_graph_evidence_on(model_runner_class, decode_graph_runner_class):
    """Qualification-only transparent native capture/replay observation.

    The pinned runner's public replay entry is ``execute``, not ``replay``.
    Its unmodified implementation calls backend.replay after loading the batch.
    No forward, collective, stream, tensor, or synchronization is replaced.
    """
    if not _graph_evidence_enabled():
        return False
    if (model_runner_class.__dict__.get('_qwen_fp8_graph_evidence_installed', False)
            or decode_graph_runner_class.__dict__.get('_qwen_fp8_graph_evidence_installed', False)):
        return False
    original_init = model_runner_class.init_cuda_graphs
    original_execute = decode_graph_runner_class.execute

    @wraps(original_init)
    def init_cuda_graphs(self, *args, **kwargs):
        result = original_init(self, *args, **kwargs)
        if _graph_evidence_enabled():
            _record_graph_capture(self, decode_graph_runner_class)
        return result

    @wraps(original_execute)
    def execute(self, *args, **kwargs):
        result = original_execute(self, *args, **kwargs)
        state = getattr(self, '_qwen_fp8_decode_graph_evidence', None)
        if _graph_evidence_enabled() and state is not None:
            batch = args[0] if args else kwargs.get('forward_batch')
            mode = getattr(batch, 'forward_mode', None)
            if mode is not None and mode.is_decode():
                key = getattr(self, '_replay_graph_key', None)
                if (self.backend is not state['backend']
                        or getattr(self.backend, '_graphs', None) is not state['registry']
                        or key not in state['registry']):
                    raise RuntimeError('Successful native Decode call did not use its attested capture registry')
                proof = state['proof']
                proof['replay_count'] += 1
                proof['successful_native_decode_replays'] = proof['replay_count']
                proof['last_replayed_shape'] = _graph_shape(key)
                proof['actual_model_decode_graph_replay_verified'] = True
                proof['status'] = 'passed'
                _write_graph_evidence(state)
        return result

    model_runner_class.init_cuda_graphs = init_cuda_graphs
    decode_graph_runner_class.execute = execute
    model_runner_class._qwen_fp8_graph_evidence_installed = True
    decode_graph_runner_class._qwen_fp8_graph_evidence_installed = True
    return True


def install():
    if os.environ.get('QWEN_A100_FP8_MOE_MARLIN') == '1':
        importlib.import_module('fp8_marlin_adapter').install()
    native = importlib.import_module('sglang.srt.model_executor.model_runner')
    install_on(native.ModelRunner)
    if _graph_evidence_enabled():
        graph = importlib.import_module(
            'sglang.srt.model_executor.runner.decode_cuda_graph_runner')
        install_graph_evidence_on(native.ModelRunner, graph.DecodeCudaGraphRunner)
