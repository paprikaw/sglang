"""Bounded, startup-prepared native Decode runners for resident elastic PP.

This deliberately retains native storage, not VMM pages. Each local layer
interval owns its attention metadata/backend and Graph pool. Runtime selection
never captures, and fails closed if any external tensor binding has changed.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import torch

from sglang.srt.elastic_pp.autoscaling import parse_elastic_pp_partitions
from sglang.srt.elastic_pp.graph_bank import binding_signature
from sglang.srt.elastic_pp.topology import PPTopology

logger = logging.getLogger(__name__)

_BACKEND_FIELDS = (
    "attn_backend",
    "decode_attn_backend",
    "decode_attn_backend_group",
    "shared_read_done_event",
)


def validate_graph_partitions(args, initial):
    partitions = parse_elastic_pp_partitions(args.elastic_pp_graph_partitions)
    if getattr(args, "elastic_pp_hicache_allocator", "mmap") == "pytorch" and (
        not args.enable_hierarchical_cache or args.hicache_storage_backend is not None
    ):
        raise ValueError("elastic pinned allocator requires L2-only HiCache enabled")
    if len(partitions) > 4:
        raise ValueError("prepared Decode Graph supports at most four partitions")
    if args.disable_cuda_graph:
        raise ValueError("prepared Decode Graph requires CUDA Graph enabled")
    if (
        args.elastic_pp_kv_backend != "native"
        or args.elastic_pp_transition_backing != "overlap"
    ):
        raise ValueError("prepared Decode Graph requires native/overlap backing")
    if args.enable_lora or args.enable_two_batch_overlap or args.enable_pdmux:
        raise ValueError("prepared Decode Graph does not support LoRA, TBO or PDMux")
    if tuple(initial) not in partitions:
        raise ValueError("prepared Decode Graph must include the initial partition")
    for counts in partitions:
        if len(counts) != args.pp_size or sum(counts) != sum(initial):
            raise ValueError(
                "prepared Decode Graph partitions must cover the same resident group and model"
            )
        if not counts[0] or not counts[-1]:
            raise ValueError(
                "prepared Decode Graph requires fixed first and last resident ranks"
            )
    if args.enable_elastic_pp_autoscaling:
        frontier = parse_elastic_pp_partitions(args.elastic_pp_autoscale_partitions)
        if not set(frontier) <= set(partitions):
            raise ValueError(
                "autoscaling frontier must be included in prepared Graph partitions"
            )
    return partitions


def _tensors(value, prefix, result):
    if isinstance(value, torch.Tensor):
        result[prefix] = value
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _tensors(item, f"{prefix}.{index}", result)
    elif isinstance(value, dict):
        for name, item in value.items():
            _tensors(item, f"{prefix}.{name}", result)


def external_bindings(mr, runner):
    """Mutable owners outside the private Graph pool, checked at every handoff.

    The retained runner owns capture inputs/outputs; its independent backend
    owns attention metadata and the owning CUDAGraph references to workspace.
    Native global-layer caches keep detached parameter and KV storage alive.
    """
    start, end = mr.model.start_layer, mr.model.end_layer

    def active(name):
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        return match is None or start <= int(match[1]) < end

    result = {f"weight.{n}": t for n, t in mr.model.named_parameters() if active(n)}
    result.update({f"buffer.{n}": t for n, t in mr.model.named_buffers() if active(n)})
    req = mr.req_to_token_pool
    for name in ("req_to_token", "req_index_to_mamba_index_mapping", "req_generation"):
        _tensors(getattr(req, name, None), f"request.{name}", result)
    pool = mr.token_to_kv_pool
    full = getattr(pool, "full_kv_pool", pool)
    for name in (
        "k_buffer",
        "v_buffer",
        "k_data_ptrs",
        "v_data_ptrs",
        "data_ptrs",
        "data_strides",
    ):
        _tensors(getattr(full, name, None), f"kv.{name}", result)
    if hasattr(req, "mamba_pool"):
        state = req.mamba_pool.mamba_cache
        _tensors(state.conv, "gdn.conv", result)
        _tensors(state.temporal, "gdn.temporal", result)
    buffers = getattr(runner, "buffers", None)
    if buffers is not None:
        for name, value in vars(buffers).items():
            _tensors(value, f"input.{name}", result)
    return result


@dataclass
class PreparedDecodeRunner:
    runner: object
    backend_fields: dict
    pool_id: object
    bindings: dict
    signature: tuple
    capture_seconds: float
    weight_bindings: dict


class PreparedDecodeGraphBank:
    def __init__(self, mr, partitions):
        self.partitions = tuple(partitions)
        self.rank = mr.ps.pp_rank
        self.entries = {}
        self.building = True
        self.startup_seconds = None
        self.startup_allocated_delta_bytes = None
        self.runtime_selections = 0
        self.resident_weights = {}
        self.host_allocator_warmups = []
        self.torch_host_allocator = (
            getattr(
                getattr(mr, "server_args", None), "elastic_pp_hicache_allocator", "mmap"
            )
            == "pytorch"
        )

    @staticmethod
    def _weight_identity(tensor):
        # Inference tensors have no version counter. Public weight updates are
        # rejected for this immutable bank; normal tensors additionally detect
        # accidental in-place writes without reading or hashing GPU contents.
        try:
            version = tensor._version
        except RuntimeError:
            version = None
        return binding_signature({"tensor": tensor}), version

    def reuse_weights(self, mr, transition):
        """Validate resident target weights before declaring new layers loaded.

        Startup must still transfer every layer normally. Runtime never writes
        cached parameters: a different address or version invalidates the bank.
        """
        if self.building:
            raise RuntimeError("prepared weights cannot be reused before sealing")
        self.check_topology(transition.target)
        target = transition.target.layer_range(self.rank)
        states = mr.export_elastic_pp_layer_state(range(target.start, target.end))
        for layer_id, values in states.items():
            expected = self.resident_weights.get(layer_id)
            if expected is None or set(values) != set(expected):
                raise RuntimeError("prepared weight state is missing or changed")
            if any(
                self._weight_identity(tensor) != expected[name]
                for name, tensor in values.items()
            ):
                raise RuntimeError("prepared weight storage or version changed")
        source = transition.source.layer_range(self.rank)
        incoming = set(range(target.start, target.end)) - set(
            range(source.start, source.end)
        )
        mr.mark_elastic_pp_layers_loaded(
            generation=transition.target.generation, layer_ids=incoming
        )
        return sum(
            t.numel() * t.element_size()
            for layer_id in incoming
            for t in states[layer_id].values()
        )

    @staticmethod
    def key(mr):
        return (mr.model.start_layer, mr.model.end_layer)

    def check_topology(self, topology):
        counts = tuple(r.size for r in topology.layer_ranges)
        if counts not in self.partitions:
            raise ValueError("target PP partition was not prepared at startup")

    def remember(self, mr, capture_seconds):
        from sglang.srt.model_executor.runner_utils.pool import (
            get_global_graph_memory_pool,
        )

        if not self.building:
            raise RuntimeError("runtime Decode Graph capture is forbidden")
        key = self.key(mr)
        if key in self.entries:
            raise RuntimeError("prepared Decode Graph interval already captured")
        runner = mr.decode_cuda_graph_runner
        if runner is None:
            raise RuntimeError("prepared Decode Graph capture produced no runner")
        bindings = external_bindings(mr, runner)
        self.entries[key] = PreparedDecodeRunner(
            runner=runner,
            backend_fields={n: getattr(mr, n, None) for n in _BACKEND_FIELDS},
            pool_id=get_global_graph_memory_pool(),
            bindings=bindings,
            signature=binding_signature(bindings),
            capture_seconds=capture_seconds,
            weight_bindings={
                name: tensor
                for name, tensor in mr.model.state_dict().items()
                if re.search(r"(?:^|\.)layers\.\d+\.", name)
            },
        )

    def suspend(self, mr):
        # The bank, its native storage caches and private backend keep owners
        # alive; detach only the serving selector after the old stream drains.
        mr.decode_cuda_graph_runner = None
        mr.prefill_cuda_graph_runner = None
        mr._elastic_pp_decode_graph_state.invalidate()

    def select(self, mr, entry):
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            set_graph_pool_id,
        )
        from sglang.srt.model_executor.runner_utils.pool import (
            set_global_graph_memory_pool,
        )

        bundle = self.entries[self.key(mr)]
        for name, value in bundle.backend_fields.items():
            setattr(mr, name, value)
        if binding_signature(external_bindings(mr, bundle.runner)) != bundle.signature:
            raise RuntimeError("prepared Decode Graph tensor bindings changed")
        set_global_graph_memory_pool(bundle.pool_id)
        set_graph_pool_id(bundle.pool_id)
        mr.decode_cuda_graph_runner = bundle.runner
        entry.update(status="ready", capture_seconds=0.0, prepared=True)
        if not self.building:
            self.runtime_selections += 1

    def seal(self):
        expected = {
            (t.layer_range(self.rank).start, t.layer_range(self.rank).end)
            for t in map(PPTopology.from_layer_counts, self.partitions)
            if not t.layer_range(self.rank).empty
        }
        if set(self.entries) != expected:
            raise RuntimeError("startup did not capture every declared local interval")
        # Snapshot after the complete startup walk: loading the same immutable
        # layer during that walk can increment its version counter.
        for bundle in self.entries.values():
            for name, tensor in bundle.weight_bindings.items():
                match = re.search(r"(?:^|\.)layers\.(\d+)\.(.*)", name)
                if match:
                    layer_id, state_name = int(match[1]), match[2]
                    self.resident_weights.setdefault(layer_id, {})[state_name] = (
                        self._weight_identity(tensor)
                    )
        self.building = False

    def metrics(self):
        result = dict(
            sealed=not self.building,
            local_intervals=[list(key) for key in self.entries],
            startup_runner_captures=len(self.entries),
            startup_capture_seconds=sum(
                e.capture_seconds for e in self.entries.values()
            ),
            startup_seconds=self.startup_seconds,
            startup_allocated_delta_bytes=self.startup_allocated_delta_bytes,
            runtime_captures=0,
            runtime_selections=self.runtime_selections,
            host_allocator_warmups=self.host_allocator_warmups,
        )
        if self.torch_host_allocator:
            stats = torch.cuda.memory.host_memory_stats()
            result["pinned_allocator"] = {
                key: stats[key]
                for key in (
                    "allocated_bytes.current",
                    "active_bytes.current",
                    "num_host_alloc",
                    "num_host_free",
                )
            }
        return result


def init_prepared_graph_bank(mr):
    args = mr.server_args
    initial = tuple(map(int, args.elastic_pp_initial_partition.split(",")))
    partitions = validate_graph_partitions(args, initial)
    from sglang.srt.model_executor.runner_utils.pool import disable_graph_pool_borrow

    disable_graph_pool_borrow("elastic PP retains independent prepared Graph pools")
    mr._elastic_pp_model_method("retain_elastic_pp_graph_layers")()
    mr.token_to_kv_pool.retain_elastic_pp_graph_buffers()
    mr._elastic_pp_prepared_graphs = PreparedDecodeGraphBank(mr, partitions)


def _reset_startup_residency(mr):
    # No requests or public generation exist yet. Re-create the transactional
    # ledgers after the private startup walk, preserving all pinned storage.
    model = mr.model
    while not hasattr(model, "_elastic_layer_residency"):
        model = model.model
    pool = mr.token_to_kv_pool
    owners = [(model, "_elastic_layer_residency", "scheduled_range")]
    if hasattr(pool, "_residency"):
        owners.append((pool, "_residency", "active_range"))
    full = getattr(pool, "full_kv_pool", pool)
    owners.append((full, "_elastic_pp_residency", "active_range"))
    for owner, name, range_name in owners:
        ledger = getattr(owner, name)
        if ledger is None:
            continue
        if ledger.pending is not None or ledger._cleanup_generation is not None:
            raise RuntimeError("startup Graph walk left a pending storage transaction")
        setattr(
            owner,
            name,
            type(ledger)(
                num_layers=ledger.num_layers,
                initial_generation=0,
                initial_range=getattr(ledger, range_name),
            ),
        )


def prewarm_scheduler_graph_bank(scheduler):
    """Visit finite layouts before readiness, then restore public generation 0.

    The HiCache tree is still empty and retains its initial native bindings.
    Every temporary layout uses the regular transactional transports with no
    online patching, and restores those exact initial addresses before serving.
    """
    mr = scheduler.tp_worker.model_runner
    bank = getattr(mr, "_elastic_pp_prepared_graphs", None)
    if bank is None:
        return
    from sglang.srt.elastic_pp.coordinator import (
        ElasticPPMigrationCoordinator,
    )
    from sglang.srt.elastic_pp.coordinator import ElasticPPMigrationPhase as Phase
    from sglang.srt.elastic_pp.decode_graph import ElasticPPDecodeGraphState
    from sglang.srt.elastic_pp.topology import DynamicPPTransition

    if (
        not scheduler.is_initializing
        or scheduler.waiting_queue
        or not scheduler.running_batch.is_empty()
    ):
        raise RuntimeError("prepared Graph warmup requires startup with no requests")
    original = scheduler._pp_topology()
    initial = tuple(r.size for r in original.layer_ranges)
    started = time.monotonic()
    before = torch.cuda.memory_allocated()
    coordinator = ElasticPPMigrationCoordinator(
        rank=scheduler.ps.pp_rank,
        current_topology=original,
        group=mr.elastic_pp_migration_group,
        model_runner=mr,
        layer_transport=mr.elastic_pp_layer_transport,
        kv_transport=mr.elastic_pp_kv_transport,
        strategy="sync",
        enable_kv_patch=False,
    )
    # All requests are excluded until initialization returns. NIXL endpoints
    # stay initialized; no live-patch generations should leak from this walk.
    walk = [counts for counts in bank.partitions if counts != initial] + [initial]
    with mr.elastic_pp_kv_transport.empty_startup_transport():
        for counts in walk:
            target = PPTopology.from_layer_counts(
                counts, generation=coordinator.current_topology.generation + 1
            )
            coordinator.begin(
                DynamicPPTransition.build(coordinator.current_topology, target)
            )
            state = coordinator.wait_for_phase(
                {Phase.BULK_READY, Phase.FAILED}, timeout=120
            )
            if state.phase == Phase.FAILED:
                raise RuntimeError(f"startup Graph preparation failed: {state.error}")
            coordinator.finalize_at_fence(scheduler._pp_install_topology)
            coordinator.start_cleanup()
            state = coordinator.wait_for_phase({Phase.DONE, Phase.FAILED}, timeout=120)
            if state.phase == Phase.FAILED:
                raise RuntimeError(f"startup Graph cleanup failed: {state.error}")
            if (
                getattr(mr.server_args, "elastic_pp_hicache_allocator", "mmap")
                == "pytorch"
            ):
                from sglang.srt.elastic_pp.hicache_transition import (
                    prewarm_host_allocator,
                )

                bank.host_allocator_warmups.append(prewarm_host_allocator(scheduler))
    _reset_startup_residency(mr)
    # Private preparation generations were never advertised to a client. The
    # normal route installer must retain its monotonic serving-generation gate.
    scheduler._elastic_pp_topology = None
    scheduler._pp_install_topology(original)
    bank.seal()
    bank.startup_seconds = time.monotonic() - started
    bank.startup_allocated_delta_bytes = torch.cuda.memory_allocated() - before
    state = mr._elastic_pp_decode_graph_state = ElasticPPDecodeGraphState()
    entry = state.begin(0)
    if original.layer_range(scheduler.ps.pp_rank).empty:
        entry.update(status="inactive", capture_seconds=0.0, prepared=True)
    else:
        bank.select(mr, entry)
        bank.runtime_selections = 0
    scheduler.init_elastic_pp_migration_coordinator()
    logger.info("ELASTIC_PP_PREPARED_GRAPHS %s", bank.metrics())
