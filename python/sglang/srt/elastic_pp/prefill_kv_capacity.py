"""Size the elastic Prefill KV pool from GPU memory after startup preparation."""

import json
import logging
import math
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


def prefill_capacity_layouts(args):
    from sglang.srt.elastic_pp.autoscaling import parse_elastic_pp_partitions

    initial = tuple(map(int, args.elastic_pp_initial_partition.split(",")))
    layouts = parse_elastic_pp_partitions(args.elastic_pp_graph_partitions)
    (expanded,) = (layout for layout in layouts if layout != initial)
    if sum(n > 0 for n in expanded) <= sum(n > 0 for n in initial):
        raise ValueError("automatic Prefill capacity must start from the smaller PP layout")
    return initial, expanded


def prefill_virtual_capacity(runner):
    """Reserve addresses for any serving pool that could fit on these devices."""
    import torch

    config = runner.model_config
    hybrid = runner.kv_cache_configurator.hybrid_gdn_config
    if hybrid is None or runner.use_mla_backend:
        raise ValueError("automatic Prefill capacity requires Qwen hybrid Attention/GDN")
    heads = config.get_num_kv_heads(1, 1)
    layer_bytes = 2 * heads * config.head_dim * runner.kv_cache_dtype.itemsize
    rank = runner.ps.pp_rank
    groups = []
    for layout in prefill_capacity_layouts(runner.server_args):
        start = sum(layout[:rank])
        groups.append(
            sum(start <= i < start + layout[rank] for i in hybrid.full_attention_layer_ids)
        )
    total = torch.cuda.mem_get_info(runner.gpu_id)[1]
    local_limits = tuple(total // (n * layer_bytes) - 1 if n else None for n in groups)
    rows = runner.elastic_pp_migration_group.all_gather_object(local_limits)
    return max(min(x[i] for x in rows if x[i] is not None) for i in range(2))


@dataclass(frozen=True)
class RankMemory:
    free_without_kv: tuple[int, int]
    groups: tuple[int, int]
    departing_groups: tuple[int, int]
    incoming_weights: tuple[int, int]
    serving_reserve: int
    prefill_workspace: int


def transition_reserve_bytes(required_bytes):
    return max(256 * 2**20, math.ceil(required_bytes * 0.05))


def resolve_prefill_pages(ranks, *, page_bytes, max_pages):
    """Size both serving pools independently and the temporary overlap pool."""
    steady = tuple(
        min(
            max_pages,
            min(
                (r.free_without_kv[i] - r.serving_reserve) // (r.groups[i] * page_bytes)
                for r in ranks
                if r.groups[i]
            ),
        )
        for i in range(2)
    )

    def fits(pages):
        for rank in ranks:
            for src, dst in ((0, 1), (1, 0)):
                old_bytes = rank.groups[src] * pages * page_bytes
                peak_pages = (rank.departing_groups[src] + rank.groups[dst]) * pages
                required = peak_pages * page_bytes + rank.incoming_weights[src]
                additional = required - old_bytes
                reserve = max(rank.prefill_workspace, transition_reserve_bytes(additional))
                if required + reserve > rank.free_without_kv[src]:
                    return False
        return True

    lo, hi = 0, max(min(steady), 0)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    if not lo:
        raise RuntimeError(f"no transition-safe Prefill KV pages: steady={steady}, ranks={ranks}")
    return steady, lo


def initialize_prefill_capacity(scheduler):
    """Publish the physical plan before readiness while Mooncake is suspended."""
    import torch
    from kvcached.tp_ipc_util import snapshot_worker_mapped_offsets

    from sglang.srt.elastic_pp.kv_capacity import full_pool
    from sglang.srt.elastic_pp.topology import LayerRange
    from sglang.srt.runtime_context import pre_capture_activation_reserve_mb

    runner = scheduler.tp_worker.model_runner
    pool = runner.token_to_kv_pool
    full = full_pool(runner)
    layouts = prefill_capacity_layouts(runner.server_args)
    rank = scheduler.ps.pp_rank
    layer_sets, groups = [], []
    for layout in layouts:
        start = sum(layout[:rank])
        interval = LayerRange(start, start + layout[rank])
        layer_sets.append(set(range(interval.start, interval.end)))
        groups.append(full._active_layer_groups(pool._attention_range(interval)))

    bank = runner._elastic_pp_prepared_graphs.weight_backing
    # Graph workspaces and the GDN pools for both layouts already exist. Only
    # the layout's physical weight mapping and attention pages will change.
    if {i for i, owner in bank.layers.items() if owner.mapped} != layer_sets[0]:
        raise RuntimeError("automatic Prefill capacity requires offloaded idle weights")
    weights = tuple(sum(bank.layers[i].size for i in ids) for ids in layer_sets)
    incoming = tuple(
        sum(bank.layers[i].size for i in layer_sets[1 - src] - layer_sets[src])
        for src in range(2)
    )
    manager = full.kvcached_allocator
    page_bytes = int(manager.physical_group_page_size)
    slots_per_page = int(manager.blocks_per_physical_page)
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info(runner.gpu_id)
    mapped_bytes = len(snapshot_worker_mapped_offsets(full._group_id)) * page_bytes
    no_kv = free_bytes + mapped_bytes
    runtime_reserve = math.ceil(
        runner.pre_model_load_memory * (1 - runner.mem_fraction_static) * 2**30
    )
    # Both Graphs are already captured. Reserve the Prefill working set without
    # the larger pre-capture floor for Graph memory on high-memory GPUs.
    workspace = math.ceil(pre_capture_activation_reserve_mb(None) * 2**20)
    local = RankMemory(
        free_without_kv=(no_kv, no_kv + weights[0] - weights[1]),
        groups=tuple(map(len, groups)),
        departing_groups=tuple(len(groups[src] - groups[1 - src]) for src in range(2)),
        incoming_weights=incoming,
        serving_reserve=runtime_reserve,
        prefill_workspace=workspace,
    )
    rows = runner.elastic_pp_migration_group.all_gather_object(local)
    steady_pages, staging_pages = resolve_prefill_pages(
        rows, page_bytes=page_bytes, max_pages=(full.size + 1) // slots_per_page
    )
    serving = tuple(n * slots_per_page - 1 for n in steady_pages)
    staging = staging_pages * slots_per_page - 1
    full.configure_serving_capacity(serving[0])
    full.auto_capacity_by_layout = dict(zip(layouts, serving))
    full.prefill_staging_capacity = staging
    if rank == 0:
        logger.info(
            "ELASTIC_PP_AUTO_KV_CAPACITY %s",
            json.dumps(
                dict(
                    partitions=layouts,
                    serving_tokens=serving,
                    transition_tokens=staging,
                    virtual_tokens=full.size,
                    physical_page_bytes=page_bytes,
                    tokens_per_page=slots_per_page,
                    ranks=[asdict(row) for row in rows],
                )
            ),
        )
