"""Prepare a node-local private writable HiCache and seed its host-only radix tree."""

import math
import os
from pathlib import Path
import time

import torch

from sglang.srt.weight_cache.replica_preparation import (
    publish,
    preparation_key,
    read_packet,
)
from sglang.srt.elastic_pp.shared_host_cache import (
    SharedHostBank,
    use_shared_host_bank,
    shared_host_params,
)


def contract(mr, root):
    args = mr.server_args
    if (
        not args.elastic_pp_hicache_shared_host
        or args.elastic_pp_initial_partition != "24,24"
        or args.disaggregation_mode != "decode"
        or args.pp_size != 2
        or args.tp_size != 1
        or args.ep_size != 1
    ):
        raise ValueError(
            "Prepared replica HiCache requires identical fixed PP2 Decode ranks"
        )
    names = (
        "max_total_tokens",
        "max_mamba_cache_size",
        "page_size",
        "hicache_size",
        "hicache_mem_layout",
        "hicache_io_backend",
        "hicache_write_policy",
        "kv_cache_dtype",
        "elastic_pp_kv_capacities",
    )
    return dict(
        key=preparation_key(root, mr.model._canonical_cpu_weight_cache_key),
        config={k: getattr(args, k) for k in names},
    )


def publish_schema(mr, bank):
    root = Path(os.environ["SGLANG_REPLICA_EXPORT_DIR"])
    if mr.ps.pp_rank == 0:
        publish(
            root / "source-schema.json", dict(**contract(mr, root), schema=bank.schema)
        )


def prepare_spare(root, key, cap):
    src = read_packet(root, "source-schema.json", key)
    total = sum(
        math.prod(s["shape"]) * getattr(torch, s["dtype"].split(".")[-1]).itemsize
        for s in src["schema"]["regions"].values()
    )
    if total <= 0 or total > cap:
        raise ValueError("Spare HiCache exceeds its admitted CPU budget")
    started = time.monotonic()
    bank = SharedHostBank.create(src["schema"], pin_memory=False)
    try:
        publish(
            root / "spare.json",
            dict(
                key=key,
                config=src["config"],
                descriptor=bank.descriptor(),
                bytes=bank.bytes,
                seconds=time.monotonic() - started,
            ),
        )
    except BaseException:
        bank.close()
        raise
    return bank


def connect(mr, root, pin):
    c = contract(mr, root)
    spare = read_packet(root, "spare.json", c["key"])
    if c["config"] != spare["config"]:
        raise ValueError("Incompatible prepared HiCache geometry or cache policy")
    return SharedHostBank.attach(spare["descriptor"], pin_memory=pin)


def seed_views(bank, rows):
    """Only committed prefix rows are needed to seed another writable generation."""
    if set(rows) != {"full", "mamba"} or any(
        not 0 < rows[kind] <= bank.schema["capacities"][kind] for kind in rows
    ):
        raise ValueError("Replica snapshot has invalid host row counts")
    views = {}
    for name, region in bank.regions.items():
        if name == "full.kv":
            views[name] = region.tensor[:, :rows["full"]]
        elif name.startswith("mamba."):
            views[name] = region.tensor[:rows["mamba"]]
        else:
            raise ValueError(f"Unknown host cache region {name}")
    return views


def snapshot_seed(bank, root, key, byte_cap):
    if any((root / f"consumer-{rank}.json").exists() for rank in (0, 1)):
        raise RuntimeError("Cannot snapshot a bank after it gained a writable consumer")
    history = [read_packet(root, f"history-{i}.json", key) for i in (0, 1)]
    if (history[0]["rows"] != history[1]["rows"] or
            history[0]["nodes"] != history[1]["nodes"]):
        raise ValueError("Stage prefix metadata must agree before cloning a seed")
    views = seed_views(bank, history[0]["rows"])
    size = sum(t.numel()*t.element_size() for t in views.values())
    if size != sum(h["bytes"] for h in history) or size > byte_cap:
        raise ValueError("Seed byte coverage or admitted CPU size differs")
    return history, {k: t.clone() for k,t in views.items()}


def new_seeded_spare(source, target, key, cap, history, seed):
    publish(target / "source-schema.json", read_packet(source, "source-schema.json", key))
    bank = prepare_spare(target, key, cap)
    try:
        for name, dest in seed_views(bank, history[0]["rows"]).items():
            dest.copy_(seed[name])
        for rank, record in enumerate(history):
            publish(target / f"history-{rank}.json", record)
    except BaseException:
        bank.close()
        raise
    return bank


def initialize_cache(cache, params, ctx, server_args):
    root = Path(os.environ["SGLANG_REPLICA_PREPARED_DIR"])
    mr = ctx.tp_worker.model_runner
    # A live writable generation belongs to exactly one fresh PP2. Failed or
    # retired consumers cannot silently reopen its already-used slots.
    claim = root / f"consumer-{mr.ps.pp_rank}.json"
    publish(claim, dict(pid=os.getpid()))
    started = time.monotonic()
    bank = connect(mr, root, True)
    with use_shared_host_bank(bank):
        cache.init_hicache(server_args, shared_host_params(params))
    cache._elastic_pp_shared_host_bank = bank
    mr._replica_attached_bank = bank
    mr._replica_attach_seconds = time.monotonic() - started


def _tree_paths(cache):
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    full, recurrent = ComponentType.FULL, ComponentType.MAMBA

    def visit(parent, parent_index):
        # Parent order and namespacing must be identical across the two ranks.
        children = sorted(
            parent.children.values(),
            key=lambda n: (
                str(n.key.extra_key),
                str(n.key.cache_salt),
                tuple(n.key.raw_token_ids()),
            ),
        )
        for node in children:
            if node.key.is_bigram:
                raise ValueError("Bigram replica snapshots are unsupported")
            f = node.component_data[full].host_value
            m = node.component_data[recurrent].host_value
            if f is None:
                continue  # A descendant cannot be restored over a missing ancestor.
            ids = list(node.key.raw_token_ids())
            if len(ids) != len(f) or not ids:
                raise ValueError("Host prefix KV rows disagree with key length")
            record = dict(
                parent=parent_index,
                tokens=ids,
                extra=node.key.extra_key,
                salt=node.key.cache_salt,
                hashes=list(node.hash_value or []),
            )
            index = len(records)
            records.append((record, f, m))
            visit(node, index)

    records = []
    visit(cache.tree_core.root_node, -1)
    return records


def seed_history(scheduler):
    """Copy stable committed host rows into a spare, without changing donor slots."""
    from sglang.srt.elastic_pp.hicache_transition import (
        assert_host_quiescent,
        cache_host_views,
    )
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    root = Path(os.environ["SGLANG_REPLICA_EXPORT_DIR"])
    mr, cache = scheduler.tp_worker.model_runner, scheduler.tree_cache
    rank = mr.ps.pp_rank
    receipt = root / f"history-{rank}.json"
    if receipt.exists():
        return True  # same snapshot, immutable. Never refill a live spare.
    if not scheduler.is_fully_idle():
        return False
    cache.writing_check(write_back=True)
    try:
        assert_host_quiescent(cache, allow_pp_control_sends=True)
    except RuntimeError:
        return False
    snapshot = _tree_paths(cache)
    if not any(m is not None for _, _, m in snapshot):
        return False
    started = time.monotonic()
    bank = connect(mr, root, False)
    source_views = cache_host_views(cache)
    cursors = dict(full=0, mamba=0)
    copied = 0
    try:
        for rec, full, mamba in snapshot:
            for kind, ct, slots in (
                ("full", ComponentType.FULL, full),
                ("mamba", ComponentType.MAMBA, mamba),
            ):
                if slots is None:
                    rec[kind] = None
                    continue
                start, count = cursors[kind], int(slots.numel())
                if count <= 0 or start + count > bank.schema["capacities"][kind]:
                    raise ValueError("Historical prefix exceeds replica HiCache budget")
                rec[kind] = [start, count]
                cursors[kind] += count
                # PP ranks own disjoint global layers. The scheduler thread
                # owns this source tree until every CPU copy below completes.
                for layer, fields in source_views[int(ct)].items():
                    ordinal = bank.schema["layers"][kind].index(layer)
                    for field, tensor in fields.items():
                        if kind == "full":
                            dest = bank.regions["full.kv"].tensor[
                                {"k": 0, "v": 1}[field], :, ordinal
                            ]
                        else:
                            dest = bank.regions["mamba." + field].tensor[:, ordinal]
                        for off in range(0, count, 256):
                            rows = slots[off : off + 256].to(
                                dtype=torch.int64, device="cpu"
                            )
                            dest[start + off : start + off + len(rows)].copy_(
                                tensor.index_select(0, rows)
                            )
                        copied += count * tensor[0].numel() * tensor.element_size()
        publish(
            receipt,
            dict(
                **contract(mr, root),
                nodes=[r for r, _, _ in snapshot],
                rows=cursors,
                bytes=copied,
                seconds=time.monotonic() - started,
            ),
        )
    finally:
        bank.close()
    return True


def restore_history(cache, mr, root):
    from array import array
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.hicache_storage import (
        PoolName,
        PoolTransfer,
        PoolTransferResult,
    )
    from sglang.srt.mem_cache.unified_cache.components.tree_component import (
        CacheTransferPhase,
        ComponentType,
    )

    key = contract(mr, root)["key"]
    # Both complete receipts must agree before either rank installs any index.
    a, b = [read_packet(root, f"history-{i}.json", key) for i in (0, 1)]
    if (
        a["nodes"] != b["nodes"]
        or a["rows"] != b["rows"]
        or min(a["bytes"], b["bytes"]) <= 0
    ):
        raise RuntimeError("Incomplete cross-rank prefix snapshot")
    tree = cache.tree_core
    if tree.root_node.children:
        raise RuntimeError("Prepared history requires a fresh empty radix tree")

    def reserve(pool, extent):
        start, count = extent
        slots = pool.alloc(count)
        if slots is None or not torch.equal(
            slots.cpu(), torch.arange(start, start + count)
        ):
            raise RuntimeError("New replica allocator cannot claim seeded rows")
        return slots

    ids = []
    for row in a["nodes"]:
        parent = tree.root_node.id if row["parent"] == -1 else ids[row["parent"]]
        key = RadixKey(
            array("q", row["tokens"]), extra_key=row["extra"], cache_salt=row["salt"]
        )
        slots = reserve(cache.full_kv_pool_host, row["full"])
        result = tree.insert_host(parent, key, slots, row["hashes"])
        if (
            result.host_insert_dropped
            or result.inserted_host_node is None
            or result.prefix_len
        ):
            raise RuntimeError("Native radix insertion rejected prepared history")
        cache._apply_cache_actions(result.cache_actions)
        ids.append(result.inserted_host_node)
        if row["mamba"] is not None:
            state = reserve(cache.mamba_pool_host, row["mamba"])
            actions = []
            tree.commit_hicache_transfers(
                parent,
                CacheTransferPhase.PREFETCH,
                {
                    ComponentType.MAMBA: [
                        PoolTransfer(name=PoolName.MAMBA, host_indices=state)
                    ]
                },
                cache_actions=actions,
                insert_result=result,
                pool_storage_result=PoolTransferResult(
                    kv_hit_pages=len(slots) // cache.page_size,
                    extra_pool_hit_pages={PoolName.MAMBA: 1},
                ),
            )
            if actions or result.mamba_exist:
                raise RuntimeError(
                    "Native recurrent cache insertion rejected prepared history"
                )
    return dict(host_rows=a["rows"], metadata_nodes=len(ids), rank=mr.ps.pp_rank)
