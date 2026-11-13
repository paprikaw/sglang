import copy
import multiprocessing
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.elastic_pp.shared_host_cache import (
    SharedHostBank,
    SharedHostRegion,
    merge_schemas,
    metadata_signature,
    shared_pool_allocator,
    use_shared_host_bank,
    validate_shared_host_args,
)


def schema():
    return dict(
        layers={"full": [1, 3], "mamba": [0, 2]},
        capacities={"full": 32, "mamba": 8},
        regions={
            "full.kv": dict(shape=(2, 32, 2, 1, 2), dtype="torch.float32"),
            "mamba.temporal": dict(shape=(8, 2, 1, 2, 3, 3), dtype="torch.float32"),
            "mamba.conv_0": dict(shape=(8, 2, 1, 2, 3), dtype="torch.bfloat16"),
        },
    )


@pytest.fixture
def bank():
    result = SharedHostBank.create(schema())
    yield result
    result.close()


def _creator(pipe):
    region = SharedHostRegion.create((7, 3), torch.float32)
    region.tensor.copy_(torch.arange(21).reshape(7, 3))
    pipe.send(region.descriptor())
    pipe.recv()
    region.close()
    pipe.close()


def test_importer_survives_original_worker_exit():
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_creator, args=(child,))
    process.start()
    child.close()
    imported = None
    try:
        assert parent.poll(25), "creator did not publish a shared region"
        imported = SharedHostRegion.attach(parent.recv())
        parent.send("attached")
        process.join(15)
        assert process.exitcode == 0
        assert torch.equal(imported.tensor, torch.arange(21).reshape(7, 3))
        # Re-export from the surviving owner; no old-process pointer is needed.
        another = SharedHostRegion.attach(imported.descriptor())
        try:
            another.tensor[2, 1] = -7
            assert imported.tensor[2, 1].item() == -7
        finally:
            another.close()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        if imported is not None:
            imported.close()
        parent.close()


@pytest.mark.parametrize("change", ["boot", "inode", "bounds", "dtype"])
def test_import_rejects_stale_or_incompatible_region(change):
    region = SharedHostRegion.create((4, 3), torch.float32)
    try:
        desc = copy.deepcopy(region.descriptor())
        if change == "boot":
            desc["boot"] = ("other-node", desc["boot"][1])
        elif change == "inode":
            desc["inode"] += 1
        elif change == "bounds":
            desc["shape"] = (desc["bytes"] + 1,)
        else:
            desc["dtype"] = "not-a-dtype"
        with pytest.raises(ValueError):
            SharedHostRegion.attach(desc)
    finally:
        region.close()


def test_layer_views_keep_physical_cpu_rows_across_layouts(bank):
    original = bank.regions["full.kv"].tensor
    original.copy_(torch.arange(original.numel()).reshape(original.shape))
    with use_shared_host_bank(bank):
        old = shared_pool_allocator("full", NS(start_layer=0, end_layer=2, layer_num=2))
        old_view = old.allocate((2, 32, 2, 1, 2), torch.float32, "cpu")
        new = shared_pool_allocator("full", NS(start_layer=1, end_layer=2, layer_num=1))
        new_view = new.allocate((2, 32, 1, 1, 2), torch.float32, "cpu")
    try:
        assert new_view.stride(1) == original.stride(1)
        assert new_view.data_ptr() == old_view[:, :, 1:].data_ptr()
        assert torch.equal(new_view, old_view[:, :, 1:])
        old.release()  # retiring a source layout must not invalidate the target
        new_view[0, 7, 0, 0, 1] = -23
        assert original[0, 7, 1, 0, 1].item() == -23
        with pytest.raises(RuntimeError, match="live pool"):
            bank.close()
    finally:
        old.release()
        new.release()


def test_context_isolation_and_mamba_shape_checks(bank):
    assert shared_pool_allocator("mamba", NS()) is None
    with use_shared_host_bank(bank):
        allocator = shared_pool_allocator("mamba", NS(mamba_layer_ids=[2]))
        try:
            with pytest.raises(ValueError, match="shape/dtype"):
                allocator.allocate((99, 1), torch.float32, "cpu")
            temporal = allocator.allocate((8, 1, 1, 2, 3, 3), torch.float32, "cpu")
            conv = allocator.allocate((8, 1, 1, 2, 3), torch.bfloat16, "cpu")
            assert temporal.stride(0) == 36
            assert conv.stride(0) == 12
            assert (
                temporal.data_ptr()
                == bank.regions["mamba.temporal"].tensor[:, 1:].data_ptr()
            )
        finally:
            allocator.release()
    assert shared_pool_allocator("mamba", NS()) is None


def test_real_host_pools_use_bank_capacity_and_physical_strides(
    bank, hybrid_pool_factory
):
    from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

    device = hybrid_pool_factory(start_layer=2, end_layer=4)
    with use_shared_host_bank(bank):
        full = MHATokenToKVPoolHost(
            device.full_kv_pool, 2, 0, 1, "page_first", pin_memory=False
        )
        mamba = MambaPoolHost(
            device.mamba_pool, 2, 0, pin_memory=False, layout="page_first"
        )
    try:
        assert full.size == 32 and mamba.size == 8
        assert full.layer_num == 1 and mamba.num_mamba_layers == 1
        assert full.layout_dim == 16  # two physical full layers, one local layer
        assert not full.can_use_write_back_jit
        assert mamba.temporal_buffer.stride(0) == 36
        assert mamba.conv_buffer[0].stride(0) == 12
        full.k_buffer[4, 0].fill_(17)
        assert bank.regions["full.kv"].tensor[0, 4, 1].eq(17).all()
    finally:
        full.destroy()
        mamba.destroy()
    assert bank.leases == 0


def test_mamba_kernel_arguments_use_physical_row_pitch(
    bank, hybrid_pool_factory, monkeypatch
):
    import sglang.srt.mem_cache.pool_host.mamba as module

    device = hybrid_pool_factory(start_layer=2, end_layer=4)
    with use_shared_host_bank(bank):
        host = module.MambaPoolHost(
            device.mamba_pool, 2, 0, pin_memory=False, layout="page_first"
        )
    reads, writes = [], []

    def read(**kw):
        assert kw["src_layout_dim"] == kw["src"].stride(0) * kw["src"].element_size()
        reads.append(kw["src_layout_dim"])
        kw["dst"].index_copy_(
            0,
            kw["dst_indices"],
            kw["src"][kw["src_indices"], kw["layer_id"]].reshape(
                len(kw["dst_indices"]), *kw["dst"].shape[1:]
            ),
        )

    def write(**kw):
        assert kw["dst_layout_dim"] == kw["dst"].stride(0) * kw["dst"].element_size()
        assert kw["num_layers"] == 1
        writes.append(kw["dst_layout_dim"])

    monkeypatch.setattr(module, "transfer_kv_mamba_pf_lf", read, raising=False)
    monkeypatch.setattr(module, "transfer_kv_mamba_lf_pf", write, raising=False)
    try:
        host.temporal_buffer[2].fill_(19)
        host.conv_buffer[0][2].fill_(23)
        host.load_to_device_per_layer(
            device.mamba_pool, torch.tensor([2]), torch.tensor([1]), 0
        )
        assert device.mamba_pool.mamba_cache.temporal[0, 1].eq(19).all()
        assert device.mamba_pool.mamba_cache.conv[0][0, 1].eq(23).all()
        host.backup_from_device_all_layer(
            device.mamba_pool, torch.tensor([3]), torch.tensor([1])
        )
        assert reads == writes == [144, 24]
    finally:
        host.destroy()


def test_schema_requires_unique_complete_layers_and_consistent_formats():
    first = {
        "full": dict(
            layers=[0, 1],
            fields={
                "k": dict(shape=(8, 1, 2), dtype="torch.float32"),
                "v": dict(shape=(8, 1, 2), dtype="torch.float32"),
            },
        )
    }
    second = copy.deepcopy(first)
    second["full"]["layers"] = [2, 3]
    result = merge_schemas([first, second], 4)
    assert result["regions"]["full.kv"]["shape"] == (2, 8, 4, 1, 2)
    with pytest.raises(ValueError, match="duplicate"):
        merge_schemas([first, first], 4)
    with pytest.raises(ValueError, match="cover"):
        merge_schemas([first], 4)
    second["full"]["fields"]["k"]["dtype"] = "torch.bfloat16"
    with pytest.raises(ValueError, match="shape/dtype"):
        merge_schemas([first, second], 4)


def test_slot_metadata_signature_does_not_scan_payload(monkeypatch):
    from test_hicache_fast_copy import cache

    src = cache([1, 2])
    pool = NS(
        size=8, free_slots=torch.arange(3, 8), release_slots=[], num_release_slots=0
    )
    src.host_pool_group = NS(entries=[NS(name="kv", host_pool=pool)])
    monkeypatch.setattr(torch, "equal", lambda *a: pytest.fail("payload comparison"))
    a = metadata_signature(src)
    pool.free_slots = torch.tensor([4, 3, 5, 6, 7])
    assert a != metadata_signature(src)


def test_shared_mode_rejects_unsupported_configuration():
    args = NS(
        enable_elastic_pp=True,
        enable_hierarchical_cache=True,
        elastic_pp_graph_partitions="20,0,20;13,13,14",
        disable_cuda_graph=False,
        nnodes=1,
        tp_size=1,
        ep_size=1,
        pp_stage_tp_sizes=None,
        elastic_pp_kv_backend="native",
        hicache_mem_layout="page_first",
        hicache_io_backend="kernel",
        hicache_storage_backend=None,
        hicache_host_memory_mode="cache",
        elastic_pp_transition_backing="overlap",
        elastic_pp_prepared_weight_policy="offload",
    )
    validate_shared_host_args(args)
    for key, value in (
        ("nnodes", 2),
        ("tp_size", 2),
        ("ep_size", 2),
        ("hicache_mem_layout", "layer_first"),
        ("hicache_io_backend", "direct"),
        ("hicache_storage_backend", "nixl"),
        ("elastic_pp_prepared_weight_policy", "resident"),
    ):
        other = copy.copy(args)
        setattr(other, key, value)
        with pytest.raises(ValueError, match="shared host"):
            validate_shared_host_args(other)


def _shared_transition_adapter():
    from test_hicache_fast_copy import cache

    from sglang.srt.elastic_pp.hicache_transition import SchedulerHiCacheTransition
    from sglang.srt.elastic_pp.topology import DynamicPPTransition, PPTopology

    src = cache([1, 2])
    src.host_pool_group = NS(
        entries=[
            NS(
                name="kv",
                host_pool=NS(
                    size=8,
                    free_slots=torch.arange(3, 8),
                    release_slots=[],
                    num_release_slots=0,
                ),
            )
        ]
    )
    runner = NS(
        server_args=NS(elastic_pp_hicache_shared_host=True),
        _elastic_pp_shared_host_bank=object(),
    )
    adapter = SchedulerHiCacheTransition(
        scheduler=NS(tree_cache=src), model_runner=runner, group=None, rank=0
    )
    adapter._source_cache = src
    transition = DynamicPPTransition.build(
        PPTopology.from_layer_counts([2, 0, 2]),
        PPTopology.from_layer_counts([1, 1, 2], generation=1),
    )
    return adapter, transition


@pytest.mark.parametrize("fault", [None, "slots", "error", "duplicate_rank"])
def test_shared_handoff_requires_complete_equal_source_metadata(fault):
    adapter, transition = _shared_transition_adapter()
    signature = metadata_signature(adapter._source_cache)
    reports = [
        dict(rank=0, signature=signature, error=None),
        dict(rank=1, signature=None, error=None),
        dict(rank=2, signature=signature, error=None),
    ]
    if fault == "slots":
        reports[2]["signature"] = "different-slot-map"
    elif fault == "error":
        reports[2]["error"] = "inspection failed"
    elif fault == "duplicate_rank":
        reports[2]["rank"] = 0
    adapter.group = NS(all_gather_object=lambda report: reports)
    if fault:
        with pytest.raises(RuntimeError):
            adapter.exchange_at_fence(transition)
        assert adapter._shared_verified_generation is None
    else:
        adapter.exchange_at_fence(transition)
        assert adapter._shared_verified_generation == 1


def test_shared_begin_avoids_payload_snapshot_and_does_not_publish_busy_loadback(
    monkeypatch,
):
    adapter, transition = _shared_transition_adapter()
    monkeypatch.setattr(torch, "clone", lambda *a: pytest.fail("CPU payload snapshot"))
    adapter.begin_background(transition)
    assert adapter.initial_base_ready and adapter.precopy is None
    adapter.scheduler.tree_cache.ongoing_load_back = {1: object()}
    adapter.begin_background(transition)
    assert not adapter.initial_base_ready
    assert adapter.precopy is None


def test_target_views_cannot_publish_without_slot_validation():
    adapter, transition = _shared_transition_adapter()
    with pytest.raises(RuntimeError, match="validated slot"):
        adapter.apply_staged(transition)


def test_failed_shared_target_construction_never_retires_source(monkeypatch):
    from dataclasses import make_dataclass

    from sglang.srt.elastic_pp import hicache_transition as module
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    ct = ComponentType.FULL
    pool, controller, group = object(), object(), object()
    params_type = make_dataclass(
        "Params", ["token_to_kv_pool_allocator", "req_to_token_pool"]
    )
    source = NS(
        host_pool_group=group,
        cache_controller=controller,
        token_to_kv_pool_host=group,
        full_kv_pool_host=pool,
        components={ct: NS(_full_kv_pool_host=pool)},
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=NS(),
        _hicache_init_params=params_type(None, None),
        _hicache_init_server_args=NS(),
    )
    clone = copy.copy(source)
    clone.components = {ct: copy.copy(source.components[ct])}
    retired = []
    clone.release_host_resources = lambda: retired.append(clone.host_pool_group)

    def fail_init(*args):
        assert clone.host_pool_group is None
        assert clone.cache_controller is None
        assert clone.full_kv_pool_host is None
        assert clone.components[ct]._full_kv_pool_host is None
        raise MemoryError("target allocation refused")

    clone.init_hicache = fail_init
    monkeypatch.setattr(module, "assert_host_quiescent", lambda _: None)
    monkeypatch.setattr(
        module, "project_target_device_pool", lambda *a, **k: (NS(), None)
    )
    monkeypatch.setattr(module, "clone_cache_runtime_metadata", lambda _: clone)
    with pytest.raises(MemoryError, match="target allocation refused"):
        module.prepare_target_host_cache(
            source, device_pool=NS(), target=NS(size=2), generation=1, reuse_shared=True
        )
    assert retired == [None]
    assert source.host_pool_group is group
    assert source.cache_controller is controller
    assert source.full_kv_pool_host is source.components[ct]._full_kv_pool_host is pool
