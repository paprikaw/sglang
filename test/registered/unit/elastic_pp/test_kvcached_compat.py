import sys
from bisect import bisect_left
from types import ModuleType, SimpleNamespace

import pytest

from sglang.srt.mem_cache.kvcached_compat import (
    probe_layer_mapping_capabilities,
    validate_layer_stacking_range,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture
def dependency(monkeypatch):
    utils = ModuleType("kvcached.utils")
    utils.ELASTIC_PP_LAYER_MAPPING_API_VERSION = 1
    utils.LAYER_STACKING = utils.CONTIGUOUS_LAYOUT = True
    utils.PAGE_PREALLOC_ENABLED = False
    native = ModuleType("kvcached.vmm_ops")
    native.elastic_pp_layer_mapping_abi_version = 1
    ipc = ModuleType("kvcached.tp_ipc_util")
    for name in (
        "register_worker_offset_filter",
        "map_worker_offsets_for_migration",
        "unmap_worker_offsets_for_layer_groups",
        "snapshot_worker_mapped_offsets",
        "pin_worker_offsets",
        "unpin_worker_offsets",
    ):
        setattr(ipc, name, lambda *args, **kwargs: None)
    manager = ModuleType("kvcached.kv_cache_manager")
    manager.KVCacheManager = SimpleNamespace(
        snapshot_inuse_page_ids=lambda: [], get_layer_group_total_span=lambda: 1
    )
    for module in (utils, native, ipc, manager):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return utils, native, ipc, manager


def test_matching_python_and_native_interfaces_qualify(dependency):
    capabilities = probe_layer_mapping_capabilities()
    assert capabilities.supported
    capabilities.require()


@pytest.mark.parametrize(
    "index,name,value,reason",
    [
        (0, "ELASTIC_PP_LAYER_MAPPING_API_VERSION", 0, "Python layer-mapping API"),
        (1, "elastic_pp_layer_mapping_abi_version", 0, "native layer-mapping ABI"),
        (0, "LAYER_STACKING", False, "KVCACHED_LAYER_STACKING"),
        (0, "CONTIGUOUS_LAYOUT", False, "contiguous reservation"),
        (0, "PAGE_PREALLOC_ENABLED", True, "KVCACHED_PAGE_PREALLOC_ENABLED"),
        (
            2,
            "map_worker_offsets_for_migration",
            None,
            "map_worker_offsets_for_migration",
        ),
        (2, "pin_worker_offsets", None, "pin_worker_offsets"),
    ],
)
def test_incompatible_dependency_fails_with_actionable_reason(
    dependency, index, name, value, reason
):
    setattr(dependency[index], name, value)
    capabilities = probe_layer_mapping_capabilities()
    assert not capabilities.supported
    with pytest.raises(RuntimeError, match=reason):
        capabilities.require()


def test_35b_pp2_to_pp3_all_attention_boundaries_support_group_size_one():
    attention_ids = tuple(range(3, 40, 4))
    ranges = [(0, 20), (20, 20), (20, 40), (0, 13), (13, 26), (26, 40)]
    ordinals = [
        (bisect_left(attention_ids, start), bisect_left(attention_ids, end))
        for start, end in ranges
    ]
    assert ordinals == [(0, 5), (5, 5), (5, 10), (0, 3), (3, 6), (6, 10)]
    for start, end in ordinals:
        assert (
            validate_layer_stacking_range(
                total_layers=10,
                start=start,
                end=end,
                page_bytes=128,
                physical_block_bytes=128,
            )
            == 1
        )


def test_35b_stacking_two_rejects_pp2_and_pp3_unaligned_attention_boundaries():
    for start, end in ((0, 5), (5, 10), (0, 3), (3, 6)):
        with pytest.raises(ValueError, match="aligned"):
            validate_layer_stacking_range(
                total_layers=10,
                start=start,
                end=end,
                page_bytes=128,
                physical_block_bytes=64,
            )


@pytest.mark.parametrize(
    "layers,start,end,page,block",
    [
        (10, 0, 5, 128, 32),
        (4, -1, 2, 128, 128),
        (4, 0, 5, 128, 128),
        (4, 0, 2, 128, 0),
        (4, 0, 2, 128, 48),
    ],
)
def test_invalid_geometry_rejected(layers, start, end, page, block):
    with pytest.raises(ValueError):
        validate_layer_stacking_range(
            total_layers=layers,
            start=start,
            end=end,
            page_bytes=page,
            physical_block_bytes=block,
        )
