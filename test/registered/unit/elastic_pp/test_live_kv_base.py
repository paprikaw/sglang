from types import SimpleNamespace

import pytest
import torch

from sglang.srt.elastic_pp.live_kv_patch import LiveKVSnapshot
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def fixture(*, size=4, page_size=1):
    allocator = (
        TokenToKVPoolAllocator(size, torch.float32, "cpu", object(), False)
        if page_size == 1
        else PagedTokenToKVPoolAllocator(
            size, page_size, torch.float32, "cpu", object(), False
        )
    )
    slots = allocator.alloc(2 if page_size == 1 else page_size)
    capacity = size + page_size
    source = {0: (torch.arange(capacity).view(-1, 1).float(), torch.zeros(capacity, 1))}
    patch = LiveKVSnapshot(
        generation=1,
        allocator=allocator,
        source=source,
        max_rows=2,
        admit_host_bytes=lambda _: None,
    )
    patch.seed_initial_base()
    return allocator, slots, source, patch


def ready():
    return SimpleNamespace(synchronize=lambda: None)


def test_initial_cache_is_copied_online_but_later_loadback_waits():
    allocator, slots, source, patch = fixture()
    loaded = allocator.alloc(1)
    source[0][0][loaded] = 77
    patch.progress(slots[:1], ready())
    patch.progress(slots[:1], ready())
    assert patch.online_completed_rows == 2
    assert set(slots.tolist()) <= patch.published
    assert loaded.item() not in patch.published
    patch.finish_at_fence(allocator.elastic_pp_live_slots(), ready())
    assert patch.metrics()["fence"]["rows_copied_at_fence"] == 1
    assert patch.metrics()["initial_base_rows_published"] == 2
    assert patch.export(0, loaded)[0].item() == 77


def test_freed_initial_incarnation_is_not_published_after_slot_reuse():
    allocator, slots, source, patch = fixture(size=2)
    allocator.free(slots[:1])
    again = allocator.alloc(1)
    assert again.item() == slots[0].item()
    source[0][0][again] = 91
    patch.progress(slots[1:], ready())
    assert again.item() not in patch.published
    assert patch.metrics()["initial_base_rows_published"] == 1
    patch.finish_at_fence(allocator.elastic_pp_live_slots(), ready())
    assert patch.export(0, again)[0].item() == 91


@pytest.mark.parametrize("page_size", [1, 4])
def test_initial_base_rewrite_still_advances_incarnation_before_write(page_size):
    allocator, slots, source, patch = fixture(
        size=4 if page_size == 1 else 12, page_size=page_size
    )
    patch.progress(slots[:1], ready())
    rewritten = slots[-1:]
    patch.prepare_writes(rewritten)
    assert patch.epochs[rewritten.item()] == 2
    source[0][0][rewritten] = 55
    patch.progress(rewritten, ready())
    patch.finish_at_fence(allocator.elastic_pp_live_slots(), ready())
    assert patch.export(0, rewritten)[0].item() == 55
    untouched = slots[:1]
    assert torch.equal(patch.export(0, untouched)[0], source[0][0][untouched])


def test_base_seed_is_startup_only_and_abort_drains_copies():
    allocator, slots, source, patch = fixture()
    with pytest.raises(RuntimeError, match="once before forward"):
        patch.seed_initial_base()
    patch.progress(slots, ready())
    patch.abort()
    assert patch.initial_base is None
    assert allocator._elastic_pp_live_patch is None
    allocator.free(slots)


@pytest.mark.parametrize(
    "missing",
    [
        "enable_elastic_pp_kv_patch",
        "enable_hierarchical_cache",
        "elastic_pp_hicache_background",
    ],
)
def test_base_config_requires_quiescent_cache_capture_path(missing):
    options = dict(
        model_path="unused",
        pp_size=3,
        enable_elastic_pp=True,
        elastic_pp_initial_partition="20,0,20",
        disable_cuda_graph=True,
        elastic_pp_transition_strategy="background",
        enable_elastic_pp_kv_patch=True,
        enable_hierarchical_cache=True,
        elastic_pp_hicache_background=True,
        elastic_pp_hicache_fast_copy=True,
        elastic_pp_kv_base_precopy=True,
    )
    options[missing] = False
    args = ServerArgs(**options)
    args.get_model_config = lambda: SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
    )
    with pytest.raises(ValueError, match="initial KV base pre-copy requires"):
        args._handle_pipeline_parallelism()
