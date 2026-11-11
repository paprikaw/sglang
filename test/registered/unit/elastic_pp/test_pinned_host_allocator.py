"""Registration ownership and lifetime of reusable elastic HiCache backing."""

import weakref
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.elastic_pp.hicache_transition import prewarm_host_allocator
from sglang.srt.mem_cache.pool_host import base, common
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("managed", [False, True])
def test_registration_has_exactly_one_owner(monkeypatch, managed):
    registrations = []
    tensor = torch.empty(8)
    allocator = NS(
        manages_registration=managed,
        allocate=lambda *a, **k: tensor,
    )
    monkeypatch.setattr(common, "_cuda_host_register", registrations.append)
    result = common.alloc_with_host_register(
        (8,), torch.float32, "cpu", True, allocator
    )
    assert result is tensor
    assert len(registrations) == (0 if managed else 1)


def test_retired_pool_drops_cpu_aliases_without_unregistering_torch_memory(monkeypatch):
    tensor = torch.empty(8)
    reference = weakref.ref(tensor)
    pool = object.__new__(MHATokenToKVPoolHost)
    pool.allocator = common.TorchPinnedHostTensorAllocator()
    pool.kv_buffer = tensor
    pool.staging_k_buffer = tensor[:4]
    pool.data_refs = [tensor]
    monkeypatch.setattr(
        base, "_cuda_host_unregister", lambda _: pytest.fail("double unpin")
    )
    del tensor
    pool.destroy()
    pool.destroy()
    assert pool.kv_buffer is None and pool.staging_k_buffer is None
    assert pool.data_refs == [] and reference() is None


@pytest.mark.parametrize("backend,partitions", [("shm", "2,0;1,1"), (None, None)])
def test_pinned_policy_rejects_external_storage_or_missing_bank(backend, partitions):
    args = NS(
        elastic_pp_hicache_allocator="pytorch",
        elastic_pp_graph_partitions=partitions,
        hicache_storage_backend=backend,
    )
    with pytest.raises(ValueError, match="L2-only"):
        common.get_allocator_type(args)


def test_pinned_policy_uses_torch_allocator_and_default_stays_mmap():
    args = NS(
        elastic_pp_hicache_allocator="pytorch",
        elastic_pp_graph_partitions="2,0;1,1",
        hicache_storage_backend=None,
    )
    assert isinstance(
        common.get_allocator_from_storage(common.get_allocator_type(args)),
        common.TorchPinnedHostTensorAllocator,
    )
    assert common.get_allocator_type(NS()) == "default"


@pytest.mark.parametrize("initializing,storage", [(False, False), (True, True)])
def test_host_warmup_cannot_run_during_serving_or_with_storage(initializing, storage):
    scheduler = NS(
        is_initializing=initializing,
        tree_cache=NS(enable_storage=storage),
        tp_worker=NS(
            model_runner=NS(server_args=NS(elastic_pp_hicache_allocator="pytorch"))
        ),
    )
    with pytest.raises((RuntimeError, ValueError), match="startup|L2-only"):
        prewarm_host_allocator(scheduler)
