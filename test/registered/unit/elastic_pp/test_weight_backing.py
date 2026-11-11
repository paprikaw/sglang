"""Weight alias correctness and physical-map lifecycle without GPU allocation."""

from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from sglang.srt import cuda_vmm_utils as vmm
from sglang.srt.elastic_pp.weight_backing import (
    EvictableWeightBank,
    copy_and_rebind_weight_storages,
    weight_storage_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_packed_weights_rebind_unregistered_strided_views_and_keep_parameters():
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(4, 6))
    layer.attn = nn.Module()
    layer.attn.conv_weights = layer.weight[:, 1::2].detach()
    layer.aliases = {"tuple": (layer.weight.t().detach(),)}
    layer.register_buffer("scale", torch.tensor([2.0, 3.0], dtype=torch.float64))
    rotary = torch.ones(3)
    layer.register_buffer("rotary", rotary, persistent=False)
    original_parameter = layer.weight
    old_ptr = layer.weight.data_ptr()
    expected = layer.weight.detach().clone()
    expected_conv = layer.attn.conv_weights.clone()
    groups, size = weight_storage_plan(layer, device_type="cpu")
    assert len(groups) == 2  # Non-persistent shared rotary is not evicted.
    assert all(g["offset"] % 256 == 0 for g in groups.values())
    storage = torch.empty(size, dtype=torch.uint8).untyped_storage()
    copy_and_rebind_weight_storages(groups, storage)
    assert layer.weight is original_parameter
    assert layer.weight.data_ptr() != old_ptr
    torch.testing.assert_close(layer.weight, expected)
    torch.testing.assert_close(layer.attn.conv_weights, expected_conv)
    torch.testing.assert_close(layer.aliases["tuple"][0], expected.t())
    assert layer.attn.conv_weights.stride() == (6, 2)
    assert layer.rotary is rotary
    with torch.no_grad():
        layer.weight[0, 1] = 77
    assert layer.attn.conv_weights[0, 0].item() == 77
    assert layer.aliases["tuple"][0][1, 0].item() == 77
    assert layer.scale.dtype == torch.float64
    assert layer.scale.tolist() == [2, 3]


def test_vmm_eviction_keeps_virtual_address_and_retries_driver_failure(monkeypatch):
    calls = []
    fail = True

    def unmap(address, size):
        calls.append((address, size))
        return (1 if fail else 0,)

    driver = NS(CUresult=NS(CUDA_SUCCESS=0), cuMemUnmap=unmap)
    monkeypatch.setattr(vmm, "_get_cuda_driver", lambda: driver)
    reservation = object.__new__(vmm.VmmReservation)
    reservation.base, reservation.size = 1024, 4096
    reservation._closed = False
    reservation._mappings = [(1024, 4096, None)]
    with pytest.raises(ValueError, match="complete"):
        reservation.unmap(0, 2048)
    with pytest.raises(RuntimeError, match="cuMemUnmap"):
        reservation.unmap(0, 4096)
    assert len(reservation._mappings) == 1
    fail = False
    reservation.unmap(0, 4096)
    assert reservation.base == 1024 and not reservation._closed
    assert reservation._mappings == []
    assert calls == [(1024, 4096), (1024, 4096)]
    reservation._mappings = [(1024, 4096, "exported")]
    with pytest.raises(ValueError, match="retained"):
        reservation.unmap(0, 4096)
    assert len(calls) == 2


def test_eviction_drains_readers_and_reactivation_allocates_before_copy(monkeypatch):
    calls = []

    class Owner:
        def __init__(self):
            self.mapped = True
            self.size = 4096
            self.device_id = 0
            self.allocations = 1
            self.evictions = 0

        def evict(self):
            calls.append("unmap")
            self.mapped = False
            self.evictions += 1

        def restore(self):
            calls.append("map-empty")
            self.mapped = True
            self.allocations += 1

    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: calls.append("drain"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (10000, 20000))
    bank = EvictableWeightBank()
    bank.layers[3] = Owner()
    bank.evict([3])
    assert calls == ["drain", "unmap"]
    metrics = bank.metrics(set())
    assert metrics["mapped_physical_bytes"] == 0
    assert metrics["inactive_mapped_physical_bytes"] == 0
    assert metrics["reserved_virtual_bytes"] == 4096
    bank.prepare(3, None)
    assert calls[-1] == "map-empty"
    assert bank.metrics({3})["mapped_physical_bytes"] == 4096
    assert bank.metrics(set())["inactive_mapped_physical_bytes"] == 4096
