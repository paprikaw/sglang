"""A promised host prefix must wait for capacity, never run partly restored."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.disaggregation import decode_hicache_mixin as hc
from sglang.srt.disaggregation.base import KVPoll
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def fixture(monkeypatch, *, device=0, host=4):
    clock = [10.0]
    monkeypatch.setattr(hc, "time", NS(monotonic=lambda: clock[0]), raising=False)
    monkeypatch.delenv("SGLANG_DISAGGREGATION_WAITING_TIMEOUT", raising=False)
    match = NS(device_indices=torch.arange(device), host_hit_length=host,
               best_match_node=2, last_device_node=1)
    rematch = Mock(return_value=match)
    monkeypatch.setattr(hc, "match_prefix_for_req", rematch)
    empty = torch.tensor([], dtype=torch.int64)
    req = NS(rid="host-pressure", origin_input_ids=[1, 2, 3, 4, 5],
             mamba_pool_idx=torch.tensor(7))
    dr = NS(req=req, prefix_match=hc.DecodePrefixMatch(empty, 4, 0, 1),
            hicache_restore_status=hc.HiCacheRestoreResult.PENDING,
            hicache_restored_node=None, hicache_restored_kv_indices=None,
            hicache_load_consumer_index=-1, hicache_load_back_retry_started=None,
            kv_receiver=NS(poll=lambda: KVPoll.Success))
    cache = NS(init_load_back=Mock(return_value=(empty, 1)),
               supports_mamba=lambda: True, inc_lock_ref=Mock(),
               dec_lock_ref=Mock(),
               is_load_back_event_done=Mock(return_value=False),
               cache_controller=NS(layer_done_counter=NS(producer_index=0, num_counters=3)),
               ready_to_load_host_cache=Mock(return_value=1))
    queue = hc.DecodeHiCacheTransferMixin()
    queue.tree_cache = cache
    return queue, dr, cache, match, clock


def test_capacity_failure_waits_then_restores_without_touching_transfer_state(monkeypatch):
    queue, dr, cache, _, _ = fixture(monkeypatch)
    assert not queue._try_hicache_queue_load_back(dr)
    assert dr.hicache_restore_status is hc.HiCacheRestoreResult.PENDING
    assert dr.hicache_restored_node is None
    cache.inc_lock_ref.assert_not_called()
    assert hc.HiCacheRestoreGatedKVReceiver(dr).poll() is KVPoll.Transferring
    cache.init_load_back.return_value = (torch.arange(4), 2)
    assert queue._try_hicache_queue_load_back(dr)
    assert dr.hicache_restored_kv_indices.tolist() == [0, 1, 2, 3]
    cache.inc_lock_ref.assert_called_once_with(2)
    assert dr.req.mamba_pool_idx.item() == 7
    assert cache.init_load_back.call_args.args[0].copy_mamba_to_request is False
    # DMA queued is not DMA complete; the existing event gate must still hold.
    assert hc.HiCacheRestoreGatedKVReceiver(dr).poll() is KVPoll.Transferring
    cache.is_load_back_event_done.return_value = True
    queue._process_hicache_local_restores([dr])
    assert dr.hicache_restore_status is hc.HiCacheRestoreResult.READY
    assert hc.HiCacheRestoreGatedKVReceiver(dr).poll() is KVPoll.Success


def test_missing_promised_host_prefix_is_not_retried_or_used(monkeypatch):
    queue, dr, cache, _, _ = fixture(monkeypatch, host=2)
    assert not queue._try_hicache_queue_load_back(dr)
    assert dr.hicache_restore_status is hc.HiCacheRestoreResult.FAILED
    cache.init_load_back.assert_not_called()
    cache.inc_lock_ref.assert_not_called()


def test_capacity_retry_has_a_finite_timeout(monkeypatch):
    queue, dr, cache, _, clock = fixture(monkeypatch)
    assert not queue._try_hicache_queue_load_back(dr)
    assert dr.hicache_restore_status is hc.HiCacheRestoreResult.PENDING
    clock[0] += 301
    assert not queue._try_hicache_queue_load_back(dr)
    assert dr.hicache_restore_status is hc.HiCacheRestoreResult.FAILED
    cache.inc_lock_ref.assert_not_called()


def test_partial_restoration_does_not_commit_incomplete_prefix(monkeypatch):
    queue, dr, cache, _, _ = fixture(monkeypatch)
    cache.init_load_back.return_value = (torch.arange(2), 2)
    assert not queue._try_hicache_queue_load_back(dr)
    assert dr.hicache_restore_status is hc.HiCacheRestoreResult.FAILED
    cache.inc_lock_ref.assert_not_called()


def test_prefix_now_resident_on_device_needs_no_dma(monkeypatch):
    queue, dr, cache, _, _ = fixture(monkeypatch, device=4, host=0)
    assert not queue._try_hicache_queue_load_back(dr)
    assert dr.hicache_restore_status is hc.HiCacheRestoreResult.READY
    assert dr.hicache_restored_kv_indices.tolist() == [0, 1, 2, 3]
    cache.inc_lock_ref.assert_called_once_with(1)


def test_transport_failure_still_propagates_during_local_retry(monkeypatch):
    _, dr, _, _, _ = fixture(monkeypatch)
    dr.kv_receiver.poll = lambda: KVPoll.Failed
    assert hc.HiCacheRestoreGatedKVReceiver(dr).poll() is KVPoll.Failed
