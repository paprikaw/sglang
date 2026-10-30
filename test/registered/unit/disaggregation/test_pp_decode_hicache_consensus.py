from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.decode_hicache_mixin import HiCacheRestoreResult
from sglang.srt.managers import scheduler_pp_mixin as pp


@pytest.mark.parametrize('first_rank', [True, False])
def test_pp_waits_for_local_restore_before_releasing_transferred_request(monkeypatch, first_rank):
    req = SimpleNamespace(req=SimpleNamespace(rid='pending-host'),
                          kv_receiver=SimpleNamespace(poll=lambda: KVPoll.Success),
                          hicache_restore_status=HiCacheRestoreResult.PENDING)
    queue = SimpleNamespace(queue=[req], _process_hicache_local_restores=Mock())
    scheduler = SimpleNamespace(
        enable_decode_hicache=True, disagg_decode_transfer_queue=queue,
        pp_group=SimpleNamespace(is_first_rank=first_rank),
        attn_cp_cpu_group=None, attn_tp_cpu_group=None,
        _pp_recv_pyobj_from_prev_stage=lambda: ['pending-host'])
    scheduler.get_rids = pp.SchedulerPPMixin.get_rids.__get__(scheduler)
    monkeypatch.setattr(pp, 'poll_and_all_reduce_attn_cp_tp_group',
                        lambda receivers, *_: [receiver.poll() for receiver in receivers])
    collect = pp.SchedulerPPMixin._pp_pd_get_decode_transferred_ids
    assert collect(scheduler) == []
    queue._process_hicache_local_restores.assert_called_once_with([req])
    req.hicache_restore_status = HiCacheRestoreResult.READY
    assert collect(scheduler) == ['pending-host']


def test_preallocation_poll_is_not_blocked_by_future_host_restore(monkeypatch):
    req = SimpleNamespace(req=SimpleNamespace(rid='preallocate'),
                          kv_receiver=SimpleNamespace(poll=lambda: KVPoll.WaitingForInput),
                          hicache_restore_status=HiCacheRestoreResult.PENDING)
    scheduler = SimpleNamespace(attn_cp_cpu_group=None, attn_tp_cpu_group=None)
    monkeypatch.setattr(pp, 'poll_and_all_reduce_attn_cp_tp_group',
                        lambda receivers, *_: [receiver.poll() for receiver in receivers])
    assert pp.SchedulerPPMixin.get_rids(scheduler, [req], False,
                                      [KVPoll.WaitingForInput]) == ['preallocate']
