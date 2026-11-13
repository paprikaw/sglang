from types import SimpleNamespace as NS

import pytest

from sglang.srt.elastic_pp.pd_registration import (
    DecodeRegistrationTransaction,
    pending_pd_work,
    validate_decode_args,
)


def _scheduler():
    events = []
    manager = NS(
        kv_args="old",
        deregister_buffer_to_engine=lambda: events.append("deregister"),
        register_buffer_to_engine=lambda: events.append(("register", manager.kv_args)),
    )
    prealloc = NS(
        queue=[], pending_reqs=[], retracted_queue=[], kv_manager=manager,
        _build_kv_args=lambda: "new",
    )
    transfer = NS(queue=[], has_pending_deferred_releases=lambda: False)
    return NS(disagg_decode_prealloc_queue=prealloc,
              disagg_decode_transfer_queue=transfer), events


def test_registration_uses_new_addresses_only_after_deregister():
    scheduler, events = _scheduler()
    transaction = DecodeRegistrationTransaction(scheduler)
    transaction.suspend()
    transaction.refresh()
    transaction.refresh()
    assert events == ["deregister", ("register", "new")]
    assert not transaction.suspended


@pytest.mark.parametrize("queue_name,field", [
    ("disagg_decode_prealloc_queue", "queue"),
    ("disagg_decode_prealloc_queue", "pending_reqs"),
    ("disagg_decode_prealloc_queue", "retracted_queue"),
    ("disagg_decode_transfer_queue", "queue"),
])
def test_no_buffer_is_deregistered_while_a_writer_can_be_live(queue_name, field):
    scheduler, events = _scheduler()
    getattr(getattr(scheduler, queue_name), field).append(object())
    assert pending_pd_work(scheduler) == 1
    with pytest.raises(RuntimeError, match="transfers must finish"):
        DecodeRegistrationTransaction(scheduler).suspend()
    assert not events


def test_deferred_abort_ack_also_prevents_backing_release():
    scheduler, events = _scheduler()
    scheduler.disagg_decode_transfer_queue.has_pending_deferred_releases = lambda: True
    with pytest.raises(RuntimeError):
        DecodeRegistrationTransaction(scheduler).suspend()
    assert not events


def test_registration_failure_keeps_transaction_suspended():
    scheduler, _ = _scheduler()
    transaction = DecodeRegistrationTransaction(scheduler)
    transaction.suspend()

    def fail():
        raise RuntimeError("injected transport registration failure")

    scheduler.disagg_decode_prealloc_queue.kv_manager.register_buffer_to_engine = fail
    with pytest.raises(RuntimeError):
        transaction.refresh()
    assert transaction.suspended


def _args():
    return NS(elastic_pp_disagg_decode=True, disaggregation_mode="decode",
              disaggregation_transfer_backend="mooncake",
              elastic_pp_transition_strategy="sync",
              elastic_pp_transition_backing="cpu_spill", disable_cuda_graph=True,
              enable_hierarchical_cache=True, pp_async_batch_depth=0)


def test_two_senders_cover_a_repartitioned_or_dormant_decode_stage():
    from sglang.srt.disaggregation.utils import build_transfer_entry_pairs

    def pairs(source, target):
        return build_transfer_entry_pairs(source, target, len(source), len(target),
                                          allow_partial_overlap=True)

    assert pairs([0, 1, 0, 1], [1, 2, 1, 2]) == [(1, 0), (3, 2)]
    assert pairs([2, 3, 2, 3], [1, 2, 1, 2]) == [(0, 1), (2, 3)]
    assert pairs([0, 1, 0, 1], []) == []
    with pytest.raises(RuntimeError, match="missing a transfer entry"):
        pairs([1, 1, 1], [1, 1])


def test_explicit_narrow_profile():
    validate_decode_args(_args())


@pytest.mark.parametrize("name,value", [
    ("elastic_pp_disagg_decode", False), ("disaggregation_mode", "prefill"),
    ("disable_cuda_graph", False), ("enable_hierarchical_cache", False),
    ("elastic_pp_transition_strategy", "background"),
    ("elastic_pp_transition_backing", "overlap"),
    ("disaggregation_transfer_backend", "nixl"), ("pp_async_batch_depth", 1),
])
def test_unwired_modes_fail_before_loading(name, value):
    args = _args()
    setattr(args, name, value)
    with pytest.raises(ValueError, match=name):
        validate_decode_args(args)
