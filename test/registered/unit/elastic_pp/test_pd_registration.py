from types import SimpleNamespace as NS

import pytest

from sglang.srt.elastic_pp.pd_registration import (
    DecodeRegistrationTransaction,
    align_decode_hicache_polls_at_fence,
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
              enable_hierarchical_cache=True, pp_async_batch_depth=0,
              max_running_requests=1, pp_max_micro_batch_size=1)


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


def _optimized_args():
    args = _args()
    args.elastic_pp_transition_strategy = "background"
    args.elastic_pp_transition_backing = "overlap"
    args.disable_cuda_graph = False
    args.elastic_pp_graph_partitions = "24,0,24;16,16,16"
    args.elastic_pp_prepared_weight_policy = "offload"
    args.elastic_pp_hicache_shared_host = True
    args.elastic_pp_weight_staging = "cpu_prepacked"
    args.enable_elastic_pp_kv_patch = True
    args.elastic_pp_kv_patch_transport = "nixl"
    args.elastic_pp_kv_backend = "native"
    return args


def test_explicit_optimized_single_request_profile():
    validate_decode_args(_optimized_args())


@pytest.mark.parametrize("name,value", [
    ("elastic_pp_hicache_shared_host", False),
    ("elastic_pp_prepared_weight_policy", "reuse"),
    ("elastic_pp_graph_partitions", None),
    ("elastic_pp_weight_staging", "legacy"),
    ("enable_elastic_pp_kv_patch", False),
    ("elastic_pp_kv_patch_transport", "cpu"),
])
def test_incomplete_optimized_profiles_fail_closed(name, value):
    args = _optimized_args()
    setattr(args, name, value)
    with pytest.raises(ValueError):
        validate_decode_args(args)


@pytest.mark.parametrize("counts", [(4, 2, 2), (3, 3, 2), (3, 3, 3)])
def test_fence_catches_up_receivers_without_sending_an_extra_cache_round(counts):
    import concurrent.futures
    import threading

    condition = threading.Condition()
    rendezvous = threading.Barrier(3)
    sent = list(counts[:2])
    received = [0, counts[1], counts[2]]
    reports = [{"polls": count, "microsteps": count} for count in counts]

    def rank_worker(rank):
        def poll():
            with condition:
                if rank:
                    assert condition.wait_for(
                        lambda: received[rank] < sent[rank - 1], timeout=2
                    ), "the previous rank did not send this cache round"
                    received[rank] += 1
                if rank < 2:
                    sent[rank] += 1
                condition.notify_all()

        def gather(local):
            assert local == reports[rank]
            rendezvous.wait(timeout=2)
            return reports

        scheduler = NS(
            _pp_pd_hicache_polls=counts[rank], _pp_pd_microsteps=counts[rank],
            tree_cache=NS(check_hicache_events_pp_decode=poll,
                          check_hicache_events=lambda: pytest.fail("blocking poll")),
        )
        result = align_decode_hicache_polls_at_fence(scheduler, NS(all_gather_object=gather))
        # The old fence blocks here on a leading rank's unmatched isends.
        # With catch-up, every actual packet is consumed before proceeding.
        with condition:
            if rank < 2:
                assert condition.wait_for(lambda: received[rank + 1] == sent[rank], timeout=2)
        assert scheduler._pp_pd_hicache_polls == max(counts)
        assert scheduler._pp_pd_microsteps == max(counts)
        assert result["catchup_rounds"] == max(counts) - counts[rank]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(rank_worker, rank) for rank in range(3)]
        for future in futures:
            future.result(timeout=3)
    assert sent == [max(counts)] * 2
    assert received[1:] == [max(counts)] * 2


@pytest.mark.parametrize("name,value", [
    ("elastic_pp_disagg_decode", False), ("disaggregation_mode", "prefill"),
    ("disable_cuda_graph", False), ("enable_hierarchical_cache", False),
    ("elastic_pp_transition_strategy", "background"),
    ("elastic_pp_transition_backing", "overlap"),
    ("disaggregation_transfer_backend", "nixl"), ("pp_async_batch_depth", 1),
    ("max_running_requests", 2), ("pp_max_micro_batch_size", 2),
])
def test_unwired_modes_fail_before_loading(name, value):
    args = _args()
    setattr(args, name, value)
    with pytest.raises(ValueError, match=name):
        validate_decode_args(args)
