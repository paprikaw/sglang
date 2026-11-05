"""CPU regressions for the bounded PP2/MTP two-slot scheduler.

The queue test models stage-local execution and both directions of feedback,
not GPU performance. It makes the tail's first forward depend on the head
submitting the other independent slot; output-before-forward cannot satisfy
that dependency. Real CUDA/NCCL validation remains a separate guarded test.
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Event

import pytest
import torch

from sglang.srt.managers.pp_spec_microbatch import (
    launch_then_exchange,
    own_payload,
    validate_microbatch_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _valid_config():
    return dict(
        enabled=True,
        pp_size=2,
        pp_async_batch_depth=0,
        pp_spec_enabled=True,
        speculative_algorithm="EAGLE",
        topk=1,
        overlap_disabled=True,
        elastic=False,
        disaggregation_mode="null",
        dp_attention=False,
    )


def test_bounded_configuration_and_disabled_legacy_path():
    validate_microbatch_config(**_valid_config())
    # Disabled mode must not constrain the existing PP/scheduler paths.
    validate_microbatch_config(**dict(_valid_config(), enabled=False, pp_size=4))


@pytest.mark.parametrize(
    "override",
    [
        {"pp_size": 1},
        {"pp_size": 4},
        {"pp_async_batch_depth": 1},
        {"pp_spec_enabled": False},
        {"speculative_algorithm": "EAGLE3"},
        {"topk": 2},
        {"overlap_disabled": False},
        {"elastic": True},
        {"disaggregation_mode": "decode"},
        {"dp_attention": True},
    ],
)
def test_rejects_unvalidated_scheduler_variants(override):
    with pytest.raises(ValueError, match="PP2"):
        validate_microbatch_config(**dict(_valid_config(), **override))


def test_launch_proxy_and_exchange_order_including_empty_slot():
    calls = []
    result, feedback = launch_then_exchange(
        lambda: calls.append("launch") or "proxy",
        lambda proxy: calls.append(("send", proxy)),
        lambda: calls.append("exchange") or "feedback",
    )
    assert calls == ["launch", ("send", "proxy"), "exchange"]
    assert (result, feedback) == ("proxy", "feedback")
    calls.clear()
    assert launch_then_exchange(
        None,
        lambda _: pytest.fail("empty slot submitted a proxy"),
        lambda: calls.append("drain") or "last-feedback",
    ) == (None, "last-feedback")
    assert calls == ["drain"]


def test_payload_owns_views_before_graph_buffer_reuse():
    graph = torch.arange(24).reshape(6, 4)
    metadata = ("slot-A", 3, ("r0", "r1"))
    payload = own_payload({"hidden": graph[1:4], "metadata": metadata})
    expected = payload["hidden"].clone()
    graph.fill_(-1)
    assert torch.equal(payload["hidden"], expected)
    assert payload["hidden"].data_ptr() != graph[1:4].data_ptr()
    assert payload["metadata"] is metadata


def test_two_b4_slots_preserve_feedback_ancestry_and_drain():
    rounds = 3
    count = 2 * rounds
    # These model separate activation and feedback transports. The echo is the
    # same feedback transport in the other direction, retained by the ring.
    proxies, feedback, echo = Queue(), Queue(), Queue()
    second_head_slot_launched = Event()
    observed = []
    head_seq = {slot: torch.full((4,), 100, dtype=torch.int64) for slot in (0, 1)}
    tail_seq = {slot: torch.full((4,), 100, dtype=torch.int64) for slot in (0, 1)}

    def rids(slot):
        return tuple(f"slot{slot}-r{row}" for row in range(4))

    def chain(slot, round_id):
        # Root and two drafts per request, matching B4 x draft-width3.
        return torch.arange(12).reshape(4, 3) + slot * 1000 + round_id * 100

    def receive(queue):
        # A bug must fail finitely rather than leave CI hanging forever.
        return queue.get(timeout=3)

    def head():
        pending_echo = None
        graph_hidden = torch.empty((12, 4), dtype=torch.int64)
        for step in range(count + 2):
            slot, round_id = step % 2, step // 2

            def launch():
                if step == 1:
                    second_head_slot_launched.set()
                expected_seq = 100 + round_id * torch.tensor([2, 3, 2, 3])
                assert torch.equal(head_seq[slot], expected_seq)
                graph_hidden.fill_(step)
                observed.append(("head-launch", slot, round_id))
                return {
                    "slot": slot,
                    "round": round_id,
                    "rids": rids(slot),
                    "chain": chain(slot, round_id),
                    "seq": head_seq[slot],
                    "hidden": graph_hidden,
                }

            def exchange():
                nonlocal pending_echo
                if pending_echo is not None:
                    echo.put(own_payload(pending_echo))
                    pending_echo = None
                if 0 <= step - 1 < count:
                    result = receive(feedback)
                    prev_slot, prev_round = (step - 1) % 2, (step - 1) // 2
                    assert (result["slot"], result["round"]) == (
                        prev_slot, prev_round
                    )
                    assert result["rids"] == rids(prev_slot)
                    assert torch.equal(
                        result["next_chain"], chain(prev_slot, prev_round + 1)
                    )
                    head_seq[prev_slot] = result["new_seq"].clone()
                    pending_echo = result
                    observed.append(("head-retire", prev_slot, prev_round))
                    return result

            launch_then_exchange(
                launch if step < count else None,
                lambda payload: proxies.put(own_payload(payload)),
                exchange,
            )
        assert pending_echo is None

    def tail():
        graph_new_seq = torch.empty(4, dtype=torch.int64)
        graph_chain = torch.empty((4, 3), dtype=torch.int64)
        for step in range(count + 1):
            slot, round_id = step % 2, step // 2
            packet = receive(proxies) if step < count else None
            produced = None

            def launch():
                nonlocal produced
                assert packet is not None
                if step == 0:
                    assert second_head_slot_launched.wait(timeout=3), (
                        "head waited for slot-A feedback before submitting slot-B"
                    )
                assert (packet["slot"], packet["round"]) == (slot, round_id)
                assert packet["rids"] == rids(slot)
                assert torch.equal(packet["hidden"], torch.full((12, 4), step))
                assert torch.equal(packet["chain"], chain(slot, round_id))
                assert torch.equal(packet["seq"], tail_seq[slot])
                graph_new_seq.copy_(packet["seq"] + torch.tensor([2, 3, 2, 3]))
                graph_chain.copy_(chain(slot, round_id + 1))
                produced = own_payload(
                    {
                        "slot": slot,
                        "round": round_id,
                        "rids": rids(slot),
                        "new_seq": graph_new_seq,
                        "next_chain": graph_chain,
                    }
                )
                observed.append(("tail-launch", slot, round_id))
                return produced

            def exchange():
                if step > 0:
                    returned = receive(echo)
                    prev_slot, prev_round = (step - 1) % 2, (step - 1) // 2
                    assert (returned["slot"], returned["round"]) == (
                        prev_slot, prev_round
                    )
                    assert returned["rids"] == rids(prev_slot)
                    tail_seq[prev_slot] = returned["new_seq"].clone()
                    observed.append(("tail-retire", prev_slot, prev_round))
                if produced is not None:
                    feedback.put(produced)

            launch_then_exchange(
                launch if step < count else None, lambda _: None, exchange
            )

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(head), workers.submit(tail)]
        for future in futures:
            future.result(timeout=10)
    expected = 100 + rounds * torch.tensor([2, 3, 2, 3])
    assert all(torch.equal(value, expected) for value in head_seq.values())
    assert all(torch.equal(value, expected) for value in tail_seq.values())
    assert proxies.empty() and feedback.empty() and echo.empty()
    counts = Counter(item[0] for item in observed)
    assert counts == {
        "head-launch": count,
        "tail-launch": count,
        "head-retire": count,
        "tail-retire": count,
    }
    assert observed.index(("head-launch", 1, 0)) < observed.index(
        ("head-retire", 0, 0)
    )
