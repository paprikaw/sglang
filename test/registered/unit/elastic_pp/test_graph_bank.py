from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.elastic_pp.graph_bank import ElasticGraphBank, GraphKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Event:
    def __init__(self):
        self.ready = False

    def query(self):
        return self.ready

    def synchronize(self):
        self.ready = True


class Backend:
    def __init__(self):
        self.events = []
        self.graphs = []
        self.synchronize = Mock()

    def capture(self, forward):
        output = forward()
        graph = SimpleNamespace(replay=Mock(side_effect=forward))
        self.graphs.append(graph)
        return graph, output

    def completion(self):
        event = Event()
        self.events.append(event)
        return event


@pytest.fixture
def setup():
    backend = Backend()
    bank = ElasticGraphBank(backend=backend, max_entries=2)
    key = GraphKey((2, 0, 2), 1)
    bindings = {"kv": torch.zeros(8, 2)}
    forward = lambda: bindings["kv"].add_(1)
    bank.capture(key, bindings=bindings, forward=forward)
    bank.activate()
    return bank, backend, key, bindings, forward


def test_replay_uses_captured_graph_and_preserves_page_lease_until_event(setup):
    bank, backend, key, bindings, eager = setup
    releases = []

    @contextmanager
    def lease():
        releases.append("pin")
        try:
            yield
        finally:
            releases.append("unpin")

    result = bank.run(key, bindings=bindings, eager=eager, lease=lease)
    assert torch.all(result == 2)
    assert releases == ["pin"]
    bank.poll()
    assert releases == ["pin"]
    backend.events[0].ready = True
    bank.poll()
    bank.poll()
    assert releases == ["pin", "unpin"]
    assert bank.counters == {"captures": 1, "replays": 1}


@pytest.mark.parametrize("change", ["pointer", "stride", "dtype", "shape"])
def test_changed_binding_uses_eager_without_replaying_stale_graph(setup, change):
    bank, backend, key, bindings, _ = setup
    old = bindings["kv"]
    replacement = {
        "pointer": old.clone(),
        "stride": old.as_strided((8, 2), (1, 8)),
        "dtype": old.to(torch.float64),
        "shape": old.view(4, 4),
    }[change]
    eager = Mock(return_value="eager")
    assert bank.run(key, bindings={"kv": replacement}, eager=eager) == "eager"
    backend.graphs[0].replay.assert_not_called()
    assert bank.counters["fallback_bindings"] == 1


def test_capture_after_activation_is_forbidden_and_unknown_layout_falls_back(setup):
    bank, backend, key, bindings, forward = setup
    target = GraphKey((1, 1, 2), 1)
    with pytest.raises(RuntimeError, match="before activation"):
        bank.capture(target, bindings=bindings, forward=forward)
    assert bank.run(target, bindings=bindings, eager=lambda: 17) == 17
    assert bank.counters["fallback_uncaptured"] == 1


def test_eager_fallback_keeps_page_lease_until_gpu_completion(setup):
    bank, backend, _, bindings, _ = setup
    released = []

    @contextmanager
    def lease():
        try:
            yield
        finally:
            released.append(True)

    assert (
        bank.run(
            GraphKey((1, 1, 2), 1), bindings=bindings, eager=lambda: 17, lease=lease
        )
        == 17
    )
    assert not released
    backend.events[0].ready = True
    bank.poll()
    assert released == [True]


def test_unmapped_pages_block_both_replay_and_eager(setup):
    bank, backend, key, bindings, _ = setup
    eager = Mock()
    with pytest.raises(RuntimeError, match="not mapped"):
        bank.run(key, bindings=bindings, eager=eager, pages_ready=False)
    eager.assert_not_called()
    backend.graphs[0].replay.assert_not_called()


def test_replay_failure_is_not_executed_a_second_time_via_eager(setup):
    bank, backend, key, bindings, _ = setup
    backend.graphs[0].replay.side_effect = RuntimeError("launch failed")
    eager = Mock()
    with pytest.raises(RuntimeError, match="launch failed"):
        bank.run(key, bindings=bindings, eager=eager)
    eager.assert_not_called()
    backend.synchronize.assert_called_once()


def test_close_drains_outstanding_replays_once(setup):
    bank, backend, key, bindings, eager = setup
    bank.run(key, bindings=bindings, eager=eager)
    bank.close()
    assert backend.events[0].ready
    bank.close()
    with pytest.raises(RuntimeError, match="not active"):
        bank.run(key, bindings=bindings, eager=eager)


def test_capture_requires_mapped_pages_and_obeys_capacity():
    backend = Backend()
    bank = ElasticGraphBank(backend=backend, max_entries=1)
    bindings = {"kv": torch.zeros(1)}
    key = GraphKey((1, 1), 1)
    with pytest.raises(RuntimeError, match="mapped"):
        bank.capture(key, bindings=bindings, forward=lambda: 1, pages_ready=False)
    bank.capture(key, bindings=bindings, forward=lambda: 1)
    with pytest.raises(ValueError, match="capacity"):
        bank.capture(GraphKey((1, 0, 1), 1), bindings=bindings, forward=lambda: 1)
