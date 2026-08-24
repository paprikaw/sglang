import pytest

from sglang.srt.elastic_pp.kv_residency import ElasticKVResidency
from sglang.srt.elastic_pp.topology import LayerRange
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_kv_residency_requires_copy_before_publish_and_cleanup_after_commit():
    state = ElasticKVResidency(
        num_layers=4,
        initial_generation=0,
        initial_range=LayerRange(0, 2),
    )
    prepared = state.begin_prepare(
        generation=1,
        target=LayerRange(1, 4),
        resident_layer_ids={0, 1},
    )
    assert prepared.new_layer_ids == frozenset({2, 3})
    assert state.active_range == LayerRange(0, 2)

    state.mark_loaded(1, {2})
    assert not state.ready_to_commit(1)
    with pytest.raises(RuntimeError, match=r"missing=\[3\]"):
        state.commit(1)

    state.mark_loaded(1, {3})
    assert state.commit(1) == LayerRange(1, 4)
    assert state.cleanup_candidates(1, {0, 1, 2, 3}) == frozenset({0})
    with pytest.raises(RuntimeError, match=r"missing=\[0\]"):
        state.finish_cleanup(1, set())
    state.finish_cleanup(1, {0})


def test_kv_residency_abort_keeps_committed_interval():
    state = ElasticKVResidency(
        num_layers=4,
        initial_generation=2,
        initial_range=LayerRange(2, 4),
    )
    state.begin_prepare(
        generation=3,
        target=LayerRange(0, 2),
        resident_layer_ids={2, 3},
    )
    assert state.abort(3) == frozenset({0, 1})
    assert state.generation == 2
    assert state.active_range == LayerRange(2, 4)
