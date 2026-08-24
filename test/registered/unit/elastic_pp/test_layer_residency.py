import pytest

from sglang.srt.elastic_pp.layer_residency import ElasticLayerResidency
from sglang.srt.elastic_pp.topology import LayerRange
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_prepare_commit_and_post_commit_cleanup_are_generation_safe():
    state = ElasticLayerResidency(
        num_layers=8,
        initial_generation=2,
        initial_range=LayerRange(0, 4),
    )
    preparation = state.begin_prepare(
        generation=3,
        target=LayerRange(2, 6),
        resident_layer_ids={0, 1, 2, 3},
    )

    assert preparation.new_layer_ids == frozenset({4, 5})
    assert not state.ready_to_commit(3)
    state.mark_loaded(3, {4})
    with pytest.raises(RuntimeError, match=r"missing=\[5\]"):
        state.commit(3)
    state.mark_loaded(3, {5})

    assert state.commit(3) == LayerRange(2, 6)
    assert state.generation == 3
    assert state.cleanup_candidates(3, {0, 1, 2, 3, 4, 5}) == frozenset({0, 1})
    with pytest.raises(RuntimeError, match="deferred cleanup"):
        state.begin_prepare(
            generation=4,
            target=LayerRange(0, 4),
            resident_layer_ids={0, 1, 2, 3, 4, 5},
        )
    state.finish_cleanup(3, {0, 1})
    state.begin_prepare(
        generation=4,
        target=LayerRange(0, 4),
        resident_layer_ids={2, 3, 4, 5},
    )


def test_preparation_is_idempotent_but_rejects_competing_generation():
    state = ElasticLayerResidency(
        num_layers=4,
        initial_generation=0,
        initial_range=LayerRange(0, 2),
    )
    first = state.begin_prepare(
        generation=1,
        target=LayerRange(0, 4),
        resident_layer_ids={0, 1},
    )
    assert (
        state.begin_prepare(
            generation=1,
            target=LayerRange(0, 4),
            resident_layer_ids={0, 1},
        )
        is first
    )

    with pytest.raises(RuntimeError, match="already preparing"):
        state.begin_prepare(
            generation=2,
            target=LayerRange(0, 3),
            resident_layer_ids={0, 1},
        )


def test_abort_returns_only_layers_created_for_pending_generation():
    state = ElasticLayerResidency(
        num_layers=6,
        initial_generation=4,
        initial_range=LayerRange(0, 3),
    )
    state.begin_prepare(
        generation=5,
        target=LayerRange(1, 5),
        resident_layer_ids={0, 1, 2},
    )

    assert state.abort(5) == frozenset({3, 4})
    assert state.generation == 4
    assert state.scheduled_range == LayerRange(0, 3)
