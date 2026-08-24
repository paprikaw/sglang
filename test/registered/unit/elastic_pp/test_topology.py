import pytest

from sglang.srt.elastic_pp.topology import (
    DynamicPPTransition,
    DynamicPPTransitionState,
    LayerRange,
    PPTopology,
    TransitionPhase,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_topology_routes_over_nonempty_resident_ranks():
    topology = PPTopology.from_layer_counts([4, 0, 4, 0], generation=3)

    assert topology.active_ranks == (0, 2)
    assert topology.num_layers == 8
    assert topology.first_rank == 0
    assert topology.last_rank == 2
    assert topology.prev_rank(2) == 0
    assert topology.next_rank(0) == 2
    assert topology.owner(6) == 2


def test_topology_rejects_noncontiguous_pipeline_order():
    with pytest.raises(ValueError, match="contiguous"):
        PPTopology(
            generation=0,
            max_pp_size=2,
            active_ranks=(1, 0),
            layer_ranges=(LayerRange(0, 4), LayerRange(4, 8)),
        )


def test_pp2_to_pp4_transition_builds_exact_transfer_and_weight_plan():
    source = PPTopology.from_layer_counts([4, 4, 0, 0], generation=0)
    target = PPTopology.from_layer_counts([2, 2, 2, 2], generation=1)

    transition = DynamicPPTransition.build(source, target)

    assert [
        (item.src_rank, item.dst_rank, item.layer_range)
        for item in transition.kv_transfers
    ] == [
        (0, 1, LayerRange(2, 4)),
        (1, 2, LayerRange(4, 6)),
        (1, 3, LayerRange(6, 8)),
    ]
    assert transition.weight_loads == (
        (1, LayerRange(2, 4)),
        (2, LayerRange(4, 6)),
        (3, LayerRange(6, 8)),
    )
    assert transition.ranks_to_activate == frozenset({2, 3})
    assert transition.required_weight_ranks == frozenset({1, 2, 3})
    assert transition.required_kv_ranks == frozenset({1, 2, 3})
    assert transition.required_cleanup_ranks == frozenset({0, 1})


def test_commit_waits_for_prepare_and_every_resident_rank_fence():
    source = PPTopology.from_layer_counts([2, 2, 2, 2], generation=7)
    target = PPTopology.from_layer_counts([4, 4, 0, 0], generation=8)
    state = DynamicPPTransitionState(DynamicPPTransition.build(source, target))

    assert state.phase == TransitionPhase.PREPARING
    for rank in sorted(state.transition.required_weight_ranks):
        state.mark_weights_ready(rank)
    for rank in sorted(state.transition.required_kv_ranks):
        state.mark_kv_ready(rank)
    assert state.phase == TransitionPhase.READY_TO_FENCE

    state.begin_fence()
    for rank in (0, 1, 2):
        state.mark_fence_ready(rank)
    with pytest.raises(RuntimeError, match="every resident rank"):
        state.commit()

    state.mark_fence_ready(3)
    assert state.commit() == target
    assert state.phase == TransitionPhase.CLEANING

    for rank in sorted(state.transition.required_cleanup_ranks):
        state.mark_cleanup_ready(rank)
    assert state.phase == TransitionPhase.DONE


def test_transition_rejects_stale_generation_and_model_size_change():
    source = PPTopology.from_layer_counts([4, 4], generation=2)

    with pytest.raises(ValueError, match="newer"):
        DynamicPPTransition.build(
            source,
            PPTopology.from_layer_counts([2, 6], generation=2),
        )
    with pytest.raises(ValueError, match="same model layers"):
        DynamicPPTransition.build(
            source,
            PPTopology.from_layer_counts([4, 5], generation=3),
        )
