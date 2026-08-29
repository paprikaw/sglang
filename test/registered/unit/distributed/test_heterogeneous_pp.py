from collections import defaultdict, deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.distributed.heterogeneous_pp import (
    HeterogeneousPPTopology,
    incompatible_attention_tp_sizes,
    parse_pp_layer_partition,
)
from sglang.srt.distributed import parallel_state
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin


@pytest.mark.parametrize(
    "raw,tp_groups,leaders,world_size",
    [
        ("4,1", ((0, 1, 2, 3), (4,)), (0, 4), 5),
        ("4,2", ((0, 1, 2, 3), (4, 5)), (0, 4), 6),
    ],
)
def test_topology_groups(raw, tp_groups, leaders, world_size):
    topology = HeterogeneousPPTopology.parse(raw)

    assert topology.tp_groups == tp_groups
    assert topology.stage_leader_ranks == leaders
    assert topology.world_size == world_size
    assert topology.pp_transport_groups[0] == leaders
    assert sorted(
        rank for group in topology.pp_transport_groups for rank in group
    ) == list(range(world_size))


def test_rank_mapping_round_trip():
    topology = HeterogeneousPPTopology.parse("4,2")

    for global_rank in range(topology.world_size):
        pp_rank, tp_rank = topology.stage_for_global_rank(global_rank)
        assert topology.global_rank(pp_rank, tp_rank) == global_rank


def test_single_node_placements_use_consecutive_gpus():
    topology = HeterogeneousPPTopology.parse("4,1")

    placements = topology.placements_for_node(nnodes=1, node_rank=0)

    assert [placement.global_rank for placement in placements] == [0, 1, 2, 3, 4]
    assert [placement.local_gpu_index for placement in placements] == [0, 1, 2, 3, 4]
    assert [(placement.pp_rank, placement.tp_rank) for placement in placements] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
    ]


def test_one_stage_per_node_placements():
    topology = HeterogeneousPPTopology.parse("4,2")

    node0 = topology.placements_for_node(nnodes=2, node_rank=0)
    node1 = topology.placements_for_node(nnodes=2, node_rank=1)

    assert [(item.global_rank, item.local_gpu_index) for item in node0] == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]
    assert [(item.global_rank, item.local_gpu_index) for item in node1] == [
        (4, 0),
        (5, 1),
    ]


@pytest.mark.parametrize("raw", ["", "4,0", "4,-1", "4,x", "1"])
def test_invalid_topology_rejected(raw):
    with pytest.raises(ValueError):
        HeterogeneousPPTopology.parse(raw)


def test_unsupported_partial_stage_placement_rejected():
    topology = HeterogeneousPPTopology.parse("4,1")

    with pytest.raises(ValueError, match="one complete stage per node"):
        topology.placements_for_node(nnodes=3, node_rank=0)


def test_layer_partition_validation():
    assert parse_pp_layer_partition("73,19", pp_size=2, num_hidden_layers=92) == (
        73,
        19,
    )

    with pytest.raises(ValueError, match="one layer count per stage"):
        parse_pp_layer_partition("92", pp_size=2, num_hidden_layers=92)
    with pytest.raises(ValueError, match="cover every transformer layer"):
        parse_pp_layer_partition("72,19", pp_size=2, num_hidden_layers=92)


def test_glm47_attention_tp_compatibility_rejects_tp6():
    assert (
        incompatible_attention_tp_sizes(
            (1, 2, 4, 8),
            num_attention_heads=96,
            num_key_value_heads=8,
        )
        == ()
    )
    assert incompatible_attention_tp_sizes(
        (4, 6),
        num_attention_heads=96,
        num_key_value_heads=8,
    ) == (6,)


def test_logical_pipeline_identity_is_independent_of_physical_group():
    coordinator = parallel_state.GroupCoordinator.__new__(
        parallel_state.GroupCoordinator
    )
    coordinator.ranks = [2]
    coordinator.rank = 2
    coordinator.rank_in_group = 0
    coordinator.world_size = 1
    coordinator._routing_ranks = None

    coordinator.set_logical_pipeline_topology(rank=0, size=2, leader_ranks=(0, 4))

    assert coordinator.world_size == 1
    assert coordinator.logical_world_size == 2
    assert coordinator.logical_rank_in_group == 0
    assert coordinator.is_first_rank
    assert not coordinator.is_last_rank
    assert coordinator.next_rank == 4
    assert coordinator.prev_rank == 4


def test_heterogeneous_parallel_group_construction():
    created_groups = {}

    def mock_init_model_parallel_group(group_ranks, local_rank, backend, **kwargs):
        group_name = kwargs.get("group_name", "unknown")
        created_groups[group_name] = group_ranks
        group = Mock()
        group.device_group = Mock()
        group.pynccl_comm = None
        return group

    globals_to_reset = (
        "_TP",
        "_DCP",
        "_ATTN_CP",
        "_ATTN_TP",
        "_MOE_DP",
        "_MOE_EP",
        "_MOE_TP",
        "_PP",
        "_ELASTIC_PP_MIGRATION",
    )
    reset_patches = [
        patch.object(parallel_state, name, None) for name in globals_to_reset
    ]
    for reset_patch in reset_patches:
        reset_patch.start()
    try:
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=5),
            patch("torch.distributed.get_backend", return_value="nccl"),
            patch.object(
                parallel_state,
                "init_model_parallel_group",
                side_effect=mock_init_model_parallel_group,
            ),
            patch.object(parallel_state, "get_world_group") as get_world_group,
        ):
            world_group = Mock()
            world_group.device_group = Mock()
            world_group.local_rank = 0
            get_world_group.return_value = world_group

            parallel_state.initialize_heterogeneous_model_parallel(
                stage_tp_sizes=(4, 1), pipeline_rank=0
            )

        assert created_groups["tp"] == [[0, 1, 2, 3], [4]]
        assert created_groups["heterogeneous_singleton"] == [
            [0],
            [1],
            [2],
            [3],
            [4],
        ]
        assert created_groups["pp"] == [[0, 4], [1], [2], [3]]
        parallel_state._PP.set_logical_pipeline_topology.assert_called_once_with(
            rank=0, size=2, leader_ranks=(0, 4)
        )
    finally:
        for reset_patch in reversed(reset_patches):
            reset_patch.stop()


def test_nonleader_does_not_send_pipeline_tensor():
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(pp_stage_tp_sizes="4,2"),
        ps=SimpleNamespace(attn_tp_rank=1),
        _pp_topology=lambda: None,
        require_attn_tp_allgather=False,
        pp_group=Mock(),
    )

    work = SchedulerPPMixin._pp_send_dict_to_next_stage(
        scheduler, {"hidden_states": object()}, msg_type="proxy"
    )

    assert work == []
    scheduler.pp_group.send_tensor_dict.assert_not_called()


def test_nonleader_receives_pipeline_tensor_from_stage_broadcast():
    received = {"__msg_type__": "proxy", "hidden_states": object()}
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(pp_stage_tp_sizes="4,2"),
        ps=SimpleNamespace(attn_tp_rank=1),
        _pp_tensor_dict_inbox=defaultdict(deque),
        _pp_topology=lambda: None,
        pp_group=Mock(),
        attn_tp_group=Mock(),
    )
    scheduler.attn_tp_group.broadcast_tensor_dict.return_value = received

    result = SchedulerPPMixin._pp_recv_typed_dict(scheduler, expected_kind="proxy")

    assert result is received
    scheduler.pp_group.recv_tensor_dict.assert_not_called()
    scheduler.attn_tp_group.broadcast_tensor_dict.assert_called_once_with(None, src=0)


def test_pipeline_pyobj_uses_stage_leader_ranks():
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(pp_stage_tp_sizes="4,2"),
        ps=SimpleNamespace(pp_rank=1, attn_tp_rank=0, attn_cp_rank=0),
        world_group=SimpleNamespace(cpu_group=object()),
    )

    with patch(
        "sglang.srt.managers.scheduler_pp_mixin.point_to_point_pyobj",
        return_value=[],
    ) as point_to_point:
        SchedulerPPMixin._pp_send_pyobj_to_next_stage(scheduler, ["request"])

    point_to_point.assert_called_once_with(
        ["request"],
        4,
        scheduler.world_group.cpu_group,
        4,
        0,
        async_send=False,
    )
