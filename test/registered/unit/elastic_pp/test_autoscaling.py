from __future__ import annotations

import pytest

from sglang.srt.elastic_pp.autoscaling import (
    ElasticPPBacklogPolicy,
    parse_elastic_pp_partitions,
    serialize_elastic_pp_partitions,
    validate_elastic_pp_autoscale_partitions,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _frontier():
    return ((4, 0, 0, 4), (2, 2, 2, 2))


def test_parse_validate_and_serialize_autoscale_frontier():
    partitions = parse_elastic_pp_partitions("4,0,0,4; 2,2,2,2")

    assert partitions == _frontier()
    assert (
        validate_elastic_pp_autoscale_partitions(
            partitions,
            pp_size=4,
            num_layers=8,
        )
        == _frontier()
    )
    assert serialize_elastic_pp_partitions(partitions) == "4,0,0,4;2,2,2,2"


@pytest.mark.parametrize(
    "raw,match",
    [
        ("4,0,0,4", "at least two"),
        ("4,0,0,4;", "empty entries"),
        ("4,x,0,4;2,2,2,2", "integers"),
    ],
)
def test_parse_autoscale_frontier_rejects_invalid_text(raw, match):
    with pytest.raises(ValueError, match=match):
        parse_elastic_pp_partitions(raw)


def test_validate_autoscale_frontier_allows_moving_last_and_requires_rank_zero():
    assert validate_elastic_pp_autoscale_partitions(
        ((8, 0), (4, 4)),
        pp_size=2,
        num_layers=8,
    ) == ((8, 0), (4, 4))
    with pytest.raises(ValueError, match="rank 0"):
        validate_elastic_pp_autoscale_partitions(
            ((0, 8, 0, 0), (2, 2, 2, 2)),
            pp_size=4,
            num_layers=8,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_elastic_pp_autoscale_partitions(
            ((4, 0, 0, 4), (3, 0, 0, 5)),
            pp_size=4,
            num_layers=8,
        )


def test_backlog_policy_uses_stable_hysteresis_and_post_commit_residence():
    policy = ElasticPPBacklogPolicy(
        partitions=_frontier(),
        scale_up_total_requests=8,
        scale_down_total_requests=1,
        scale_up_stable_checks=2,
        scale_down_stable_checks=2,
        min_operation_interval_s=10,
    )

    assert (
        policy.observe(
            current_partition=_frontier()[0],
            waiting_requests=5,
            running_requests=3,
            now_s=1,
        )
        is None
    )
    assert (
        policy.observe(
            current_partition=_frontier()[0],
            waiting_requests=6,
            running_requests=3,
            now_s=2,
        )
        == _frontier()[1]
    )

    # The cooldown starts at commit, not at the earlier decision.
    policy.mark_operation_completed(20)
    assert (
        policy.observe(
            current_partition=_frontier()[1],
            waiting_requests=0,
            running_requests=1,
            now_s=21,
        )
        is None
    )
    assert (
        policy.observe(
            current_partition=_frontier()[1],
            waiting_requests=0,
            running_requests=0,
            now_s=29.9,
        )
        is None
    )
    assert (
        policy.observe(
            current_partition=_frontier()[1],
            waiting_requests=0,
            running_requests=0,
            now_s=30,
        )
        == _frontier()[0]
    )


def test_backlog_policy_failure_backoff_and_operation_limit():
    policy = ElasticPPBacklogPolicy(
        partitions=_frontier(),
        scale_up_total_requests=2,
        scale_down_total_requests=0,
        scale_up_stable_checks=1,
        failure_backoff_s=5,
        max_operations=1,
    )
    assert (
        policy.observe(
            current_partition=_frontier()[0],
            waiting_requests=2,
            running_requests=0,
            now_s=1,
        )
        == _frontier()[1]
    )
    policy.mark_operation_failed(1)
    assert (
        policy.observe(
            current_partition=_frontier()[0],
            waiting_requests=2,
            running_requests=0,
            now_s=5.9,
        )
        is None
    )
    assert (
        policy.observe(
            current_partition=_frontier()[0],
            waiting_requests=2,
            running_requests=0,
            now_s=6,
        )
        == _frontier()[1]
    )
    policy.mark_operation_completed(7)
    assert policy.completed_operations == 1
    assert (
        policy.observe(
            current_partition=_frontier()[1],
            waiting_requests=0,
            running_requests=0,
            now_s=100,
        )
        is None
    )
