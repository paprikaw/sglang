from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from sglang.srt.elastic_pp.autoscaling import ElasticPPBacklogPolicy
from sglang.srt.managers.io_struct import (
    GetElasticPPStateReqOutput,
    ReconfigureElasticPPReqInput,
    ReconfigureElasticPPReqOutput,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _output(
    *,
    success=True,
    message="ok",
    generation=1,
    counts=(2, 2),
    prepare_seconds=1.0,
):
    return ReconfigureElasticPPReqOutput(
        success=success,
        message=message,
        old_generation=0,
        new_generation=generation,
        old_layer_counts=[3, 1],
        new_layer_counts=list(counts),
        migration_phase="committed" if success else "failed",
        source_usable=True,
        prepare_seconds=prepare_seconds,
        fence_seconds=0.1,
        total_seconds=1.1,
    )


class _Tokenizer(TokenizerControlMixin):
    def __init__(self, results):
        self.results = results
        self.loop_started = False

    def auto_create_handle_loop(self):
        self.loop_started = True

    async def reconfigure_elastic_pp_communicator(self, obj):
        self.last_request = obj
        return self.results


@pytest.mark.asyncio
async def test_control_plane_returns_consistent_replica_commit():
    tokenizer = _Tokenizer([_output(), _output(prepare_seconds=1.2)])
    request = ReconfigureElasticPPReqInput(layer_counts=[2, 2], generation=1)

    result = await tokenizer.reconfigure_elastic_pp(request)

    assert tokenizer.loop_started
    assert tokenizer.last_request is request
    assert result.success
    assert result.new_generation == 1


@pytest.mark.asyncio
async def test_control_plane_rejects_partial_or_divergent_replica_result():
    tokenizer = _Tokenizer(
        [
            _output(prepare_seconds=1.0),
            _output(
                success=False,
                message="rank allocation failed",
                counts=(1, 3),
                prepare_seconds=1.4,
            ),
        ]
    )

    result = await tokenizer.reconfigure_elastic_pp(
        ReconfigureElasticPPReqInput(layer_counts=[2, 2], generation=1)
    )

    assert not result.success
    assert "rank allocation failed" in result.message
    assert result.prepare_seconds == 1.4


class _AutoscaleTokenizer(TokenizerControlMixin):
    def __init__(self):
        self.loop_started = False
        self._elastic_pp_reconfigure_lock = asyncio.Lock()
        self._elastic_pp_autoscale_policy = ElasticPPBacklogPolicy(
            partitions=((4, 0, 0, 4), (2, 2, 2, 2)),
            scale_up_total_requests=8,
            scale_down_total_requests=0,
            scale_up_stable_checks=1,
            scale_down_stable_checks=1,
            min_operation_interval_s=0,
        )
        self.state = GetElasticPPStateReqOutput(
            enabled=True,
            generation=0,
            layer_counts=[4, 0, 0, 4],
            migration_phase="idle",
            source_usable=True,
            waiting_requests=6,
            running_requests=2,
        )
        self.requests = []
        self.communicator_delay_s = 0.0
        self.communicator_entered = asyncio.Event()
        self.active_communicator_calls = 0
        self.max_active_communicator_calls = 0
        self.server_args = SimpleNamespace(
            elastic_pp_autoscale_poll_interval_s=0.1,
        )

    def auto_create_handle_loop(self):
        self.loop_started = True

    async def get_elastic_pp_state_communicator(self, obj):
        return [self.state]

    async def reconfigure_elastic_pp_communicator(self, obj):
        self.requests.append(obj)
        self.active_communicator_calls += 1
        self.max_active_communicator_calls = max(
            self.max_active_communicator_calls,
            self.active_communicator_calls,
        )
        self.communicator_entered.set()
        try:
            if self.communicator_delay_s:
                await asyncio.sleep(self.communicator_delay_s)
            old_counts = list(self.state.layer_counts)
            self.state = GetElasticPPStateReqOutput(
                enabled=True,
                generation=self.state.generation + 1,
                layer_counts=list(obj.layer_counts),
                migration_phase="done",
                source_usable=True,
                waiting_requests=self.state.waiting_requests,
                running_requests=self.state.running_requests,
            )
            return [
                ReconfigureElasticPPReqOutput(
                    success=True,
                    message="ok",
                    old_generation=self.state.generation - 1,
                    new_generation=self.state.generation,
                    old_layer_counts=old_counts,
                    new_layer_counts=list(obj.layer_counts),
                    migration_phase="committed",
                    source_usable=True,
                    prepare_seconds=0.2,
                    fence_seconds=0.03,
                    total_seconds=0.25,
                )
            ]
        finally:
            self.active_communicator_calls -= 1


@pytest.mark.asyncio
async def test_autoscale_tick_uses_scheduler_backlog_and_transactional_path():
    tokenizer = _AutoscaleTokenizer()

    result = await tokenizer._elastic_pp_autoscale_tick()

    assert result.success
    assert tokenizer.requests[0].layer_counts == [2, 2, 2, 2]
    assert tokenizer.requests[0].generation is None
    assert tokenizer._elastic_pp_current_layer_counts == (2, 2, 2, 2)
    assert tokenizer._elastic_pp_autoscale_policy.completed_operations == 1


@pytest.mark.asyncio
async def test_autoscale_tick_waits_for_stable_migration_phase():
    tokenizer = _AutoscaleTokenizer()
    tokenizer.state.migration_phase = "cleaning"

    assert await tokenizer._elastic_pp_autoscale_tick() is None
    assert tokenizer.requests == []


@pytest.mark.asyncio
async def test_autoscale_tick_can_retry_a_source_usable_failed_prepare():
    tokenizer = _AutoscaleTokenizer()
    tokenizer.state.migration_phase = "failed"

    result = await tokenizer._elastic_pp_autoscale_tick()

    assert result.success
    assert tokenizer.requests[0].layer_counts == [2, 2, 2, 2]


@pytest.mark.asyncio
async def test_manual_transition_and_autoscale_observation_are_serialized():
    tokenizer = _AutoscaleTokenizer()
    tokenizer.communicator_delay_s = 0.02
    manual = asyncio.create_task(
        tokenizer.reconfigure_elastic_pp(
            ReconfigureElasticPPReqInput(layer_counts=[2, 2, 2, 2])
        )
    )
    await tokenizer.communicator_entered.wait()
    automatic = asyncio.create_task(tokenizer._elastic_pp_autoscale_tick())

    manual_result, automatic_result = await asyncio.gather(manual, automatic)

    assert manual_result.success
    assert automatic_result is None
    assert tokenizer.max_active_communicator_calls == 1
    assert len(tokenizer.requests) == 1
