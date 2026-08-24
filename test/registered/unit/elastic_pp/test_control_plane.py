from __future__ import annotations

import pytest

from sglang.srt.managers.io_struct import (
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
