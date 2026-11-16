"""Process-scoped finite AgentX replay; not an official steady-state score.

Only the scenario's duration default/floor changes. DAG, warmup, tool delays,
loader, cache-bust and native request accounting locks remain enabled.
"""
from __future__ import annotations

import importlib.metadata


def install(expected_requests: int):
    if importlib.metadata.version('agentx-harness') != '1.0.7':
        raise RuntimeError('Finite replay adapter requires agentx-harness==1.0.7')
    if expected_requests <= 0:
        raise ValueError('A positive finite request count is required')
    from aiperf.config.flags import converter

    original = converter._apply_scenario_defaults
    if getattr(original, '_finite_profile_count', None) == expected_requests:
        return
    if hasattr(original, '_finite_profile_count'):
        raise RuntimeError('Finite profile request count changed in one process')
    from aiperf.common.scenario.registry import SCENARIOS

    native = SCENARIOS['agentx']
    if (native.default_benchmark_duration_seconds,
            native.min_benchmark_duration_seconds) != (3600, 900):
        raise RuntimeError('Unknown AgentX duration policy')
    SCENARIOS['agentx'] = native.model_copy(update={
        'default_benchmark_duration_seconds': None,
        'min_benchmark_duration_seconds': 0,
    })

    def finite_defaults(cli):
        if cli.scenario != 'agentx' or cli.request_count != expected_requests:
            raise ValueError('Finite replay requires its frozen AgentX request count')
        if cli.benchmark_duration is not None or 'benchmark_grace_period' in cli.model_fields_set:
            raise ValueError('Finite replay must use request count without a duration/grace phase')
        # Mark duration=None explicitly before scenario defaults are applied.
        # The official converter then builds requests=N, duration=None. Native
        # request accounting and metrics stay enabled; duration coverage has no
        # configured time window to validate. No validator is patched.
        resolved = original(cli.model_copy(update={'benchmark_duration': None}))
        # The native phase schema accepts an omitted duration (default=None),
        # but its gt=0 constraint rejects an explicitly supplied None. Keep the
        # duration absent from the builder's explicit flag routes as well.
        resolved.model_fields_set.discard('benchmark_duration')
        return resolved

    finite_defaults._finite_profile_count = expected_requests
    converter._apply_scenario_defaults = finite_defaults
