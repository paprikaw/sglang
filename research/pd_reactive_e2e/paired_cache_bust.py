"""Freeze only cache marker identity for fresh-server paired AgentX replays.

Benchmark IDs remain unique for accounting. Warmup/profile and tree isolation
still use the native marker ledger; no shared installed package is modified.
"""
from __future__ import annotations

import importlib.metadata


def install(namespace: str):
    if importlib.metadata.version('agentx-harness') != '1.0.7':
        raise RuntimeError('Paired cache markers require agentx-harness==1.0.7')
    if len(namespace) != 64 or any(c not in '0123456789abcdef' for c in namespace):
        raise ValueError('A frozen SHA256 workload namespace is required')
    from aiperf.timing.strategies import cache_bust

    original = cache_bust.build_cache_bust_marker
    installed = getattr(original, '_paired_cache_namespace', None)
    if installed is not None:
        if installed != namespace:
            raise RuntimeError('Paired cache namespace changed in one process')
        return

    def paired_marker(benchmark_id, recycle_pass, trajectory_index, trace_id, *, target):
        return original(namespace, recycle_pass, trajectory_index, trace_id, target=target)

    paired_marker._paired_cache_namespace = namespace
    cache_bust.build_cache_bust_marker = paired_marker
