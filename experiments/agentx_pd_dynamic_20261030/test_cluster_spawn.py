"""Regression: spawn entrypoint must install rank runtime before model creation."""
import os
from pathlib import Path
import runpy
import sys
import types
from unittest.mock import patch
ROOT = Path(__file__).resolve().parent


def test_spawned_rank_initializes_runtime():
    calls = []
    adapter = types.ModuleType('runtime_adapter')
    adapter.install = lambda: calls.append('installed')
    with patch.dict(os.environ, {'EP2_PP2_ARM': 'ep2', 'EP2_PP2_RUN': '/unused'}), patch.dict(sys.modules, {'runtime_adapter': adapter}):
        runpy.run_path(str(ROOT/'cluster_runner_v4.py'), run_name='__mp_main__')
    assert calls == ['installed']


def test_cpu_coordinator_does_not_initialize_rank_runtime():
    calls = []
    adapter = types.ModuleType('runtime_adapter')
    adapter.install = lambda: calls.append('installed')
    clean = {k: v for k, v in os.environ.items() if k not in ('EP2_PP2_ARM', 'EP2_PP2_RUN')}
    with patch.dict(os.environ, clean, clear=True), patch.dict(sys.modules, {'runtime_adapter': adapter}):
        runpy.run_path(str(ROOT/'cluster_runner_v4.py'), run_name='__mp_main__')
    assert calls == []
