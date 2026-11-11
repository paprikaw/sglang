"""Source inference and destination construction can share a worker process."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_attn_backend,
    get_forward_context,
    get_req_to_token_pool,
    get_token_to_kv_pool,
    has_forward_context,
    set_forward_context,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_nested_backend_override_restores_after_error():
    source, other = ForwardContext(object()), ForwardContext(object())
    assert not has_forward_context()
    with forward_context(source):
        with pytest.raises(RuntimeError, match="failed override"):
            with forward_context(other):
                assert get_attn_backend() is other.attn_backend
                raise RuntimeError("failed override")
        assert get_forward_context() is source
    assert not has_forward_context()
    with pytest.raises(AssertionError, match="no forward context active"):
        get_forward_context()


def test_explicit_context_save_restore_resolves_active_pools():
    backend = SimpleNamespace(token_to_kv_pool=object(), req_to_token_pool=object())
    context = ForwardContext(backend)
    previous = set_forward_context(context)
    try:
        assert get_token_to_kv_pool() is backend.token_to_kv_pool
        assert get_req_to_token_pool() is backend.req_to_token_pool
    finally:
        assert set_forward_context(previous) is context
    assert not has_forward_context()


def test_background_construction_does_not_inherit_or_replace_source_context():
    source, worker = ForwardContext(object()), ForwardContext(object())
    entered, release = Event(), Event()

    def construct():
        assert not has_forward_context()
        with forward_context(worker):
            entered.set()
            assert release.wait(5), "source did not release background builder"
            assert get_forward_context() is worker
        assert not has_forward_context()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with forward_context(source):
            pending = executor.submit(construct)
            try:
                assert entered.wait(5), "background builder failed to enter"
                assert get_forward_context() is source
            finally:
                release.set()
            pending.result(timeout=5)
            assert get_forward_context() is source
    assert not has_forward_context()


def test_compiled_backend_reads_remain_guarded_when_context_changes():
    import torch

    def forward(x):
        return x + get_attn_backend().offset

    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    try:
        for offset in (1, 2):
            with forward_context(ForwardContext(SimpleNamespace(offset=offset))):
                torch.testing.assert_close(
                    compiled(torch.ones(1)), torch.tensor([1.0 + offset])
                )
    finally:
        torch._dynamo.reset()
