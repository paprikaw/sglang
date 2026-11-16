"""A real send wait must progress independently of the scheduler's PP ring."""

import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.mem_cache import pp_control_work
from sglang.srt.mem_cache.pp_control_work import BackgroundSendWork
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def waiter(monkeypatch):
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pp_control_work, "_executor", lambda: executor)
        yield


class Send:
    def __init__(self, *, error=None, result=True):
        self.release = threading.Event()
        self.started = threading.Event()
        self.finished = threading.Event()
        self.calls = 0
        self.error, self.result = error, result

    def is_completed(self):
        return self.finished.is_set()

    def wait(self):
        self.calls += 1
        self.started.set()
        if not self.release.wait(5):
            raise TimeoutError("test receiver never consumed its consensus")
        if self.error:
            raise self.error
        self.finished.set()
        return self.result


def drain(cache, *, blocking):
    UnifiedRadixCache._drain_async_work(cache, blocking=blocking)


def test_nonblocking_poll_does_not_wait_ahead_of_receiver_consensus():
    native = Send()
    cache = NS(work_list=[native])
    try:
        drain(cache, blocking=False)
        assert native.started.wait(2)
        assert len(cache.work_list) == 1
        assert isinstance(cache.work_list[0], BackgroundSendWork)
        assert not native.finished.is_set()
        # The scheduler can now issue the consensus that releases its peer.
        native.release.set()
        drain(cache, blocking=True)
        assert cache.work_list == []
        assert native.calls == 1
    finally:
        native.release.set()


def test_received_gloo_style_work_is_reaped_by_later_nonblocking_poll():
    native = Send()
    native.release.set()  # Peer has received it; the native flag is still false.
    assert not native.is_completed()
    cache = NS(work_list=[native])
    drain(cache, blocking=False)
    if cache.work_list:
        cache.work_list[0].wait()
    drain(cache, blocking=False)
    assert cache.work_list == []
    assert native.calls == 1


def test_wrapping_twice_does_not_start_a_second_native_wait():
    native = Send()
    native.release.set()
    work = BackgroundSendWork(native)
    assert BackgroundSendWork.wrap(work) is work
    assert work.wait() is True
    assert native.calls == 1


def test_payload_stays_alive_until_native_wait_finishes():
    class Payload:
        pass

    native, payload = Send(), Payload()
    alive = weakref.ref(payload)
    work = BackgroundSendWork(native, keepalive=payload)
    del payload
    try:
        assert native.started.wait(2)
        gc.collect()
        assert alive() is not None
        native.release.set()
        work.wait()
        gc.collect()
        assert alive() is None
    finally:
        native.release.set()


@pytest.mark.parametrize("blocking", [False, True])
def test_failed_send_is_never_silently_removed(blocking):
    native = Send(error=RuntimeError("injected send failure"))
    native.release.set()
    work = BackgroundSendWork(native)
    with pytest.raises(RuntimeError, match="injected send failure"):
        work.wait()
    assert work.is_completed()
    cache = NS(work_list=[work])
    with pytest.raises(RuntimeError, match="injected send failure"):
        drain(cache, blocking=blocking)
    assert cache.work_list == [work]


def test_false_native_completion_is_rejected():
    native = Send(result=False)
    native.release.set()
    with pytest.raises(RuntimeError, match="control send did not complete"):
        BackgroundSendWork(native).wait()


def test_synchronous_poll_uses_its_existing_native_wait():
    native = Send()
    native.release.set()
    cache = NS(work_list=[native])
    drain(cache, blocking=True)
    assert cache.work_list == []
    assert not getattr(cache, "_pp_background_send_waits", False)
    assert native.calls == 1


def test_pp_sync_keeps_new_sends_out_of_the_scheduler_wait(monkeypatch):
    native = Send()
    monkeypatch.setattr(torch.distributed, "isend", lambda *a, **k: native)
    cache = NS(work_list=[], pp_rank=0, pp_size=2, pp_group=object())
    drain(cache, blocking=False)
    try:
        UnifiedRadixCache._pp_sync(cache, torch.tensor([0, 0], dtype=torch.int32))
        assert native.started.wait(2)
        assert len(cache.work_list) == 1
        assert isinstance(cache.work_list[0], BackgroundSendWork)
        assert not native.finished.is_set()
        native.release.set()
        drain(cache, blocking=True)
        assert cache.work_list == []
    finally:
        native.release.set()
