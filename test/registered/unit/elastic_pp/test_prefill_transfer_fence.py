import pytest
from sglang.srt.elastic_pp.prefill_transfer import PrefillTransferFence


def test_queued_and_deferred_chunks_pin_the_source():
    fence = PrefillTransferFence()
    first, second = object(), object()
    fence.admit(first)
    fence.admit(second)
    fence.begin_io(23)
    assert fence.sending(23) and not fence.sending(24)
    with pytest.raises(RuntimeError):
        fence.close()
    fence.end_io(23)
    fence.finish(first)
    assert fence.pending() == 1
    with pytest.raises(RuntimeError):
        fence.close()
    fence.finish(second)
    fence.close()
    with pytest.raises(RuntimeError):
        fence.admit(first)
    fence.reopen()
    fence.admit(first)
    fence.finish(first)
