from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


def test_decode_progress_keeps_pending_sends_without_waiting_on_downstream():
    pending = Mock()
    pending.is_completed.return_value = False
    pending.wait.side_effect = AssertionError('Would deadlock before sending consensus')
    completed = Mock()
    completed.is_completed.return_value = True
    cache = SimpleNamespace(work_list=[pending, completed])
    UnifiedRadixCache._drain_async_work(cache, blocking=False)
    pending.wait.assert_not_called()
    completed.wait.assert_called_once()
    assert cache.work_list == [pending]
    # Once the downstream stage consumes the packet, retain no stale handle.
    pending.is_completed.return_value = True
    pending.wait.side_effect = None
    UnifiedRadixCache._drain_async_work(cache, blocking=False)
    assert cache.work_list == []
    pending.wait.assert_called_once()


def test_default_cache_progress_preserves_blocking_semantics():
    work = Mock()
    work.is_completed.return_value = False
    cache = SimpleNamespace(work_list=[work])
    UnifiedRadixCache._drain_async_work(cache)
    work.wait.assert_called_once()
    assert cache.work_list == []
