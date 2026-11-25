"""Progress CPU PP sends without blocking the scheduler ahead of its ring."""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

_wait_executor = None
_executor_lock = Lock()


def _executor():
    # One waiter per worker process, shared by successive cache generations.
    # It only waits on already-issued CPU sends: no CUDA or new communication.
    global _wait_executor
    with _executor_lock:
        if _wait_executor is None:
            _wait_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="hicache-pp-send"
            )
        return _wait_executor


def _wait(work, keepalive):
    result = work.wait()
    if result is False:
        raise RuntimeError("HiCache PP control send did not complete")
    # An exception retains the frame and backing until its owner handles it.
    del keepalive
    return result


class BackgroundSendWork:
    """Work-compatible completion backed by an actual native wait.

    Gloo SendWork can report is_completed=False after the peer has received
    its tensor, until wait() is called. Polling that flag retains all historic
    sends. Waiting on a CPU helper lets the scheduler send the consensus
    messages the receiver needs, then reap completed work in later slots.
    The final blocking drain still propagates every send failure.
    """

    def __init__(self, work, keepalive=None):
        self._future = _executor().submit(_wait, work, keepalive)

    @classmethod
    def wrap(cls, work, keepalive=None):
        return work if isinstance(work, cls) else cls(work, keepalive)

    def is_completed(self):
        return self._future.done()

    def wait(self):
        return self._future.result()
