"""Select an existing eager path only for explicitly labelled diagnostics.

No graph nodes or model operations are changed. Zero selects the original graph
eligibility check; one uses the engine's normal eager fallback. Performance is
always measured at zero. A shared private mmap avoids filesystem polling.
"""
from functools import wraps
import mmap
import os
import stat

_INSTALLED = False
_POLICY = None


def install():
    global _INSTALLED, _POLICY
    if _INSTALLED:
        return
    from sglang.srt.model_executor.model_runner import ModelRunner
    fd = os.open(os.environ['STEADY_EAGER_POLICY_PATH'], os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_size != 1:
            raise RuntimeError('Diagnostic policy must be a private one-byte file')
        _POLICY = mmap.mmap(fd, 1, access=mmap.ACCESS_READ)
    finally:
        os.close(fd)
    original = ModelRunner.init_decode_cuda_graph

    @wraps(original)
    def initialize(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        graph = self.decode_cuda_graph_runner
        if graph is not None:
            eligibility = graph.can_run_graph

            def selected(batch):
                mode = _POLICY[0]
                if mode not in (ord('0'), ord('1')):
                    raise RuntimeError('Invalid eager diagnostic policy')
                return False if mode == ord('1') else eligibility(batch)

            graph.can_run_graph = selected
        return result

    ModelRunner.init_decode_cuda_graph = initialize
    _INSTALLED = True
