"""One opt-in native CPU stack snapshot during the incoming weight restore."""

import ctypes
import functools
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def capture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    evidence = dict(pid=os.getpid(), start=time.monotonic(), wall=time.time())
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    ptracer = 0x59616D61  # PR_SET_PTRACER, linux/prctl.h
    wrapper = "import os,sys;sys.stdin.buffer.read(1);os.execv(sys.argv[1],sys.argv[1:])"
    command = [sys.executable, "-c", wrapper,
               "/data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/py-spy",
               "dump", "--native", "--pid", str(os.getpid())]
    with (directory / "rank1.g1.native.txt").open("w") as output:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output,
                                   stderr=subprocess.STDOUT)
        try:
            # Allow this exact debugger PID and its descendants, never all PIDs.
            if libc.prctl(ptracer, process.pid, 0, 0, 0):
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            process.communicate(input=b"1", timeout=5)
            evidence["returncode"] = process.returncode
        except BaseException as error:
            evidence["error"] = f"{type(error).__name__}: {error}"
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
            libc.prctl(ptracer, 0, 0, 0, 0)
    evidence["end"] = time.monotonic()
    (directory / "rank1.g1.native.json").write_text(json.dumps(evidence) + "\n")


def install(coordinator_class, directory):
    original = coordinator_class._prepare_worker

    @functools.wraps(original)
    def traced(self):
        if (self.cache_transition is None or self.rank != 1
                or self._transition.target.generation != 1):
            return original(self)
        done = threading.Event()

        def sample():
            if not done.wait(0.3):
                capture(directory)

        sampler = threading.Thread(target=sample, name="native-stack-once", daemon=True)
        sampler.start()
        try:
            return original(self)
        finally:
            done.set()
            sampler.join(timeout=7)

    coordinator_class._prepare_worker = traced
