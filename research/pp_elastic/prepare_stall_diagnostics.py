"""Opt-in CPU stack/clock sampling for one owned PP preparation worker.

No CUDA calls, payloads, or changes to transition ordering. Files belong to the
run directory; this hook is never enabled by a normal qualification launch.
"""

import faulthandler
import functools
import json
import os
from pathlib import Path
import threading
import time


def install(coordinator_class, directory):
    original = coordinator_class._prepare_worker
    directory = Path(directory)

    @functools.wraps(original)
    def traced(self):
        # The private startup Graph walk has no serving cache adapter.
        if self.cache_transition is None:
            return original(self)
        generation = self._transition.target.generation
        directory.mkdir(parents=True, exist_ok=True)
        stem = directory / f"rank{self.rank}.g{generation}"
        stop = threading.Event()
        ticks = []

        def pulse():
            while not stop.is_set():
                ticks.append(time.monotonic())
                stop.wait(0.02)

        def clock():
            return dict(wall=time.time(), monotonic=time.monotonic())

        def threads():
            return [
                dict(name=t.name, ident=hex(t.ident), native_id=t.native_id)
                for t in threading.enumerate()
                if t.ident is not None
            ]

        evidence = dict(pid=os.getpid(), rank=self.rank, generation=generation)
        evidence.update(start=clock(), threads_start=threads())
        sampler = threading.Thread(target=pulse, name="prepare-pulse", daemon=True)
        sampler.start()
        try:
            with stem.with_suffix(stem.suffix + ".stacks.txt").open("w") as output:
                output.write(json.dumps(evidence) + "\n")
                output.flush()
                faulthandler.dump_traceback_later(0.2, repeat=True, file=output)
                try:
                    return original(self)
                finally:
                    faulthandler.cancel_dump_traceback_later()
        finally:
            evidence.update(end=clock(), threads_end=threads())
            stop.set()
            sampler.join(timeout=2)
            evidence["heartbeat_monotonic"] = ticks
            observed = [
                evidence["start"]["monotonic"],
                *ticks,
                evidence["end"]["monotonic"],
            ]
            evidence["largest_heartbeat_gap_seconds"] = max(
                (b - a for a, b in zip(observed, observed[1:])), default=0
            )
            target = stem.with_suffix(stem.suffix + ".clock.json")
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(evidence, indent=2) + "\n")
            temporary.replace(target)

    coordinator_class._prepare_worker = traced
