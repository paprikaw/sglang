"""Keep the router alive while the controller finishes its scale-down."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time


async def hold_control_after_workload(run: Path, exit_code: int, timeout_s=480):
    done = run / 'replay_workload_done.json'
    temporary = done.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(exit_code=exit_code, unix=time.time()))+'\n')
    temporary.replace(done)
    if exit_code:
        return
    deadline = time.monotonic()+timeout_s
    while not (run/'release_router').exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('controller did not release the post-workload router')
        await asyncio.sleep(.1)
