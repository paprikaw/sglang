"""Correlate completed P/D transfers and actual graph replays in the startup probe."""

import json
import time


def install():
    from sglang.srt.disaggregation.decode import DecodeTransferQueue
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )
    from sglang.srt.distributed import get_pp_group

    pop = DecodeTransferQueue.pop_transferred
    execute = DecodeCudaGraphRunner.execute
    pending = []

    def transferred(queue, *args, **kwargs):
        reqs = pop(queue, *args, **kwargs)
        for req in reqs:
            room = req.bootstrap_room
            pending.append(room)
            print(
                "PP2_STARTUP_HANDOFF",
                json.dumps(
                    dict(room=room, rank=get_pp_group().rank_in_group, unix=time.time())
                ),
                flush=True,
            )
        return reqs

    def replay(runner, *args, **kwargs):
        result = execute(runner, *args, **kwargs)
        while pending:
            print(
                "PP2_STARTUP_GRAPH",
                json.dumps(
                    dict(
                        room=pending.pop(0),
                        rank=get_pp_group().rank_in_group,
                        unix=time.time(),
                    )
                ),
                flush=True,
            )
        return result

    DecodeTransferQueue.pop_transferred = transferred
    DecodeCudaGraphRunner.execute = replay
