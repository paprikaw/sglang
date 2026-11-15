"""Exercise the native P/D admission body without loading a GPU backend."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def native_admission(cap):
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/disaggregation/decode.py"
    )
    module = ast.parse(path.read_text())
    cls = next(
        x
        for x in module.body
        if isinstance(x, ast.ClassDef)
        and x.name == "SchedulerDisaggregationDecodeMixin"
    )
    method = next(
        x
        for x in cls.body
        if isinstance(x, ast.FunctionDef) and x.name == "get_new_prebuilt_batch"
    )

    def init_new(reqs, *_args):
        return NS(
            reqs=reqs,
            prepare_for_prebuilt=lambda: None,
            process_prebuilt=lambda _future_map: None,
        )

    namespace = dict(
        get_parallel=lambda: NS(pp_max_micro_batch_size=cap),
        get_disagg=lambda: NS(disaggregation_decode_enable_radix_cache=False),
        ScheduleBatch=NS(init_new=init_new),
        set_time_batch=lambda *_args: None,
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, method], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[method.name]


def scheduler(pp):
    initialized = []
    requests = [
        NS(
            rid=i,
            kv_committed_len=None,
            init_next_round_input=lambda _cache, i=i: initialized.append(i),
        )
        for i in range(4)
    ]
    return (
        NS(
            ps=NS(pp_size=pp),
            waiting_queue=requests,
            grammar_manager=NS(has_waiting_grammars=lambda: False),
            enable_priority_scheduling=False,
            req_to_token_pool=NS(size=16),
            max_running_requests=4,
            tree_cache=object(),
            token_to_kv_pool_allocator=None,
            model_config=None,
            enable_overlap=False,
            spec_algorithm=None,
            future_map=None,
        ),
        initialized,
    )


@pytest.mark.parametrize(
    "pp,cap,running,expected",
    [
        (3, 1, 0, 1),
        (3, 1, 1, 0),
        (3, 2, 1, 1),
        (3, None, 1, 3),
        (1, 1, 1, 3),
    ],
)
def test_prebuilt_admission_honors_pp_slot_capacity(pp, cap, running, expected):
    sched, initialized = scheduler(pp)
    original = list(sched.waiting_queue)
    batch = native_admission(cap)(sched, NS(batch_size=lambda: running))
    assert (len(batch.reqs) if batch else 0) == expected
    assert sched.waiting_queue == original[expected:]
    assert initialized == list(range(expected))


def test_full_slot_preserves_request_for_an_empty_pipeline_slot():
    sched, initialized = scheduler(3)
    admit = native_admission(1)
    assert admit(sched, NS(batch_size=lambda: 1)) is None
    assert len(sched.waiting_queue) == 4 and not initialized
    batch = admit(sched, NS(batch_size=lambda: 0))
    assert [r.rid for r in batch.reqs] == [0]
    assert [r.rid for r in sched.waiting_queue] == [1, 2, 3]
    assert initialized == [0]
