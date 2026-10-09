"""CPU regressions for opt-in PP slot ownership; no torch/CUDA import required.

The merge/abort/pause tests execute the real scheduler methods, AST-extracted
without their GPU-heavy module imports. Only model admission/forward is faked.
"""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[4]
MANAGERS = ROOT / "python/sglang/srt/managers"
_spec = importlib.util.spec_from_file_location(
    "_pp_slot_affinity_cpu", MANAGERS / "pp_slot_affinity.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
PPSlotAffinityState = _module.PPSlotAffinityState


def _scheduler_method(name, **globals_):
    tree = ast.parse((MANAGERS / "scheduler.py").read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    method.decorator_list = []
    code = "from __future__ import annotations\n" + ast.unparse(method)
    namespace = dict(globals_)
    exec(compile(code, str(MANAGERS / "scheduler.py"), "exec"), namespace)
    return namespace[name]


class _Req:
    def __init__(self, rid):
        self.rid = rid
        self.remaining_chunks = 32
        self.prefix_indices = []
        self.extend_range = SimpleNamespace(end=0)
        self.req_pool_idx = int(rid)
        self.kv_identity = object()
        self.inflight_middle_chunks = 0
        self.to_finish = None
        self.done = False
        self.time_stats = SimpleNamespace(trace_ctx=SimpleNamespace(abort=MagicMock()))

    def finished(self):
        return self.done


class _Batch:
    def __init__(self, reqs=(), *, extend=False, chunk=None):
        self.reqs = list(reqs)
        self.chunked_req = chunk
        self.forward_mode = SimpleNamespace(is_extend=lambda: extend)
        self.is_prefill_only = False
        self.batch_is_full = False

    def is_empty(self):
        return not self.reqs

    def batch_size(self):
        return len(self.reqs)

    def filter_batch(self, chunked_req_to_exclude=()):
        self.reqs = [
            req
            for req in self.reqs
            if req not in chunked_req_to_exclude and not req.finished()
        ]

    def merge_batch(self, other):
        self.reqs += other.reqs


class TestPPSlotAffinity(unittest.TestCase):
    def test_real_merge_keeps_sixteen_long_requests_in_two_b8_slots(self):
        state = PPSlotAffinityState(2, 8)
        requests = [_Req(str(i)) for i in range(16)]
        kv_ids = [req.kv_identity for req in requests]
        scheduler = SimpleNamespace(
            _pp_slot_affinity=state,
            waiting_queue=list(requests),
            enable_fpm=False,
            dllm_config=None,
            enable_hisparse=False,
            require_mlp_sync=False,
            _pending_chunked_abort_req=None,
            _abort_on_waiting_timeout=lambda: None,
            _abort_on_running_timeout=lambda batch: None,
            process_pending_chunked_abort=lambda: None,
            _should_defer_prefill=lambda: False,
            _arm_prefill_decode_interval=lambda batch: None,
            dp_attn_adapter=SimpleNamespace(
                maybe_prepare_mlp_sync_batch=lambda batch, **kwargs: batch
            ),
            ngram_embedding_manager=SimpleNamespace(
                prepare_for_forward=lambda batch, **kwargs: batch
            ),
        )
        scheduler.stash_chunked_request = lambda req: req.prefix_indices.append(1)

        def update_running(batch):
            batch.forward_mode = SimpleNamespace(is_extend=lambda: False)
            return batch

        scheduler.update_running_batch = update_running
        running = [_Batch(), _Batch()]
        last = [None, None]
        seen = {id(req): [] for req in requests}

        def prefill(batch):
            slot = scheduler._pp_current_slot
            req = scheduler.chunked_req
            if req is None and len(batch.reqs) < 8:
                req = next(
                    (r for r in scheduler.waiting_queue if state.can_admit(r, slot)),
                    None,
                )
                if req is not None:
                    scheduler.waiting_queue.remove(req)
            if req is None:
                return SimpleNamespace(batch_to_run=None, running_batch=batch)
            req.remaining_chunks -= 1
            req.extend_range.end = len(req.prefix_indices) + 1
            scheduler.chunked_req = req if req.remaining_chunks else None
            return SimpleNamespace(
                batch_to_run=_Batch([req], extend=True, chunk=scheduler.chunked_req),
                running_batch=batch,
            )

        scheduler.get_new_batch_prefill = prefill
        get_next = _scheduler_method(
            "get_next_batch_to_run",
            NextBatchPlan=SimpleNamespace,
            set_schedule_time_batch=lambda batch: None,
        )
        for step in range(600):
            slot = step % 2
            state.load(scheduler, slot)
            plan = get_next(scheduler, running[slot], last[slot])
            state.check_plan(
                slot, plan.running_batch, plan.batch_to_run, scheduler.chunked_req
            )
            state.save(scheduler)
            running[slot] = plan.running_batch
            last[slot] = plan.batch_to_run
            if last[slot] and last[slot].forward_mode.is_extend():
                for req in last[slot].reqs:
                    seen[id(req)].append(slot)
            if (
                not scheduler.waiting_queue
                and state.is_drained()
                and all(len(batch.reqs) == 8 for batch in running)
            ):
                break
        else:
            self.fail("slot-sticky chunked prefill did not drain")
        self.assertEqual([len(batch.reqs) for batch in running], [8, 8])
        self.assertEqual(len({id(req) for batch in running for req in batch.reqs}), 16)
        for req, kv_id in zip(requests, kv_ids):
            self.assertEqual(seen[id(req)], [req._pp_slot_affinity_owner] * 32)
            self.assertIs(req.kv_identity, kv_id)

    def test_live_chunk_reserves_capacity_and_duplicate_views_count_once(self):
        state = PPSlotAffinityState(2, 8)
        reqs = [_Req(str(i)) for i in range(9)]
        state.check_plan(0, _Batch(reqs[:7]), _Batch(reqs[7:8]), reqs[7])
        with self.assertRaisesRegex(RuntimeError, "exceeding"):
            state.check_plan(0, _Batch(reqs[:8]), _Batch(reqs[8:]), reqs[8])

    def test_retracted_request_keeps_owner_and_cannot_jump_slot(self):
        state = PPSlotAffinityState(2, 8)
        req = _Req("0")
        state.check_plan(0, _Batch(), _Batch([req]), req)
        self.assertFalse(state.can_admit(req, 1))
        self.assertTrue(state.can_admit(req, 0))
        with self.assertRaisesRegex(RuntimeError, "moved from slot"):
            state.check_plan(1, _Batch(), _Batch([req]), req)

    def test_abort_all_drains_both_slot_chunks_with_real_abort_processor(self):
        state = PPSlotAffinityState(2, 8)
        reqs = [_Req("0"), _Req("1")]
        state.chunks[:] = reqs
        released = []
        mode = SimpleNamespace(NULL="null", PREFILL="prefill")

        def prepare(req, reason):
            req.done = True

        def release(req, cache, **kwargs):
            released.append(req)
            req.req_pool_idx = None

        process = _scheduler_method(
            "process_pending_chunked_abort",
            prepare_abort=prepare,
            release_kv_cache=release,
            DisaggregationMode=mode,
            AbortReq=lambda **kwargs: SimpleNamespace(**kwargs),
            logger=MagicMock(),
        )
        scheduler = SimpleNamespace(
            disaggregation_mode="null",
            enable_hicache_storage=False,
            tree_cache=object(),
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(send_output=MagicMock())
            ),
        )
        state.load(scheduler, 0)
        state.queue_abort(scheduler, SimpleNamespace(abort_all=True, rid=""))
        process(scheduler)
        state.save(scheduler)
        self.assertFalse(state.is_drained())
        self.assertIs(state.pending_aborts[1], reqs[1])
        state.load(scheduler, 1)
        process(scheduler)
        state.save(scheduler)
        self.assertEqual(released, reqs)
        self.assertTrue(state.is_drained())
        self.assertEqual(
            scheduler.ipc_channels.send_to_tokenizer.send_output.call_count, 2
        )

    def test_target_abort_of_inactive_slot_preserves_active_pointer(self):
        state = PPSlotAffinityState(2, 8)
        reqs = [_Req("0"), _Req("1")]
        state.chunks[:] = reqs
        scheduler = SimpleNamespace()
        state.load(scheduler, 0)
        state.queue_abort(scheduler, SimpleNamespace(abort_all=False, rid="1"))
        self.assertIsNone(scheduler._pending_chunked_abort_req)
        self.assertEqual(state.pending_aborts, [None, reqs[1]])
        state.save(scheduler)
        state.load(scheduler, 1)
        self.assertIs(scheduler._pending_chunked_abort_req, reqs[1])

    def test_real_idle_check_and_reset_refuse_hidden_live_slot(self):
        state = PPSlotAffinityState(2, 8)
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(pp_size=2),
            _pp_slot_affinity=state,
            running_mbs=[_Batch(), _Batch()],
            mbs=[None, None],
        )
        drained = _scheduler_method("_pp_microbatches_drained")
        state.chunks[0] = _Req("0")
        self.assertFalse(drained(scheduler))
        with self.assertRaisesRegex(RuntimeError, "live requests"):
            state.reset()
        state.chunks[0] = None
        state.reset()
        self.assertTrue(drained(scheduler))

    def test_real_pause_rejects_before_mutating_any_state(self):
        pause = _scheduler_method("pause_generation")
        for mode in ("in_place", "retract"):
            scheduler = SimpleNamespace(
                _pp_slot_affinity=PPSlotAffinityState(2, 8), _engine_paused=False
            )
            with self.assertRaisesRegex(NotImplementedError, "unsupported"):
                pause(scheduler, SimpleNamespace(mode=mode))
            self.assertFalse(scheduler._engine_paused)

    def test_unsupported_modes_fail_closed(self):
        PPSlotAffinityState.validate_supported(pd=False, speculative=False)
        for name in ("pd", "elastic", "hicache", "speculative", "dllm", "dp_attention"):
            with self.assertRaisesRegex(ValueError, name):
                PPSlotAffinityState.validate_supported(**{name: True})

    def test_pp_loop_restores_slot_before_request_handlers_and_checks_before_launch(
        self,
    ):
        tree = ast.parse((MANAGERS / "scheduler_pp_mixin.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "SchedulerPPMixin"
        )
        loop = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "event_loop_pp"
        )
        source = ast.unparse(loop)
        self.assertLess(
            source.index(".load(self, mb_id)"),
            source.index("self.process_input_requests"),
        )
        self.assertLess(
            source.index(".check_plan("), source.index("self._pp_launch_or_skip_batch")
        )
        self.assertLess(
            source.index(".save(self)"), source.index("self._pp_process_batch_result")
        )


if __name__ == "__main__":
    unittest.main()
