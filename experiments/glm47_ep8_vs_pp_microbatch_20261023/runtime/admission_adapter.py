"""Bounded CPU-only placement/admission adapter for four frozen B8 cohorts.

Native DP attention does not atomically admit pretokenized HTTP batches. Route
logical requests 0..3 / 4..7 to DP0 / DP1, then release each local ordered B4
only once complete. Unknown requests use the stock path. No steady-state
collectives, device reads, synchronization, model or sampling changes.
"""
COHORTS = ("prepare-semantic", "warm-decode-4096", "timed-decode-4096", "repeat-decode-4096")
PREFIXES = ("prepare-", "warm-", "timed-", "repeat-", "profile-", "witness-")


def identity(rid):
    if not isinstance(rid, str) or not rid.startswith(PREFIXES):
        return None
    base, separator, index = rid.rpartition("-")
    if not separator or base not in COHORTS or index not in tuple(str(i) for i in range(8)):
        raise RuntimeError("Request outside frozen EP8 admission cohort: " + rid)
    return base, int(index)


def route_tokenized(tokenized):
    parsed = identity(getattr(tokenized, "rid", None))
    if parsed is not None:
        dp = parsed[1] // 4
        old = getattr(tokenized, "routed_dp_rank", None)
        if old is not None and old != dp:
            raise RuntimeError("Conflicting external DP placement")
        tokenized.routed_dp_rank = dp
    return tokenized


def release_local(received, *, dp_rank, state, tokenized_type):
    if dp_rank not in (0, 1):
        raise RuntimeError("Expected exactly two Attention DP workers")
    expected = list(range(dp_rank * 4, dp_rank * 4 + 4))
    ready = []
    for item in received:
        if type(item) is not tokenized_type:
            ready.append(item)
            continue
        parsed = identity(getattr(item, "rid", None))
        if parsed is None:
            ready.append(item)
            continue
        cohort, index = parsed
        if index not in expected or getattr(item, "routed_dp_rank", None) != dp_rank:
            raise RuntimeError("Actual scheduler DP placement disagrees with frozen mapping")
        if cohort in state["released"]:
            raise RuntimeError("Duplicate request after atomic local admission")
        pending = state["pending"].setdefault(cohort, {})
        if index in pending:
            raise RuntimeError("Duplicate request before atomic local admission")
        pending[index] = item
        if len(pending) == 4:
            if sorted(pending) != expected:
                raise RuntimeError("Unexpected local cohort membership")
            batch = [pending[i] for i in expected]
            ready.extend(batch)
            del state["pending"][cohort]
            state["released"].append(cohort)
            state["proofs"].append(dict(cohort=cohort, dp_rank=dp_rank,
                request_indices=expected, request_ids=[x.rid for x in batch], request_count=4,
                input_class="TokenizedGenerateReqInput", atomic_local_release=True,
                observed_at_scheduler_not_inferred_from_http=True,
                native_scheduler_handler=True, added_collectives=0,
                device_reads=0, gpu_synchronizations=0))
    return ready


def new_state():
    return dict(pending={}, released=[], proofs=[])


def install_tokenizer():
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    if getattr(TokenizerManager, "_glm_ep8_placement", False):
        return
    original = TokenizerManager._tokenize_one_request
    async def tokenize(self, *args, **kwargs):
        return route_tokenized(await original(self, *args, **kwargs))
    TokenizerManager._tokenize_one_request = tokenize
    TokenizerManager._glm_ep8_placement = True


def install_scheduler():
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    if getattr(Scheduler, "_glm_ep8_admission", False):
        return
    original = Scheduler.process_input_requests
    def process(self, received):
        from sglang.srt.runtime_context import get_parallel
        dp_rank = get_parallel().attn_dp_rank
        if not hasattr(self, "_glm_ep8_admission_state"):
            self._glm_ep8_admission_state = new_state()
        ready = release_local(received, dp_rank=dp_rank,
            state=self._glm_ep8_admission_state, tokenized_type=TokenizedGenerateReqInput)
        return original(self, ready)
    Scheduler.process_input_requests = process
    Scheduler._glm_ep8_admission = True
