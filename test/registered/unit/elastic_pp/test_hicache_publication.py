"""Exercise production cache users after the source generation is retired."""

import dataclasses
from array import array
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.elastic_pp.hicache_transition import (
    HiCacheTransitionAdapter,
    SchedulerHiCacheTransition,
    clone_cache_runtime_metadata,
    discard_host_cache,
    publish_scheduler_cache,
)
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req, ReqKvInfo
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams


def make_cache(*, session_refs=False):
    pool = MHATokenToKVPool(
        size=32,
        page_size=1,
        dtype=torch.float32,
        head_num=1,
        head_dim=2,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=32, dtype=torch.float32, device="cpu", kvcache=pool, need_sort=False
    )
    req_pool = ReqToTokenPool(
        size=4, max_context_len=32, device="cpu", enable_memory_saver=False
    )
    return UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            tree_components=(ComponentType.FULL,),
            enable_session_radix_cache=session_refs,
        )
    )


def context(cls, **values):
    # Unused resources can be None; construct the real production dataclass,
    # with its actual required field set and frozen/slots behavior.
    arguments = {
        field.name: None
        for field in dataclasses.fields(cls)
        if field.init
        and field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    }
    arguments.update(values)
    return cls(**arguments)


def make_processor(cache, **values):
    return context(
        SchedulerBatchResultProcessor,
        tree_cache=cache,
        req_to_token_pool=cache.req_to_token_pool,
        token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
        model_worker=NS(),
        **values,
    )


def make_req(cache, *, rid="request", tokens=(1, 2, 3, 4)):
    params = SamplingParams(max_new_tokens=1)
    params.normalize(None)
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", tokens[:-1]),
        sampling_params=params,
        vocab_size=128,
    )
    req.output_ids.append(tokens[-1])
    req._refresh_fill_ids()
    req.set_extend_range(0, len(tokens))
    req.req_pool_idx = cache.req_to_token_pool.alloc([req])[0]
    req.kv = ReqKvInfo(kv_allocated_len=len(tokens), swa_evicted_seqlen=0)
    req.kv_committed_len = len(tokens)
    indices = cache.token_to_kv_pool_allocator.alloc(len(tokens))
    cache.req_to_token_pool.write((req.req_pool_idx, slice(0, len(tokens))), indices)
    req.last_node = cache.tree_core.root_node.id
    return req, indices


@pytest.fixture
def ordinary_finish(monkeypatch):
    from sglang.srt.managers import schedule_batch
    from sglang.srt.managers.scheduler_components import batch_result_processor
    from sglang.srt.mem_cache import common

    monkeypatch.setattr(
        batch_result_processor,
        "get_disagg",
        lambda: NS(disaggregation_decode_enable_offload_kvcache=False),
    )
    monkeypatch.setattr(
        batch_result_processor, "get_memory", lambda: NS(enable_hisparse=False)
    )
    monkeypatch.setattr(
        batch_result_processor, "mamba_extra_buffer_lazy_enabled", lambda: False
    )
    monkeypatch.setattr(common, "get_spec", lambda: NS(speculative_algorithm=None))
    monkeypatch.setattr(common, "get_serving", lambda: NS(strip_thinking_cache=False))
    monkeypatch.setattr(
        schedule_batch, "get_serving", lambda: NS(strip_thinking_cache=False)
    )


def test_finished_decode_uses_current_frozen_context_after_retirement(ordinary_finish):
    cache = make_cache()
    scheduler = NS(tree_cache=cache, batch_result_processor=make_processor(cache))
    for generation in (1, 2):
        source = scheduler.tree_cache
        req, indices = make_req(
            source, rid=f"generation-{generation}", tokens=(generation, 9, 8, 7)
        )
        target = clone_cache_runtime_metadata(source)
        old_processor = scheduler.batch_result_processor
        publish_scheduler_cache(scheduler, target, NS(canary_manager=None))
        discard_host_cache(source)
        req.finished_reason = FINISH_LENGTH(1)
        batch = NS(
            mamba_track_mask_cpu=[False],
            mamba_decode_batch_idx_cpu=[req.decode_batch_idx],
        )
        scheduler.batch_result_processor._handle_finish_state_updated_req(
            req, batch, NS(), 0, None
        )
        assert source.req_to_token_pool is None
        assert old_processor.tree_cache is source
        assert scheduler.batch_result_processor is not old_processor
        assert req.req_pool_idx is None and req.kv is None
        matched = target.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [generation, 9, 8, 7])))
        )
        assert matched.device_indices.tolist() == indices.tolist()


@pytest.mark.parametrize("streaming", [False, True])
def test_unfinished_request_uses_target_pool_after_retirement(streaming):
    from sglang.srt.mem_cache.common import maybe_cache_unfinished_req

    source = make_cache()
    req, indices = make_req(source)
    target = clone_cache_runtime_metadata(source)
    scheduler = NS(tree_cache=source, batch_result_processor=make_processor(source))
    publish_scheduler_cache(scheduler, target, NS(canary_manager=None))
    discard_host_cache(source)
    if streaming:
        req.session = NS(streaming=True, session_id="streaming")
    maybe_cache_unfinished_req(
        req, scheduler.batch_result_processor.tree_cache, chunked=True
    )
    assert req.prefix_indices.tolist() == indices.tolist()
    assert target.session is not source.session
    assert target.session.inner is target
    assert source.session.inner is source


@pytest.mark.parametrize("publication_planning_fails", [False, True])
def test_unpublished_abort_keeps_source_contexts_and_live_request(
    ordinary_finish, monkeypatch, publication_planning_fails
):
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )
    from sglang.srt.session.streaming_session import SessionSlot

    source = make_cache()
    req, indices = make_req(source)
    slot = source.session.slots["retained"] = SessionSlot()
    target = clone_cache_runtime_metadata(source)
    target.session.slots["unpublished"] = SessionSlot()
    streamer = context(SchedulerOutputStreamer, tree_cache=source)
    processor = make_processor(source, output_streamer=streamer)
    policy, sessions = NS(tree_cache=source), NS(tree_cache=source)
    scheduler = NS(
        tree_cache=source,
        output_streamer=streamer,
        batch_result_processor=processor,
        policy=policy,
        session_controller=sessions,
    )
    transition = NS(target=NS(generation=1))
    manager = SchedulerHiCacheTransition(
        scheduler=scheduler, model_runner=NS(), group=None, rank=0
    )
    manager.adapter = HiCacheTransitionAdapter(
        cache=source, pools={}, group=None, rank=0
    )
    manager.adapter.prepare_transition(transition)
    manager.target_cache = target
    if publication_planning_fails:
        # A failed frozen-context construction must not apply previously
        # planned changes to a mutable streamer or any scheduler root.
        def fail_replace(*args, **kwargs):
            raise RuntimeError("context construction failed")

        with monkeypatch.context() as patch:
            patch.setattr(dataclasses, "replace", fail_replace)
            with pytest.raises(RuntimeError, match="context construction failed"):
                publish_scheduler_cache(scheduler, target, NS(canary_manager=None))
    manager.abort(transition)
    manager.abort(transition)
    assert scheduler.tree_cache is source
    assert scheduler.batch_result_processor is processor
    assert processor.tree_cache is source and processor.output_streamer is streamer
    assert streamer.tree_cache is source
    assert policy.tree_cache is source and sessions.tree_cache is source
    assert source.session.inner is source
    assert source.session.slots == {"retained": slot}
    assert source.components[ComponentType.FULL].cache is source
    assert target.req_to_token_pool is None
    req.finished_reason = FINISH_LENGTH(1)
    batch = NS(
        mamba_track_mask_cpu=[False],
        mamba_decode_batch_idx_cpu=[req.decode_batch_idx],
    )
    processor._handle_finish_state_updated_req(req, batch, NS(), 0, None)
    assert req.req_pool_idx is None and req.kv is None
    matched = source.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4])))
    )
    assert matched.device_indices.tolist() == indices.tolist()


def test_real_scheduler_context_graph_keeps_mutable_resources_and_shared_observer():
    from sglang.srt.managers.scheduler_components.dp_attn import SchedulerDPAttnAdapter
    from sglang.srt.managers.scheduler_components.invariant_checker import (
        SchedulerInvariantChecker,
    )
    from sglang.srt.managers.scheduler_components.kv_events_publisher import (
        SchedulerKvEventsPublisher,
    )
    from sglang.srt.managers.scheduler_components.load_inquirer import (
        SchedulerLoadInquirer,
    )
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )
    from sglang.srt.managers.scheduler_components.pool_stats_observer import (
        SchedulerPoolStatsObserver,
    )

    source = make_cache()
    target = clone_cache_runtime_metadata(source)
    observer = context(
        SchedulerPoolStatsObserver,
        tree_cache=source,
        req_to_token_pool=source.req_to_token_pool,
        token_to_kv_pool_allocator=source.token_to_kv_pool_allocator,
        is_hybrid_swa=False,
        is_hybrid_ssm=False,
        enable_hisparse=False,
        max_total_num_tokens=32,
    )
    streamer = context(SchedulerOutputStreamer, tree_cache=source)
    streamer._test_stream_output_count = 17
    checker = context(
        SchedulerInvariantChecker, tree_cache=source, pool_stats_observer=observer
    )
    checker.count_req_pool_leak_warnings = 3
    events = context(SchedulerKvEventsPublisher, tree_cache=source)
    publisher = events.kv_event_publisher = object()
    inquirer = context(SchedulerLoadInquirer, pool_stats_observer=observer)
    adapter = context(SchedulerDPAttnAdapter, tree_cache=source)
    processor = make_processor(source, output_streamer=streamer)
    scheduler = NS(
        tree_cache=source,
        batch_result_processor=processor,
        output_streamer=streamer,
        dp_attn_adapter=adapter,
        invariant_checker=checker,
        kv_events_publisher=events,
        load_inquirer=inquirer,
        pool_stats_observer=observer,
    )
    publish_scheduler_cache(scheduler, target, NS(canary_manager=None))
    discard_host_cache(source)
    assert scheduler.batch_result_processor.tree_cache is target
    assert scheduler.dp_attn_adapter.tree_cache is target
    assert scheduler.pool_stats_observer is not observer
    assert observer.tree_cache is source
    assert scheduler.pool_stats_observer.get_pool_stats().full_num_used == 0
    assert scheduler.load_inquirer is not inquirer
    assert scheduler.load_inquirer.pool_stats_observer is scheduler.pool_stats_observer
    assert checker.pool_stats_observer is scheduler.pool_stats_observer
    assert checker.tree_cache is target and checker.count_req_pool_leak_warnings == 3
    assert scheduler.output_streamer is streamer
    assert streamer.tree_cache is target and streamer._test_stream_output_count == 17
    assert scheduler.batch_result_processor.output_streamer is streamer
    assert scheduler.kv_events_publisher is events
    assert events.tree_cache is target and events.kv_event_publisher is publisher


def test_cloned_components_and_session_own_the_target_generation():
    from sglang.srt.session.streaming_session import SessionSlot

    source = make_cache(session_refs=True)
    slot = source.session.slots["retained"] = SessionSlot()
    target = clone_cache_runtime_metadata(source)
    assert target.session.slots is not source.session.slots
    assert target.session.slots["retained"] is slot
    full = target.components[ComponentType.FULL]
    assert full.cache is target and full.tree_core is target.tree_core
    assert target.tree_core.components_by_type[ComponentType.FULL] is full
    assert target.tree_core.components == (full,)
    assert target.session_refs.components == (full,)
    assert target.session_refs.tree_core is target.tree_core
    assert full.session_ref_eviction_strategy.__self__ is full
    assert (
        source.components[ComponentType.FULL].session_ref_eviction_strategy.__self__
        is source.components[ComponentType.FULL]
    )
    discard_host_cache(source)
    assert target.session.req_to_token_pool is target.req_to_token_pool
    assert (
        target.session.token_to_kv_pool_allocator is target.token_to_kv_pool_allocator
    )
