"""Transactional full-attention KV and GDN state residency for elastic PP."""

from __future__ import annotations

from bisect import bisect_left
from threading import RLock

import torch

from sglang.srt.elastic_pp.kv_residency import ElasticKVResidency
from sglang.srt.elastic_pp.topology import LayerRange
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MambaPool,
    MHATokenToKVPool,
)


class ElasticHybridLinearKVPool(HybridLinearKVPool):
    """Keep global model ids distinct from dense attention/state pool ids.

    Attention KV uses the existing native or KVCacheD transaction. Recurrent
    state has a separate request-slot address space and is copied at the drained
    fence. Preparing a target must never snapshot an actively changing state.
    """

    def __init__(
        self,
        *,
        req_to_token_pool,
        all_full_attention_layer_ids,
        total_num_layers: int,
        end_layer: int,
        full_kv_pool_class=None,
        **kwargs,
    ):
        self.req_to_token_pool = req_to_token_pool
        self._all_full_ids = tuple(all_full_attention_layer_ids)
        self._total_num_layers = int(total_num_layers)
        if (
            not self._all_full_ids
            or self._all_full_ids != tuple(sorted(set(self._all_full_ids)))
            or not 0
            <= self._all_full_ids[0]
            <= self._all_full_ids[-1]
            < total_num_layers
        ):
            raise ValueError("elastic hybrid attention ids must be nonempty and sorted")
        self._full_id_to_ordinal = {
            layer_id: i for i, layer_id in enumerate(self._all_full_ids)
        }
        self._all_mamba_ids = tuple(
            i for i in range(total_num_layers) if i not in self._full_id_to_ordinal
        )
        state = req_to_token_pool.mamba_pool.mamba_cache
        if type(state) is not MambaPool.State or any(
            getattr(state, name) is not None
            for name in vars(state)
            if name not in {"conv", "temporal"}
        ):
            raise NotImplementedError(
                "elastic hybrid PP supports plain GDN state, without speculative "
                "state or ReplaySSM"
            )
        if getattr(req_to_token_pool, "mamba_ckpt_pool", None) is not None:
            raise NotImplementedError(
                "elastic hybrid PP does not support int8 checkpoints"
            )
        if kwargs.get("use_mla") or kwargs.get("post_capture_active"):
            raise NotImplementedError(
                "elastic hybrid PP requires ordinary MHA KV backing"
            )

        initial = LayerRange(int(kwargs.get("start_layer") or 0), int(end_layer))
        full_range = self._attention_range(initial)
        pool_class = full_kv_pool_class or MHATokenToKVPool
        full_kwargs = {
            key: kwargs[key]
            for key in (
                "size",
                "page_size",
                "dtype",
                "head_num",
                "head_dim",
                "device",
                "enable_memory_saver",
                "enable_kv_cache_copy",
                "quant_method",
            )
            if key in kwargs
        }
        if getattr(pool_class, "is_kvcached_elastic_pool", False):
            full_kwargs["total_num_layers"] = len(self._all_full_ids)
        full_pool = pool_class(
            layer_num=full_range.size,
            start_layer=full_range.start,
            end_layer=full_range.end,
            **full_kwargs,
        )
        super().__init__(full_kv_pool=full_pool, **kwargs)
        self.end_layer = initial.end
        self._publish_attention_mapping(initial)
        expected_mamba = self.recurrent_layer_ids(initial)
        if tuple(req_to_token_pool.mamba_map) != expected_mamba:
            raise ValueError(
                "initial GDN layer mapping does not match the PP partition"
            )
        self._elastic_lock = RLock()
        self._residency = ElasticKVResidency(
            num_layers=total_num_layers, initial_generation=0, initial_range=initial
        )
        self._pending_state = None
        self._pending_mamba_ids = ()
        self._retired_state = None
        self._retired_layer_ids = set()

    def _attention_range(self, target: LayerRange) -> LayerRange:
        return LayerRange(
            bisect_left(self._all_full_ids, target.start),
            bisect_left(self._all_full_ids, target.end),
        )

    def _publish_attention_mapping(self, target: LayerRange):
        self.full_attention_layer_id_mapping = {
            layer_id: ordinal
            for layer_id, ordinal in self._full_id_to_ordinal.items()
            if target.start <= layer_id < target.end
        }
        self.full_layer_nums = len(self.full_attention_layer_id_mapping)

    def elastic_pp_kv_layer_ids(self, target: LayerRange):
        return tuple(i for i in self._all_full_ids if target.start <= i < target.end)

    def recurrent_layer_ids(self, target: LayerRange):
        return tuple(i for i in self._all_mamba_ids if target.start <= i < target.end)

    def get_v_head_dim(self):
        # Inactive ranks and GDN-only ranges have no attention buffers yet.
        # The backing pool's shape metadata remains valid in both cases.
        return self.full_kv_pool.get_v_head_dim()

    @property
    def recurrent_slot_count(self):
        return self.mamba_pool.size + 1

    @property
    def is_kvcached_elastic_pool(self):
        return getattr(self.full_kv_pool, "is_kvcached_elastic_pool", False)

    @property
    def physical_layer_partitioned(self):
        return getattr(self.full_kv_pool, "physical_layer_partitioned", False)

    @property
    def kvcached_allocator(self):
        return self.full_kv_pool.kvcached_allocator

    def attach_logical_allocator(self, allocator):
        self.full_kv_pool.attach_logical_allocator(allocator)

    def kvcached_physical_stats(self):
        return self.full_kv_pool.kvcached_physical_stats()

    @staticmethod
    def _state_tensors(state):
        if state is None:
            return ()
        return (*state.conv, state.temporal)

    def _pending(self, generation):
        pending = self._residency.pending
        if pending is None or pending.generation != generation:
            raise RuntimeError(f"no hybrid KV preparation for generation {generation}")
        return pending

    def prepare_elastic_pp_kv_layers(self, *, target, generation, total_num_layers):
        if total_num_layers != self._total_num_layers:
            raise ValueError("elastic hybrid model-layer count changed")
        with self._elastic_lock:
            current = set(range(self.start_layer, self.end_layer))
            pending = self._residency.begin_prepare(
                generation=generation, target=target, resident_layer_ids=current
            )
            if self._pending_state is not None:
                return pending.new_layer_ids
            try:
                self.full_kv_pool.prepare_elastic_pp_kv_layers(
                    target=self._attention_range(target),
                    generation=generation,
                    total_num_layers=len(self._all_full_ids),
                )
                ids = self.recurrent_layer_ids(target)
                old = self.mamba_pool.mamba_cache
                if ids == tuple(self.req_to_token_pool.mamba_map):
                    state = old
                else:
                    # Retained layers are copied only at commit, after every
                    # source stream is drained. Their state can change while
                    # the new buffers are allocated in the background.
                    state = MambaPool.State(
                        conv=[
                            torch.zeros(
                                (len(ids), *t.shape[1:]), dtype=t.dtype, device=t.device
                            )
                            for t in old.conv
                        ],
                        temporal=torch.zeros(
                            (len(ids), *old.temporal.shape[1:]),
                            dtype=old.temporal.dtype,
                            device=old.temporal.device,
                        ),
                    )
                self._pending_state = state
                self._pending_mamba_ids = ids
            except BaseException:
                self.full_kv_pool.abort_elastic_pp_kv_layers(generation=generation)
                self._residency.abort(generation)
                self._pending_state = None
                self._pending_mamba_ids = ()
                raise
            return pending.new_layer_ids

    def export_elastic_pp_kv_slots(self, *, layer_id, slot_ids):
        return self.full_kv_pool.export_elastic_pp_kv_slots(
            layer_id=self._full_id_to_ordinal[layer_id], slot_ids=slot_ids
        )

    def import_elastic_pp_kv_slots(self, *, generation, layer_id, **kwargs):
        self._pending(generation)
        return self.full_kv_pool.import_elastic_pp_kv_slots(
            generation=generation, layer_id=self._full_id_to_ordinal[layer_id], **kwargs
        )

    def export_recurrent_slots(self, *, layer_id, slot_ids):
        index = self.req_to_token_pool.mamba_map[layer_id]
        state = self.mamba_pool.mamba_cache.at_layer_idx(index)
        return {
            **{
                f"conv_{i}": t.index_select(0, slot_ids)
                for i, t in enumerate(state.conv)
            },
            "temporal": state.temporal.index_select(0, slot_ids),
        }

    def import_recurrent_slots(self, *, generation, layer_id, slot_ids, tensors):
        pending = self._pending(generation)
        if (
            layer_id not in pending.new_layer_ids
            or layer_id not in self._pending_mamba_ids
        ):
            raise ValueError(f"unexpected recurrent layer {layer_id}")
        if slot_ids.dtype != torch.int64 or slot_ids.ndim != 1:
            raise ValueError("recurrent slot ids must be an int64 vector")
        index = self._pending_mamba_ids.index(layer_id)
        state = self._pending_state.at_layer_idx(index)
        targets = {
            **{f"conv_{i}": t for i, t in enumerate(state.conv)},
            "temporal": state.temporal,
        }
        if set(tensors) != set(targets):
            raise ValueError("recurrent state tensor set does not match the target")
        for name, target in targets.items():
            source = tensors[name]
            if source.dtype != target.dtype or source.shape != (
                slot_ids.numel(),
                *target.shape[1:],
            ):
                raise ValueError(f"recurrent state shape/dtype mismatch for {name}")
        for name, target in targets.items():
            target.index_copy_(0, slot_ids, tensors[name])

    def prepare_elastic_pp_kv_fence(self, *, generation):
        self._pending(generation)
        method = getattr(self.full_kv_pool, "prepare_elastic_pp_kv_fence", None)
        if method is not None:
            method(generation=generation)

    def mark_elastic_pp_kv_layers_loaded(self, *, generation, layer_ids):
        self._pending(generation)
        full_ids = {
            self._full_id_to_ordinal[i]
            for i in layer_ids
            if i in self._full_id_to_ordinal
        }
        if full_ids:
            self.full_kv_pool.mark_elastic_pp_kv_layers_loaded(
                generation=generation, layer_ids=full_ids
            )
        self._residency.mark_loaded(generation, set(layer_ids))

    def elastic_pp_kv_ready(self, *, generation):
        return self._residency.ready_to_commit(generation) and (
            self.full_kv_pool.elastic_pp_kv_ready(generation=generation)
        )

    def commit_elastic_pp_kv_layers(self, *, generation):
        with self._elastic_lock:
            self._pending(generation)
            if not self.elastic_pp_kv_ready(generation=generation):
                raise RuntimeError("hybrid attention/state generation is not ready")
            old = self.mamba_pool.mamba_cache
            new = self._pending_state
            if new is not old:
                for new_index, layer_id in enumerate(self._pending_mamba_ids):
                    old_index = self.req_to_token_pool.mamba_map.get(layer_id)
                    if old_index is None:
                        continue
                    for dst, src in zip(
                        self._state_tensors(new), self._state_tensors(old)
                    ):
                        dst[new_index].copy_(src[old_index])
            self.full_kv_pool.commit_elastic_pp_kv_layers(generation=generation)
            old_ids = set(range(self.start_layer, self.end_layer))
            target = self._residency.commit(generation)
            self._retired_layer_ids = old_ids - set(range(target.start, target.end))
            self._retired_state = old if old is not new else None
            self.mamba_pool.mamba_cache = new
            self.mamba_pool.mamba_layer_ids = list(self._pending_mamba_ids)
            self.mamba_pool.num_mamba_layers = len(self._pending_mamba_ids)
            self.mamba_pool.mem_usage = new.mem_usage_bytes() / 2**30
            # Cached fused clear/COW descriptors contain the old tensor pointers.
            self.mamba_pool.__dict__.pop("_conv_slot_desc", None)
            self.mamba_pool.__dict__.pop("_conv_fuse_ok", None)
            self.req_to_token_pool.mamba_map = {
                layer_id: i for i, layer_id in enumerate(self._pending_mamba_ids)
            }
            self.req_to_token_pool.start_layer = target.start
            self.req_to_token_pool.layer_transfer_counter = None
            self.start_layer, self.end_layer = target.start, target.end
            self._publish_attention_mapping(target)
            self.layer_transfer_counter = None
            self._pending_state = None
            self._pending_mamba_ids = ()
            k_bytes, v_bytes = self.full_kv_pool.get_kv_size_bytes()
            self.mem_usage = (k_bytes + v_bytes) / 2**30
            return target

    def abort_elastic_pp_kv_layers(self, *, generation):
        with self._elastic_lock:
            released = self.full_kv_pool.abort_elastic_pp_kv_layers(
                generation=generation
            )
            self._residency.abort(generation)
            state = self._pending_state
            self._pending_state = None
            self._pending_mamba_ids = ()
            extra = (
                ()
                if state is self.mamba_pool.mamba_cache
                else self._state_tensors(state)
            )
            return (*released, *extra)

    def cleanup_elastic_pp_kv_layers(self, *, generation):
        with self._elastic_lock:
            released = self.full_kv_pool.cleanup_elastic_pp_kv_layers(
                generation=generation
            )
            ids = self._residency.cleanup_candidates(
                generation, self._retired_layer_ids
            )
            self._residency.finish_cleanup(generation, set(ids))
            state = self._retired_state
            self._retired_state = None
            self._retired_layer_ids = set()
            return (*released, *self._state_tensors(state))
