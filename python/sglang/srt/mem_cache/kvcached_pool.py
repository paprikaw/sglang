"""Explicit KVCacheD pools for transactional elastic pipeline parallelism.

This module deliberately does not use KVCacheD's SGLang auto-patch.  Elastic
PP needs one stable, global layer-indexed virtual backing while the published
SGLang attention views contain only the layers owned by the current topology.
Keeping that contract here also lets the prepare/fence/commit coordinator own
when a new view becomes visible.

The optional :mod:`kvcached` dependency is imported only when this backend is
selected.  Normal SGLang imports and the native KV-cache path therefore remain
independent of KVCacheD.
"""

from __future__ import annotations

import inspect
import logging
import math
import os
import threading
import time
from typing import Any, Iterable, Optional

import torch

from sglang.srt.elastic_pp.kv_slots import ElasticKVSlotTracker
from sglang.srt.elastic_pp.topology import LayerRange
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

_LISTENER_LOCK = threading.Lock()
_LISTENER_KEYS: set[tuple[int, int, int]] = set()


def _explicit_kvcached_device(device: str | torch.device) -> str:
    """Resolve index-less CUDA devices against this process's active GPU."""

    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return str(resolved)


def _supported_kwargs(callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep compatibility with both upstream and the autoscaling KVCacheD API."""

    parameters = inspect.signature(callable_obj).parameters
    if any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    ):
        return kwargs
    return {name: value for name, value in kwargs.items() if name in parameters}


def _load_kvcached_interfaces():
    # Ownership-aware layer filtering must be installed before the first page
    # is mapped.  KVCacheD imports this switch as a module constant, so set the
    # safe default before importing the optional package.
    os.environ.setdefault("KVCACHED_PAGE_PREALLOC_ENABLED", "false")
    os.environ.setdefault("KVCACHED_LAYER_STACKING", "true")
    os.environ.setdefault("KVCACHED_CONTIGUOUS_LAYOUT", "false")
    try:
        import kvcached.integration.sglang.interfaces as interfaces
    except ImportError as exc:
        raise RuntimeError(
            "--elastic-pp-kv-backend=kvcached requires an installed KVCacheD "
            "package with its SGLang integration"
        ) from exc
    return interfaces


class KVCacheDMHATokenToKVPool(MHATokenToKVPool):
    """MHA KV pool with global KVCacheD backing and local active layer views.

    KVCacheD reserves virtual tensors for every model layer once.  ``k_buffer``
    and ``v_buffer`` remain the ordinary local SGLang lists, so existing
    attention kernels do not need to know about elastic ownership.  A topology
    commit atomically rebinds those lists to a different slice of the global
    backing.

    The currently available KVCacheD SGLang allocator maps a token page for all
    backing layers.  This is correctness-first and makes prepare/import cheap;
    it does not yet reclaim per-rank physical KV pages when a rank gives away a
    layer.  ``physical_layer_partitioned`` exposes that fact to profiling and
    prevents us from reporting virtual-layer rebinding as physical reclamation.
    """

    is_kvcached_elastic_pool = True
    physical_layer_partitioned = False
    _group_id = 0

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        *,
        total_num_layers: int,
        v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        **kwargs,
    ) -> None:
        self._kvcached_total_num_layers = int(total_num_layers)
        if self._kvcached_total_num_layers <= 0:
            raise ValueError("KVCacheD requires a positive total model-layer count")
        effective_v_head_dim = head_dim if v_head_dim is None else int(v_head_dim)
        if effective_v_head_dim != int(head_dim):
            raise NotImplementedError(
                "elastic PP KVCacheD currently requires equal K/V head dimensions"
            )
        resolved_start = int(start_layer or 0)
        resolved_end = (
            resolved_start + int(layer_num) if end_layer is None else int(end_layer)
        )
        if not (0 <= resolved_start <= resolved_end <= self._kvcached_total_num_layers):
            raise ValueError(
                "invalid KVCacheD active layer range: "
                f"[{resolved_start}, {resolved_end}) of "
                f"{self._kvcached_total_num_layers}"
            )
        if resolved_end - resolved_start != int(layer_num):
            raise ValueError(
                "KVCacheD local layer count disagrees with active layer range: "
                f"layer_num={layer_num}, range=[{resolved_start}, {resolved_end})"
            )

        self._kvcached_interfaces = self._get_kvcached_interfaces()
        self._kvcached_partitioning_requested = (
            self._detect_physical_layer_partitioning()
        )
        if self._kvcached_partitioning_requested:
            # The autoscaling KVCacheD vLLM interface already implements the
            # layer-stacked global geometry. The legacy SGLang interface does
            # not, so adapt the common lower-level per-layer tensors below.
            from kvcached.integration.vllm import interfaces as vllm_interfaces

            self._kvcached_interfaces = vllm_interfaces
        self._kvcached_global_k_buffer: list[torch.Tensor] = []
        self._kvcached_global_v_buffer: list[torch.Tensor] = []
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            v_head_dim=effective_v_head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=resolved_start,
            end_layer=resolved_end,
            **kwargs,
        )
        self.kvcached_allocator = self._create_kvcached_manager()
        self._logical_allocator = None
        if self._kvcached_partitioning_requested:
            self._configure_physical_layer_partitioning()
        self._validate_kvcached_capacity()
        if not self.physical_layer_partitioned:
            logger.warning(
                "Elastic PP KVCacheD uses global virtual backing and this "
                "KVCacheD build maps physical token pages across all %d model "
                "layers; install the layer-stacking migration API to enable "
                "per-PP-rank physical layer ownership",
                self._kvcached_total_num_layers,
            )

    def _get_kvcached_interfaces(self):
        return _load_kvcached_interfaces()

    def _initialize_kvcached(self) -> None:
        interfaces = self._kvcached_interfaces
        init_kwargs = {
            "tp_rank": int(get_parallel().tp_rank),
            "tp_size": int(get_parallel().tp_size),
            "world_size": int(get_parallel().tp_size),
            "pp_rank": int(get_parallel().pp_rank),
            "is_worker": False,
            "device": _explicit_kvcached_device(self.device),
            "async_sched": True,
        }
        interfaces.init_kvcached(
            **_supported_kwargs(interfaces.init_kvcached, init_kwargs)
        )
        if self._kvcached_partitioning_requested:
            # KVCacheManager's background initializer queries this socket even
            # for TP=1, so make it reachable before manager construction.
            self._start_local_mapping_listener()

    @staticmethod
    def _detect_physical_layer_partitioning() -> bool:
        try:
            from kvcached import tp_ipc_util
            from kvcached import utils as kvcached_utils

            return bool(
                getattr(kvcached_utils, "LAYER_STACKING", False)
                and not getattr(kvcached_utils, "PAGE_PREALLOC_ENABLED", True)
                and hasattr(tp_ipc_util, "register_worker_offset_filter")
                and hasattr(tp_ipc_util, "map_worker_offsets_for_migration")
                and hasattr(tp_ipc_util, "unmap_worker_offsets_for_layer_groups")
                and hasattr(tp_ipc_util, "snapshot_worker_mapped_offsets")
            )
        except ImportError:
            return False

    def _allocate_kvcached_global_buffers(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if torch.device(self.device).type != "cuda":
            raise ValueError("KVCacheD elastic PP backing requires a CUDA device")
        self._initialize_kvcached()
        interfaces = self._kvcached_interfaces
        kvcached_device = _explicit_kvcached_device(self.device)
        if self._kvcached_partitioning_requested:
            if self.page_size != 1:
                raise NotImplementedError(
                    "ownership-aware elastic PP KVCacheD currently requires "
                    "SGLang page_size=1 so a layer-stacked block view can be "
                    "published as an NHD token-row tensor"
                )
            num_blocks = math.ceil(self.size / self.page_size) + 1
            kwargs = {
                "kvcache_shape": (
                    2,
                    num_blocks,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                "block_size": self.page_size,
                "dtype": self.store_dtype,
                "device": kvcached_device,
                "num_layers": self._kvcached_total_num_layers,
                "attention_type": "MHA",
                "kv_layout": "NHD",
                "group_id": self._group_id,
                "physical_block_size": self._physical_block_size(),
            }
            combined = interfaces.alloc_kv_cache(
                **_supported_kwargs(interfaces.alloc_kv_cache, kwargs)
            )
            if not isinstance(combined, list):
                raise RuntimeError(
                    "layer-stacked KVCacheD allocation did not return "
                    "per-layer tensors"
                )
            global_k = []
            global_v = []
            for layer_id, tensor in enumerate(combined):
                if tensor.ndim != 5 or int(tensor.shape[0]) != 2:
                    raise RuntimeError(
                        "unexpected layer-stacked KVCacheD tensor shape: "
                        f"layer={layer_id}, shape={tuple(tensor.shape)}"
                    )
                global_k.append(tensor[0, :, 0, :, :])
                global_v.append(tensor[1, :, 0, :, :])
            return global_k, global_v

        kwargs = {
            "kvcache_shape": (
                self.size + self.page_size,
                self.head_num,
                self.head_dim,
            ),
            "dtype": self.store_dtype,
            "device": kvcached_device,
            "num_layers": self._kvcached_total_num_layers,
            "page_size": self.page_size,
            "attention_type": "MHA",
            "kv_layout": "NHD",
            "group_id": self._group_id,
        }
        result = interfaces.alloc_kv_cache(
            **_supported_kwargs(interfaces.alloc_kv_cache, kwargs)
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("KVCacheD MHA allocation did not return K/V tensors")
        return list(result[0]), list(result[1])

    @staticmethod
    def _physical_block_size() -> Optional[int]:
        try:
            from kvcached.utils import get_physical_block_size

            return get_physical_block_size()
        except (ImportError, AttributeError):
            return None

    def _create_buffers(self) -> None:
        if self.is_quantized_kv_cache:
            raise NotImplementedError(
                "elastic PP KVCacheD does not support quantized KV storage"
            )
        if self.post_capture_active:
            raise NotImplementedError(
                "elastic PP KVCacheD owns its VMM reservation and cannot use "
                "SGLang post-capture backing"
            )
        if self.use_hnd or self.kv_cache_layout != "nhd":
            raise NotImplementedError(
                "elastic PP KVCacheD currently supports only the NHD KV layout"
            )

        global_k, global_v = self._allocate_kvcached_global_buffers()
        if len(global_k) != self._kvcached_total_num_layers or len(global_v) != len(
            global_k
        ):
            raise RuntimeError(
                "KVCacheD global tensor count disagrees with the model: "
                f"K={len(global_k)}, V={len(global_v)}, "
                f"expected={self._kvcached_total_num_layers}"
            )
        self._kvcached_global_k_buffer = global_k
        self._kvcached_global_v_buffer = global_v
        self.k_buffer = global_k[self.start_layer : self.end_layer]
        self.v_buffer = global_v[self.start_layer : self.end_layer]
        self.k_scale_buffer = None
        self.v_scale_buffer = None
        self.dq_k_buffer = None
        self.dq_v_buffer = None
        self._kv_buffer_descs = self._build_kv_buffer_descs()
        self._init_data_ptrs_and_strides()

    def _create_kvcached_manager(self):
        interfaces = self._kvcached_interfaces
        cell_size = self.head_num * self.head_dim * self.store_dtype.itemsize
        kwargs = {
            "num_blocks": math.ceil(self.size / self.page_size) + 1,
            "block_size": self.page_size,
            "cell_size": cell_size,
            "num_layers": self._kvcached_total_num_layers,
            # In ownership-aware mode, install the local worker filter before
            # the manager maps its reserved null page.
            "reserve_null_block": not self._kvcached_partitioning_requested,
            "num_kv_buffers": 2,
            "group_id": self._group_id,
            "physical_block_size": self._physical_block_size(),
        }
        return interfaces.get_kv_cache_manager(
            **_supported_kwargs(interfaces.get_kv_cache_manager, kwargs)
        )

    def _active_layer_groups(self, target: Optional[LayerRange] = None) -> set[int]:
        if not self.physical_layer_partitioned and target is None:
            return set()
        granularity = int(self.kvcached_allocator.layer_group_granularity)
        active = target or LayerRange(self.start_layer, self.end_layer)
        if active.start % granularity or active.end % granularity:
            raise ValueError(
                "KVCacheD layer-stacking requires elastic PP boundaries aligned "
                f"to layer-group granularity {granularity}: range={active}"
            )
        return set(range(active.start // granularity, active.end // granularity))

    def _filter_worker_offsets(self, offsets: list[int], group_id: int) -> list[int]:
        if int(group_id) != int(self._group_id):
            return offsets
        group_total_span = self._layer_group_total_span()
        active_groups = self._active_layer_groups()
        return [
            int(offset)
            for offset in offsets
            if int(offset) // group_total_span in active_groups
        ]

    def _layer_group_total_span(self) -> int:
        manager = self.kvcached_allocator
        get_span = getattr(manager, "get_layer_group_total_span", None)
        if get_span is not None:
            return int(get_span())
        blocks_per_page = max(1, int(manager.blocks_per_physical_page))
        num_physical_pages = (
            int(manager.num_blocks) + blocks_per_page - 1
        ) // blocks_per_page
        return num_physical_pages * int(manager.physical_group_page_size)

    def _start_local_mapping_listener(self) -> None:
        from kvcached.tp_ipc_util import start_worker_listener_thread

        tp_rank = int(get_parallel().tp_rank)
        pp_rank = int(get_parallel().pp_rank)
        key = (os.getpid(), pp_rank, tp_rank)
        with _LISTENER_LOCK:
            if key in _LISTENER_KEYS:
                return
            kwargs = {"rank": tp_rank, "pp_rank": pp_rank}
            start_worker_listener_thread(
                **_supported_kwargs(start_worker_listener_thread, kwargs)
            )
            _LISTENER_KEYS.add(key)

    def _configure_physical_layer_partitioning(self) -> None:
        manager = self.kvcached_allocator
        if not getattr(manager, "block_level_layer_groups", False):
            raise RuntimeError(
                "KVCacheD reports layer stacking but its SGLang manager does "
                "not expose block-level layer groups"
            )
        page_allocator = getattr(manager, "page_allocator", None)
        if page_allocator is None or not hasattr(
            page_allocator, "set_should_use_worker_ipc_callback"
        ):
            raise RuntimeError(
                "KVCacheD layer partitioning requires the worker-IPC "
                "PageAllocator callback"
            )

        # KVCacheManager starts post-init on a background thread. Its public
        # ``clear`` method acquires the manager lock before waiting for that
        # thread, while post-init briefly needs the same lock to finish the
        # (currently disabled) null-block reservation. Calling clear too early
        # therefore deadlocks. Wait here, outside the lock, before reconfiguring
        # the manager for SGLang's filtered null page.
        wait_post_init = getattr(manager, "_wait_post_init", None)
        if wait_post_init is None:
            raise RuntimeError(
                "KVCacheD layer partitioning requires a manager post-init fence"
            )
        wait_post_init()

        # Validate the initial range before any physical page can be mapped.
        self.physical_layer_partitioned = True
        self._active_layer_groups()
        from kvcached.tp_ipc_util import register_worker_offset_filter

        register_worker_offset_filter(self._filter_worker_offsets)
        self._start_local_mapping_listener()
        page_allocator.set_should_use_worker_ipc_callback(lambda: True)
        manager.reserve_null_block = True
        # The manager was constructed without a null page so the ownership
        # filter could be installed first.  Reinitialize now; slot/page zero is
        # mapped only for this rank's active layer groups.
        manager.clear()
        logger.info(
            "Elastic PP KVCacheD physical layer partitioning enabled: "
            "active_layers=[%d,%d) groups=%s granularity=%d",
            self.start_layer,
            self.end_layer,
            sorted(self._active_layer_groups()),
            int(manager.layer_group_granularity),
        )

    def attach_logical_allocator(self, allocator) -> None:
        if (
            self._logical_allocator is not None
            and self._logical_allocator is not allocator
        ):
            raise RuntimeError("KVCacheD pool already has a logical SGLang allocator")
        self._logical_allocator = allocator

    def _physical_page_ids_for_slots(self, slot_ids: torch.Tensor) -> set[int]:
        if not self.physical_layer_partitioned:
            return set()
        manager = self.kvcached_allocator
        logical_blocks = {
            int(slot) // self.page_size for slot in slot_ids.detach().cpu().tolist()
        }
        logical_blocks.update(int(item) for item in (manager.null_block or []))
        return {
            int(manager.page_allocator.get_page_id(block_id, manager.block_mem_size))
            for block_id in logical_blocks
        }

    def _offsets_for_layer_groups_and_slots(
        self, layer_ids: Iterable[int], slot_ids: torch.Tensor
    ) -> list[int]:
        if not self.physical_layer_partitioned:
            return []
        manager = self.kvcached_allocator
        granularity = int(manager.layer_group_granularity)
        layer_ids = {int(layer_id) for layer_id in layer_ids}
        groups = {layer_id // granularity for layer_id in layer_ids}
        incomplete = {
            group: sorted(
                layer_id for layer_id in layer_ids if layer_id // granularity == group
            )
            for group in groups
            if len(
                {layer_id for layer_id in layer_ids if layer_id // granularity == group}
            )
            != granularity
        }
        if incomplete:
            raise ValueError(
                "KVCacheD cannot map a partial physical layer group: "
                f"granularity={granularity}, incomplete={incomplete}"
            )
        physical_pages = self._physical_page_ids_for_slots(slot_ids)
        group_span = self._layer_group_total_span()
        map_size = int(manager.physical_group_page_size)
        return sorted(
            group * group_span + page_id * map_size
            for group in groups
            for page_id in physical_pages
        )

    def _map_layer_slots(
        self, *, layer_ids: Iterable[int], slot_ids: torch.Tensor
    ) -> None:
        offsets = self._offsets_for_layer_groups_and_slots(layer_ids, slot_ids)
        if not offsets:
            return
        from kvcached.tp_ipc_util import map_worker_offsets_for_migration

        if not map_worker_offsets_for_migration(offsets, group_id=self._group_id):
            raise RuntimeError(
                "KVCacheD failed to map target elastic PP layer pages: "
                f"offsets={offsets[:16]}, count={len(offsets)}"
            )

    def _unmap_layer_groups(self, layer_ids: Iterable[int]) -> None:
        if not self.physical_layer_partitioned:
            return
        layer_ids = {int(layer_id) for layer_id in layer_ids}
        if not layer_ids:
            return
        manager = self.kvcached_allocator
        granularity = int(manager.layer_group_granularity)
        groups = {layer_id // granularity for layer_id in layer_ids}
        for group in groups:
            expected = set(range(group * granularity, (group + 1) * granularity))
            if not expected.issubset(layer_ids):
                raise ValueError(
                    "KVCacheD cannot unmap a partial physical layer group: "
                    f"group={group}, expected={sorted(expected)}, "
                    f"provided={sorted(layer_ids)}"
                )
        from kvcached.tp_ipc_util import unmap_worker_offsets_for_layer_groups

        group_span = self._layer_group_total_span()
        if not unmap_worker_offsets_for_layer_groups(
            groups, group_span, group_id=self._group_id
        ):
            raise RuntimeError(
                "KVCacheD failed to unmap retired elastic PP layer groups: "
                f"groups={sorted(groups)}"
            )

    def _validate_kvcached_capacity(self) -> None:
        required_rows = self.size + self.page_size
        for kind, buffers in (
            ("K", self._kvcached_global_k_buffer),
            ("V", self._kvcached_global_v_buffer),
        ):
            if buffers and int(buffers[0].shape[0]) < required_rows:
                raise RuntimeError(
                    f"KVCacheD {kind} virtual capacity is smaller than SGLang's "
                    f"slot space: rows={int(buffers[0].shape[0])}, "
                    f"required={required_rows}"
                )

    def _require_elastic_pp_kv_supported(self) -> None:
        if not self._elastic_pp_enabled:
            raise RuntimeError("elastic PP is not enabled for this KVCacheD pool")
        if self.is_quantized_kv_cache or self.store_dtype != self.dtype:
            raise NotImplementedError(
                "elastic PP KVCacheD requires unquantized, unpacked KV storage"
            )
        if self.post_capture_active or self.use_hnd or self.kv_cache_layout != "nhd":
            raise NotImplementedError(
                "elastic PP KVCacheD requires its standard NHD VMM backing"
            )
        if self._elastic_pp_enable_kv_cache_copy:
            raise NotImplementedError(
                "elastic PP KVCacheD does not yet support speculative KV-copy tables"
            )

    def _allocate_elastic_pp_kv_pair(
        self, *, layer_id: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_id is None:
            raise ValueError("KVCacheD target backing requires a global layer id")
        layer_id = int(layer_id)
        if not 0 <= layer_id < self._kvcached_total_num_layers:
            raise IndexError(
                f"KVCacheD layer {layer_id} is outside global backing "
                f"[0, {self._kvcached_total_num_layers})"
            )
        return (
            self._kvcached_global_k_buffer[layer_id],
            self._kvcached_global_v_buffer[layer_id],
        )

    def prepare_elastic_pp_kv_layers(
        self,
        *,
        target: LayerRange,
        generation: int,
        total_num_layers: int,
    ) -> frozenset[int]:
        if self.physical_layer_partitioned:
            self._active_layer_groups(target)
        prepared = super().prepare_elastic_pp_kv_layers(
            target=target,
            generation=generation,
            total_num_layers=total_num_layers,
        )
        # Creating unpublished global tensor views is safe off-path, but
        # mapping new VMM pages performs a whole-device fence. Defer mapping
        # until staged KV rows are applied by the scheduler at its drained PP
        # fence; otherwise a background map can wait on an in-flight NCCL recv.
        return prepared

    def prepare_elastic_pp_kv_fence(self, *, generation: int) -> None:
        """Refresh on-demand VMM ownership before importing target KV rows.

        A KVCacheD physical page can remain mapped in its reserve pool after
        all logical rows on that page are freed.  If this rank later acquires a
        layer group, such a reserved page can be reused without invoking the
        normal map callback, leaving the new layer group unmapped.  Flush those
        stale reserves at the drained fence, then explicitly extend every
        currently in-use page to the gained groups.  Future free pages pass
        through the normal callback after commit and observe the new active
        range through ``_filter_worker_offsets``.
        """

        if not self.physical_layer_partitioned:
            return
        with self._elastic_pp_lock:
            residency = self._elastic_pp_residency
            pending = None if residency is None else residency.pending
            if pending is None or pending.generation != generation:
                raise RuntimeError(
                    f"no KV preparation exists for generation {generation}"
                )
            current = LayerRange(self.start_layer, self.end_layer)
            gained_groups = self._active_layer_groups(
                pending.target
            ) - self._active_layer_groups(current)
        if not gained_groups:
            return

        manager = self.kvcached_allocator
        snapshot_inuse = getattr(manager, "snapshot_inuse_page_ids", None)
        if snapshot_inuse is None:
            raise RuntimeError(
                "KVCacheD elastic ownership refresh requires snapshot_inuse_page_ids"
            )
        reserved_before = int(manager.page_allocator.get_num_reserved_pages())
        started = time.monotonic()
        manager.trim()
        reserved_after = int(manager.page_allocator.get_num_reserved_pages())
        if reserved_after:
            raise RuntimeError(
                "KVCacheD could not flush stale reserved pages before elastic "
                f"ownership expansion: remaining={reserved_after}"
            )
        page_ids = [int(page_id) for page_id in snapshot_inuse()]
        group_span = self._layer_group_total_span()
        map_size = int(manager.physical_group_page_size)
        offsets = [
            group * group_span + page_id * map_size
            for group in sorted(gained_groups)
            for page_id in page_ids
        ]
        if offsets:
            from kvcached.tp_ipc_util import map_worker_offsets_for_migration

            if not map_worker_offsets_for_migration(offsets, group_id=self._group_id):
                raise RuntimeError(
                    "KVCacheD failed to refresh gained elastic PP layer groups: "
                    f"groups={sorted(gained_groups)}, pages={page_ids}"
                )
        logger.info(
            "Elastic PP KVCacheD ownership refresh generation=%d "
            "gained_groups=%s inuse_pages=%d trimmed_reserved_pages=%d "
            "time=%.3fs",
            generation,
            sorted(gained_groups),
            len(page_ids),
            reserved_before,
            time.monotonic() - started,
        )

    def import_elastic_pp_kv_slots(
        self,
        *,
        generation: int,
        layer_id: int,
        slot_ids: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        if self.physical_layer_partitioned:
            granularity = int(self.kvcached_allocator.layer_group_granularity)
            group_start = int(layer_id) // granularity * granularity
            self._map_layer_slots(
                layer_ids=range(group_start, group_start + granularity),
                slot_ids=slot_ids,
            )
        super().import_elastic_pp_kv_slots(
            generation=generation,
            layer_id=layer_id,
            slot_ids=slot_ids,
            cache_k=cache_k,
            cache_v=cache_v,
        )

    def abort_elastic_pp_kv_layers(
        self, *, generation: int
    ) -> tuple[torch.Tensor, ...]:
        pending_ids = set(self._elastic_pp_pending_kv)
        # Keep the base transaction intact until the fallible VMM operation
        # succeeds. A failed unmap can then be retried without losing the exact
        # pending layer set.
        self._unmap_layer_groups(pending_ids)
        return super().abort_elastic_pp_kv_layers(generation=generation)

    def cleanup_elastic_pp_kv_layers(
        self, *, generation: int
    ) -> tuple[torch.Tensor, ...]:
        retired_ids = set(self._elastic_pp_retired_kv)
        # Likewise, do not mark residency cleanup complete until physical
        # reclamation succeeds. The committed target stays usable and cleanup
        # remains safely retryable after a transient unmap failure.
        self._unmap_layer_groups(retired_ids)
        return super().cleanup_elastic_pp_kv_layers(generation=generation)

    def get_kv_size_bytes(self):
        # Report the configured serving budget, not KVCacheD's much larger
        # virtual-address reservation.
        rows = self.size + self.page_size
        k_bytes = (
            self.layer_num * rows * self.head_num * self.head_dim * self.dtype.itemsize
        )
        v_bytes = (
            self.layer_num
            * rows
            * self.head_num
            * self.v_head_dim
            * self.dtype.itemsize
        )
        return k_bytes, v_bytes

    def kvcached_physical_stats(self) -> dict[str, Any]:
        stats = getattr(self.kvcached_allocator, "stats", None)
        result = dict(stats() if stats is not None else {})
        if self.physical_layer_partitioned:
            from kvcached.tp_ipc_util import snapshot_worker_mapped_offsets

            offsets = snapshot_worker_mapped_offsets(self._group_id)
            map_size = int(self.kvcached_allocator.physical_group_page_size)
            allocator_estimate = result.get("mapped_memory_bytes")
            result.update(
                {
                    "allocator_estimated_mapped_memory_bytes": allocator_estimate,
                    "mapped_memory_bytes": len(offsets) * map_size,
                    "mapped_memory_scope": "local_worker_offset_ledger",
                    "local_worker_mapped_offset_count": len(offsets),
                    "local_worker_mapped_layer_groups": sorted(
                        {
                            int(offset) // self._layer_group_total_span()
                            for offset in offsets
                        }
                    ),
                }
            )
        else:
            result["mapped_memory_scope"] = "allocator_estimate"
        result.update(
            {
                "backing_num_layers": self._kvcached_total_num_layers,
                "active_start_layer": self.start_layer,
                "active_end_layer": self.end_layer,
                "physical_layer_partitioned": self.physical_layer_partitioned,
            }
        )
        return result


class _KVCacheDSlotTrackingMixin:
    _elastic_pp_slot_tracker: ElasticKVSlotTracker | None
    _live_page_ids: set[int]

    def _initialize_slot_tracking(self) -> None:
        self._elastic_pp_slot_tracker = None
        self._live_page_ids = set()

    def _check_clear_allowed(self) -> None:
        if self._elastic_pp_slot_tracker is not None:
            raise RuntimeError(
                "cannot clear the KV allocator while an elastic PP snapshot "
                "generation is being tracked"
            )

    @staticmethod
    def _as_int_list(indices: torch.Tensor | Iterable[int]) -> list[int]:
        if isinstance(indices, torch.Tensor):
            return [int(item) for item in indices.detach().cpu().tolist()]
        return [int(item) for item in indices]

    def _record_allocated_pages(self, page_ids: Iterable[int]) -> torch.Tensor:
        pages = sorted({int(page_id) for page_id in page_ids})
        if not pages:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        if pages[0] <= 0:
            raise RuntimeError(
                "KVCacheD returned the reserved null page to SGLang: " f"pages={pages}"
            )
        self._live_page_ids.update(pages)
        page_tensor = torch.tensor(pages, dtype=torch.int64, device=self.device)
        slots = (
            page_tensor[:, None] * self.page_size
            + torch.arange(self.page_size, dtype=torch.int64, device=self.device)
        ).reshape(-1)
        if self._elastic_pp_slot_tracker is not None:
            self._elastic_pp_slot_tracker.record_alloc(slots)
        return slots

    def _record_freed_pages(self, page_ids: Iterable[int]) -> list[int]:
        pages = sorted({int(page_id) for page_id in page_ids})
        unknown = set(pages).difference(self._live_page_ids)
        if unknown:
            raise RuntimeError(
                "KVCacheD allocator was asked to free non-live pages: "
                f"{sorted(unknown)}"
            )
        self._live_page_ids.difference_update(pages)
        return pages

    def elastic_pp_live_slots(self) -> torch.Tensor:
        if not self.is_not_in_free_group or self.free_group:
            raise RuntimeError(
                "elastic PP live-slot snapshots require a closed allocator "
                "free group"
            )
        if not self._live_page_ids:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        pages = torch.tensor(
            sorted(self._live_page_ids), dtype=torch.int64, device=self.device
        )
        return (
            pages[:, None] * self.page_size
            + torch.arange(self.page_size, dtype=torch.int64, device=self.device)
        ).reshape(-1)

    def begin_elastic_pp_slot_tracking(self, *, generation: int) -> torch.Tensor:
        if self._elastic_pp_slot_tracker is not None:
            if self._elastic_pp_slot_tracker.generation != generation:
                raise RuntimeError(
                    "another elastic PP KV slot generation is already tracked: "
                    f"pending={self._elastic_pp_slot_tracker.generation}, "
                    f"requested={generation}"
                )
            return self._elastic_pp_slot_tracker.initial_live_slots
        live_slots = self.elastic_pp_live_slots()
        tracker_size = self.size + self.page_size - 1
        self._elastic_pp_slot_tracker = ElasticKVSlotTracker(
            generation=generation,
            size=tracker_size,
            initial_live_slots=live_slots,
        )
        return self._elastic_pp_slot_tracker.initial_live_slots

    def elastic_pp_delta_slots(self, *, generation: int) -> torch.Tensor:
        tracker = self._elastic_pp_slot_tracker
        if tracker is None or tracker.generation != generation:
            raise RuntimeError(
                f"no elastic PP KV slot tracking exists for generation {generation}"
            )
        return tracker.changed_live_slots(self.elastic_pp_live_slots())

    def finish_elastic_pp_slot_tracking(self, *, generation: int) -> None:
        tracker = self._elastic_pp_slot_tracker
        if tracker is None or tracker.generation != generation:
            raise RuntimeError(
                f"no elastic PP KV slot tracking exists for generation {generation}"
            )
        self._elastic_pp_slot_tracker = None


class KVCacheDTokenToKVPoolAllocator(
    _KVCacheDSlotTrackingMixin, BaseTokenToKVPoolAllocator
):
    """SGLang token allocator backed by KVCacheD's lazy physical pages."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCacheDMHATokenToKVPool,
        need_sort: bool,
    ) -> None:
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        self.kvcached_allocator = kvcache.kvcached_allocator
        self._initialize_slot_tracking()
        self.clear()
        kvcache.attach_logical_allocator(self)

    def clear(self) -> None:
        self._check_clear_allowed()
        self.kvcached_allocator.clear()
        self._live_page_ids.clear()
        self.free_pages = torch.empty(0, dtype=torch.int64, device=self.device)
        self.release_pages = torch.empty(0, dtype=torch.int64, device=self.device)
        self.is_not_in_free_group = True
        self.free_group = []

    def available_size(self):
        return int(self.kvcached_allocator.available_size())

    def alloc(self, need_size: int):
        page_ids = self.kvcached_allocator.alloc(int(need_size))
        if page_ids is None:
            return None
        return self._record_allocated_pages(page_ids)

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if not self.is_not_in_free_group:
            self.free_group.append(self._copy_for_free_group(free_index))
            return
        page_ids = self._record_freed_pages(self._as_int_list(free_index))
        self.kvcached_allocator.free(page_ids)

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )


class KVCacheDPagedTokenToKVPoolAllocator(
    _KVCacheDSlotTrackingMixin, BaseTokenToKVPoolAllocator
):
    """Page-aligned KVCacheD allocator, including prefill/decode kernels."""

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCacheDMHATokenToKVPool,
        need_sort: bool,
    ) -> None:
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.kvcached_allocator = kvcache.kvcached_allocator
        self.num_pages = size // page_size
        self.seen_max_num_extend_tokens_next_power_of_2 = 1
        self._initialize_slot_tracking()
        self.clear()
        kvcache.attach_logical_allocator(self)

    def clear(self) -> None:
        self._check_clear_allowed()
        self.kvcached_allocator.clear()
        self._live_page_ids.clear()
        self.free_pages = torch.empty(0, dtype=torch.int64, device=self.device)
        self.release_pages = torch.empty(0, dtype=torch.int64, device=self.device)
        self.is_not_in_free_group = True
        self.free_group = []

    def available_size(self):
        return int(self.kvcached_allocator.available_size()) * self.page_size

    def _alloc_pages(self, num_pages: int) -> Optional[torch.Tensor]:
        page_ids = self.kvcached_allocator.alloc(int(num_pages))
        if page_ids is None:
            return None
        slots = self._record_allocated_pages(page_ids)
        return slots.reshape(-1, self.page_size)[:, 0] // self.page_size

    def alloc(self, need_size: int):
        if need_size % self.page_size != 0:
            raise ValueError(
                "KVCacheD paged allocation must be page aligned: "
                f"need_size={need_size}, page_size={self.page_size}"
            )
        pages = self._alloc_pages(need_size // self.page_size)
        if pages is None:
            return None
        return (
            pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: Optional[int] = None,
    ):
        from sglang.kernels.ops.memory.allocator import alloc_extend_kernel
        from sglang.srt.utils import get_num_new_pages, next_power_of_2

        self.seen_max_num_extend_tokens_next_power_of_2 = max(
            self.seen_max_num_extend_tokens_next_power_of_2,
            next_power_of_2(extend_num_tokens),
        )
        if num_new_pages is None:
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
        free_pages = self._alloc_pages(num_new_pages)
        if free_pages is None:
            return None
        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )
        bs = len(prefix_lens)
        alloc_extend_kernel[(bs,)](
            prefix_lens,
            seq_lens,
            last_loc,
            free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
            self.seen_max_num_extend_tokens_next_power_of_2,
        )
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        from sglang.kernels.ops.memory.allocator import alloc_decode_kernel
        from sglang.srt.utils import get_num_new_pages, next_power_of_2

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )
        free_pages = self._alloc_pages(num_new_pages)
        if free_pages is None:
            return None
        bs = len(seq_lens)
        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        alloc_decode_kernel[(bs,)](
            seq_lens,
            last_loc,
            free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )
        return out_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if not self.is_not_in_free_group:
            self.free_group.append(self._copy_for_free_group(free_index))
            return
        pages = torch.unique(free_index // self.page_size)
        page_ids = self._record_freed_pages(self._as_int_list(pages))
        self.kvcached_allocator.free(page_ids)
