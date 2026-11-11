"""Capability and layout checks for ownership-aware KVCacheD backing.

Keep the optional dependency lazy. A matching Python wrapper alone is not
enough: the native extension must implement the same physical page geometry.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

LAYER_MAPPING_API_VERSION = 1
_WORKER_APIS = (
    "register_worker_offset_filter",
    "map_worker_offsets_for_migration",
    "unmap_worker_offsets_for_layer_groups",
    "snapshot_worker_mapped_offsets",
    "pin_worker_offsets",
    "unpin_worker_offsets",
)


@dataclass(frozen=True)
class KVCacheDLayerMappingCapabilities:
    missing: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return not self.missing

    def require(self) -> None:
        if self.missing:
            raise RuntimeError(
                "ownership-aware elastic PP requires the pinned KVCacheD "
                "layer-mapping dependency; missing/incompatible: "
                + ", ".join(self.missing)
                + ". See research/pp_elastic/KV_LAYER_MAPPING.md; rebuild the "
                "native extension after applying the dependency patch."
            )


def probe_layer_mapping_capabilities() -> KVCacheDLayerMappingCapabilities:
    missing = []
    try:
        utils = importlib.import_module("kvcached.utils")
        ipc = importlib.import_module("kvcached.tp_ipc_util")
        native = importlib.import_module("kvcached.vmm_ops")
        manager = importlib.import_module("kvcached.kv_cache_manager").KVCacheManager
    except (ImportError, AttributeError) as exc:
        return KVCacheDLayerMappingCapabilities((f"dependency import: {exc}",))
    if (
        getattr(utils, "ELASTIC_PP_LAYER_MAPPING_API_VERSION", None)
        != LAYER_MAPPING_API_VERSION
    ):
        missing.append("Python layer-mapping API v1")
    if (
        getattr(native, "elastic_pp_layer_mapping_abi_version", None)
        != LAYER_MAPPING_API_VERSION
    ):
        missing.append("native layer-mapping ABI v1")
    if not getattr(utils, "LAYER_STACKING", False):
        missing.append("KVCACHED_LAYER_STACKING=true")
    if not getattr(utils, "CONTIGUOUS_LAYOUT", False):
        missing.append("layer-group contiguous reservation")
    if getattr(utils, "PAGE_PREALLOC_ENABLED", True):
        missing.append("KVCACHED_PAGE_PREALLOC_ENABLED=false")
    missing.extend(
        name for name in _WORKER_APIS if not callable(getattr(ipc, name, None))
    )
    missing.extend(
        f"KVCacheManager.{name}"
        for name in ("snapshot_inuse_page_ids", "get_layer_group_total_span")
        if not callable(getattr(manager, name, None))
    )
    return KVCacheDLayerMappingCapabilities(tuple(missing))


def validate_layer_stacking_range(
    *,
    total_layers: int,
    start: int,
    end: int,
    page_bytes: int,
    physical_block_bytes: int,
) -> int:
    """Validate dense attention-pool ordinals before reserving device memory.

    Hybrid callers supply attention ordinals, not model-layer indices. The
    returned group size remains fixed for the reservation's entire lifetime.
    """
    if (
        page_bytes <= 0
        or physical_block_bytes <= 0
        or page_bytes % physical_block_bytes
    ):
        raise ValueError(
            "KVCacheD physical block must be a positive divisor of page size"
        )
    granularity = page_bytes // physical_block_bytes
    if total_layers <= 0 or total_layers % granularity:
        raise ValueError(
            f"KVCacheD attention layer count {total_layers} must be divisible "
            f"by layer-group granularity {granularity}"
        )
    if not 0 <= start <= end <= total_layers:
        raise ValueError("KVCacheD active attention range is outside the backing")
    if start % granularity or end % granularity:
        raise ValueError(
            "KVCacheD attention-pool PP boundaries must be aligned to "
            f"layer-group granularity {granularity}: [{start},{end})"
        )
    return granularity
