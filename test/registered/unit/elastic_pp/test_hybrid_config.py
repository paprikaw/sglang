from types import SimpleNamespace

import pytest

from sglang.srt.elastic_pp.hybrid_config import validate_qwen35_elastic_args


def args(**changes):
    values = dict(
        disable_radix_cache=False,
        max_running_requests=4,
        enable_hierarchical_cache=True,
        elastic_pp_kv_backend="native",
        hicache_storage_backend=None,
        hicache_host_memory_mode="cache",
        hicache_mem_layout="page_first",
        hicache_io_backend="kernel",
    )
    return SimpleNamespace(**(values | changes))


def test_l2_hicache_with_plain_gdn_and_prefix_cache_is_supported():
    validate_qwen35_elastic_args(args())


@pytest.mark.parametrize(
    "changes",
    [
        {"hicache_storage_backend": "file"},
        {"hicache_host_memory_mode": "buffer_only"},
        {"hicache_mem_layout": "page_first_direct"},
        {"hicache_io_backend": "direct"},
        {"enable_session_radix_cache": True},
        {"enable_linear_replayssm": True},
        {"enable_int8_mamba_checkpoint": True},
        {"speculative_algorithm": "EAGLE"},
        {"elastic_pp_kv_backend": "kvcached"},
        {"max_running_requests": None},
    ],
)
def test_unmigrated_hybrid_state_and_host_modes_remain_rejected(changes):
    with pytest.raises(ValueError):
        validate_qwen35_elastic_args(args(**changes))


def test_no_hicache_path_keeps_existing_prefix_constraint():
    validate_qwen35_elastic_args(
        args(enable_hierarchical_cache=False, disable_radix_cache=True)
    )
    with pytest.raises(ValueError, match="prefix caching without HiCache"):
        validate_qwen35_elastic_args(args(enable_hierarchical_cache=False))
