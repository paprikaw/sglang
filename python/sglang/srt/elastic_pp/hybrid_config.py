"""Conservative launch constraints for transactional Qwen3.5 hybrid PP."""

QWEN35_ELASTIC_ARCHITECTURES = (
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
)


def validate_qwen35_elastic_args(args):
    """Fail before loading weights for cache modes not covered by migration."""
    unsupported = []
    hicache = getattr(args, "enable_hierarchical_cache", False)
    if not args.disable_radix_cache and not hicache:
        unsupported.append(
            "prefix caching without HiCache (pass --disable-radix-cache)"
        )
    if hicache:
        if (getattr(args, "elastic_pp_kv_backend", "native") != "native"
                and not getattr(args, "elastic_pp_kv_capacities", None)):
            unsupported.append("HiCache requires native elastic KV backing")
        if getattr(args, "hicache_storage_backend", None):
            unsupported.append("HiCache L3 storage")
        if getattr(args, "hicache_host_memory_mode", "cache") != "cache":
            unsupported.append("HiCache buffer-only host mode")
        if getattr(args, "hicache_mem_layout", "page_first") != "page_first":
            unsupported.append("hybrid HiCache requires page_first host layout")
        if getattr(args, "hicache_io_backend", "kernel") != "kernel":
            unsupported.append("hybrid HiCache requires kernel I/O")
        if getattr(args, "enable_session_radix_cache", False):
            unsupported.append("session radix cache frontier migration")
    if not args.max_running_requests or args.max_running_requests <= 0:
        unsupported.append("automatic request capacity (set --max-running-requests)")
    for flag in (
        "enable_unified_memory",
        "enable_page_major_kv_layout",
        "enable_int8_mamba_checkpoint",
        "enable_linear_replayssm",
        "enable_linear_replayssm_spec",
        "enable_lora",
    ):
        if getattr(args, flag, False):
            unsupported.append("--" + flag.replace("_", "-"))
    if getattr(args, "speculative_algorithm", None):
        unsupported.append("speculative decoding")
    if getattr(args, "kv_cache_dtype", "auto") not in (
        "auto",
        "bfloat16",
        "float16",
        "bf16",
        "fp16",
    ):
        unsupported.append("quantized KV cache")
    if unsupported:
        raise ValueError(
            "Qwen3.5 elastic PP currently requires ordinary GDN state with "
            "fixed request capacity and unquantized KV; unsupported="
            + ", ".join(unsupported)
        )
