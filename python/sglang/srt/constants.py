# GPU Memory Types
GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
GPU_MEMORY_TYPE_WEIGHTS = "weights"
GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"
# CPU-resident HiCache buffers are not managed by torch-memory-saver, but they
# must participate in the same idle-server lifecycle when a model's GPU weights
# are parked in CPU RAM.  Keep this out of GPU_MEMORY_ALL_TYPES so existing
# callers that omit tags retain their historical GPU-only behavior.
GPU_MEMORY_TYPE_HICACHE_HOST = "hicache_host"

GPU_MEMORY_ALL_TYPES = [
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
]

HEALTH_CHECK_RID_PREFIX = "HEALTH_CHECK"

GIB_BYTES = 1073741824  # 1024**3
