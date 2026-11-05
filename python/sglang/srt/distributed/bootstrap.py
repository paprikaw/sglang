import logging
import os
import time
from typing import List, Optional

import msgspec
import torch
import torch.distributed as dist

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import (
    get_default_distributed_backend,
    get_elastic_pp_migration_group,
    get_pp_group,
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    initialize_heterogeneous_model_parallel,
    initialize_model_parallel,
    set_custom_all_reduce,
    set_flashinfer_allreduce_only,
    set_mscclpp_all_reduce,
    set_torch_symm_mem_all_reduce,
)
from sglang.srt.distributed.parallel_state import (
    _tag_groups_for_flashinfer_allreduce_only,
)
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_available_gpu_memory,
    is_hip,
    is_host_cpu_arm64,
    is_npu,
    monkey_patch_p2p_access_check,
)
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.patch_torch import register_sgl_tp_rank

logger = logging.getLogger(__name__)

_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu_arm64 = is_host_cpu_arm64()

# A representative per-peer payload for materializing the PyNCCL P2P
# connections used by TP LM-head all-to-all. The input/output tensors are
# temporary; NCCL owns the transport resources retained after the warmup.
# In dsv4-pro, assume bs per dp is 120 and the vocab_size is 129280.
# Therefore, the chunk size that each peer sends is 120*129280/8=1.849MB.
# The total warmup bytes per peer should be 1.849*2 = 4MB
_TP_ALL_TO_ALL_WARMUP_BYTES_PER_PEER = 4 << 20


class TorchDistributedResult(msgspec.Struct, frozen=True, kw_only=True):
    tp_group: object
    pp_group: object
    attention_tp_group: object
    pre_model_load_memory: float
    elastic_pp_migration_group: object = None


def _pre_model_load_memory_group(
    *, is_draft_worker: bool, pp_size: int, pp_spec_enabled: bool
):
    # Opt-in PP+spec creates a draft only on the last PP stage. A draft
    # world-group probe would therefore match earlier stages' TARGET pool
    # probes instead. Their next int64 token-capacity reduction then meets a
    # float32 free-memory reduction (Gloo: 8 vs 4 bytes). Keep the tail-only
    # probe on its participating TP group; target pool sizing still reduces
    # over every PP stage after target and draft weights are resident.
    if pp_spec_enabled and is_draft_worker and pp_size > 1:
        return get_tp_group()
    return get_world_group()


def init_torch_distributed(
    *,
    server_args: ServerArgs,
    model_config: ModelConfig,
    device: str,
    ps: ParallelState,
    dist_port: int,
    is_draft_worker: bool,
    local_omp_cpuid: Optional[List[int]],
):
    tic = time.perf_counter()
    logger.info("Init torch distributed begin.")

    backend = _resolve_backend(device=device, server_args=server_args)

    before_avail_memory = get_available_gpu_memory(device, ps.gpu_id)
    if not server_args.enable_p2p_check:
        monkey_patch_p2p_access_check()

    dist_init_method = _resolve_dist_init_method(
        server_args=server_args, dist_port=dist_port
    )
    _set_all_reduce_flags(server_args=server_args)

    if not is_draft_worker:
        if device == "cpu":
            _init_cpu_threads_env(
                tp_size=ps.tp_size, tp_rank=ps.tp_rank, local_omp_cpuid=local_omp_cpuid
            )

        # Only initialize the distributed environment on the target model worker.
        _init_parallel_groups(
            backend=backend,
            dist_init_method=dist_init_method,
            server_args=server_args,
            model_config=model_config,
            gpu_id=ps.gpu_id,
            tp_rank=ps.tp_rank,
            tp_size=ps.tp_size,
            pp_rank=ps.pp_rank,
            pp_size=ps.pp_size,
            attn_dp_size=ps.attn_dp_size,
            attn_cp_size=ps.attn_cp_size,
            moe_ep_size=ps.moe_ep_size,
            moe_dp_size=ps.moe_dp_size,
            dcp_size=ps.attn_dcp_size,
        )

        # Pre-warm NCCL/RCCL/HCCL to eliminate cold-start latency in first request
        # Controlled by --pre-warm-nccl flag (default: enabled on AMD GPUs)
        if get_exec().comm.pre_warm_nccl and (
            ps.tp_size > 1 or ps.pp_size > 1 or ps.moe_ep_size > 1
        ):
            _prewarm_nccl(
                tp_size=ps.tp_size, pp_size=ps.pp_size, moe_ep_size=ps.moe_ep_size
            )

        # This is a correctness boundary rather than an optional benchmark
        # warmup: elastic activations and the reverse scheduler output ring
        # intentionally use separate NCCL communication domains.
        if device == "cuda" and get_parallel().enable_elastic_pp and ps.pp_size > 1:
            _prewarm_elastic_pp_data_plane()

        # CUDA graph capture enables the PyNCCL communicator for TP LM-head
        # all-to-all. Exercise that exact send/recv path before measuring
        # pre_model_load_memory so its persistent transport allocations are
        # included in later KV-cache sizing instead of appearing during capture.
        if (
            device == "cuda"
            and get_parallel().enable_tp_lm_head_all_to_all
            and ps.tp_size > 1
        ):
            _prewarm_tp_lm_head_all_to_all()

    memory_group = _pre_model_load_memory_group(
        is_draft_worker=is_draft_worker,
        pp_size=ps.pp_size,
        pp_spec_enabled=envs.SGLANG_ENABLE_PP_SPEC.get(),
    )
    pre_model_load_memory = get_available_gpu_memory(
        device,
        ps.gpu_id,
        distributed=memory_group.world_size > 1,
        cpu_group=memory_group.cpu_group,
    )
    tp_group = get_tp_group()
    pp_group = get_pp_group()
    elastic_pp_migration_group = (
        get_elastic_pp_migration_group() if get_parallel().enable_elastic_pp else None
    )
    attention_tp_group = get_parallel().attn_tp_group

    # Check memory for tensor parallelism
    local_gpu_memory = get_available_gpu_memory(device, ps.gpu_id)
    if ps.tp_size > 1 and not is_draft_worker:
        _check_tp_memory_balance(
            pre_model_load_memory=pre_model_load_memory,
            local_gpu_memory=local_gpu_memory,
        )

    logger.info(
        f"Init torch distributed ends. elapsed={time.perf_counter() - tic:.2f} s, "
        f"mem usage={(before_avail_memory - local_gpu_memory):.2f} GB"
    )
    return TorchDistributedResult(
        tp_group=tp_group,
        pp_group=pp_group,
        attention_tp_group=attention_tp_group,
        pre_model_load_memory=pre_model_load_memory,
        elastic_pp_migration_group=elastic_pp_migration_group,
    )


def _resolve_backend(*, device: str, server_args: ServerArgs) -> str:
    backend = get_default_distributed_backend(device)
    if device == "cuda" and server_args.elastic_ep_backend == "mooncake":
        backend = "mooncake"
    return backend


def _resolve_dist_init_method(*, server_args: ServerArgs, dist_port: int) -> str:
    # Allow external orchestrators (e.g. trainpi) to override the distributed
    # init method.  When set to "env://", torch uses MASTER_ADDR/MASTER_PORT
    # env-vars and an externally-created TCPStore, completely avoiding port
    # conflicts with intra-host collocation.
    dist_init_method_override = envs.SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE.get()
    if dist_init_method_override:
        dist_init_method = dist_init_method_override
    elif server_args.dist_init_addr:
        na = NetworkAddress.parse(server_args.dist_init_addr)
        dist_init_method = na.to_tcp()
    else:
        dist_init_method = NetworkAddress(
            server_args.host or "127.0.0.1", dist_port
        ).to_tcp()
    return dist_init_method


def _set_all_reduce_flags(*, server_args: ServerArgs) -> None:
    set_custom_all_reduce(not get_exec().comm.disable_custom_all_reduce)
    set_mscclpp_all_reduce(server_args.enable_mscclpp)
    set_torch_symm_mem_all_reduce(get_exec().comm.enable_torch_symm_mem)
    set_flashinfer_allreduce_only(
        get_exec().comm.flashinfer_allreduce_fusion_backend is not None
    )


def _init_cpu_threads_env(
    *, tp_size: int, tp_rank: int, local_omp_cpuid: Optional[List[int]]
) -> None:
    if _is_cpu_amx_available or _is_cpu_arm64:
        # Bind OpenMP threads to CPU cores
        torch.ops.sgl_kernel.init_cpu_threads_env(local_omp_cpuid)

        # Set local size to hint SGLang to use shared memory based AllReduce
        os.environ["LOCAL_SIZE"] = str(tp_size)
        torch.ops.sgl_kernel.initialize(tp_size, tp_rank)

    else:
        logger.warning(
            "init_cpu_threads_env and shared memory based AllReduce is disabled, only intel amx backend and arm64 are supported"
        )


def _init_parallel_groups(
    *,
    backend: str,
    dist_init_method: str,
    server_args: ServerArgs,
    model_config: ModelConfig,
    gpu_id: int,
    tp_rank: int,
    tp_size: int,
    pp_rank: int,
    pp_size: int,
    attn_dp_size: int,
    attn_cp_size: int,
    moe_ep_size: int,
    moe_dp_size: int,
    dcp_size: int,
) -> None:
    is_ep_joiner = server_args.is_ep_joiner
    is_scale_joiner = server_args.is_ep_scale_joiner
    rank_offset = server_args.ep_join_rank_offset if is_scale_joiner else 0
    heterogeneous_topology = None
    if server_args.pp_stage_tp_sizes is not None:
        from sglang.srt.distributed.heterogeneous_pp import (
            HeterogeneousPPTopology,
        )

        heterogeneous_topology = HeterogeneousPPTopology.parse(
            server_args.pp_stage_tp_sizes
        )
        expected_tp_size = heterogeneous_topology.stage_tp_sizes[pp_rank]
        if tp_size != expected_tp_size:
            raise RuntimeError(
                f"stage {pp_rank} was launched with TP size {tp_size}, expected "
                f"{expected_tp_size} from --pp-stage-tp-sizes"
            )
        world_size = heterogeneous_topology.world_size
        rank = heterogeneous_topology.global_rank(pp_rank, tp_rank)
        assert server_args.pp_layer_partition is not None
        os.environ["SGLANG_PP_LAYER_PARTITION"] = server_args.pp_layer_partition
    else:
        world_size = (
            rank_offset + tp_size * pp_size if is_scale_joiner else tp_size * pp_size
        )
        rank = rank_offset + tp_size * pp_rank + tp_rank

    init_distributed_environment(
        backend=backend,
        world_size=world_size,
        rank=rank,
        local_rank=gpu_id,
        distributed_init_method=dist_init_method,
        timeout=server_args.dist_timeout,
        moe_a2a_backend=get_exec().moe.moe_a2a_backend,
        recovered_rank=is_ep_joiner,
        max_world_size=server_args.max_ep_size,
    )
    if heterogeneous_topology is not None:
        initialize_heterogeneous_model_parallel(
            stage_tp_sizes=heterogeneous_topology.stage_tp_sizes,
            pipeline_rank=pp_rank,
            enable_symm_mem=get_exec().comm.enable_symm_mem,
        )
    else:
        initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            attention_data_parallel_size=attn_dp_size,
            pipeline_model_parallel_size=pp_size,
            expert_model_parallel_size=moe_ep_size,
            attention_context_model_parallel_size=attn_cp_size,
            moe_data_model_parallel_size=moe_dp_size,
            decode_context_parallel_size=dcp_size,
            duplicate_tp_group=server_args.enable_pdmux,
            duplicate_pp_group=get_parallel().enable_elastic_pp,
            duplicate_attn_cp_group=(
                is_hip()
                and server_args.enable_two_batch_overlap
                and get_parallel().enable_dsa_prefill_context_parallel
            ),
            enable_symm_mem=get_exec().comm.enable_symm_mem,
            recovered_rank=is_ep_joiner,
            rank_offset=rank_offset,
            max_world_size=server_args.max_ep_size,
        )
    if server_args.enable_elastic_pp:
        # Install the initial logical route before model construction. This
        # lets model/runner endpoint checks see [L, 0, ...] as logical PP1
        # while every resident scheduler and communicator stays alive.
        partition = tuple(
            int(item) for item in server_args.elastic_pp_initial_partition.split(",")
        )
        active_rank_indices = [
            rank_index
            for rank_index, layer_count in enumerate(partition)
            if layer_count > 0
        ]
        get_pp_group().set_routing_rank_indices(active_rank_indices)
        logger.info(
            "Installed initial elastic PP route: partition=%s active_ranks=%s",
            partition,
            active_rank_indices,
        )
    _tag_groups_for_flashinfer_allreduce_only()
    initialize_dp_attention(
        server_args=server_args,
        model_config=model_config,
    )
    if is_npu():
        register_sgl_tp_rank(gpu_id)


def _prewarm_nccl(*, tp_size: int, pp_size: int, moe_ep_size: int) -> None:
    warmup_start = time.perf_counter()
    tp_group_handle = get_tp_group().device_group

    # Single warmup all_reduce to initialize NCCL/RCCL/HCCL communicator
    warmup_tensor = torch.zeros(1, device=torch.cuda.current_device())
    dist.all_reduce(warmup_tensor, group=tp_group_handle)
    current_platform.synchronize()

    warmup_elapsed = time.perf_counter() - warmup_start
    logger.info(
        f"NCCL/RCCL/HCCL warmup completed in {warmup_elapsed:.3f}s "
        f"(tp_size={tp_size}, pp_size={pp_size}, ep_size={moe_ep_size})"
    )


def _prewarm_elastic_pp_data_plane() -> None:
    """Materialize both communication domains used by elastic PP.

    Output tokens continue around the resident ProcessGroupNCCL ring so every
    scheduler replica advances request state.  Activations may bypass
    zero-layer ranks and therefore use the full-group PyNCCL communicator.
    Prewarming both domains keeps directed lazy-peer setup out of the first
    request.
    """

    pp_group = get_pp_group()
    if pp_group.world_size <= 1:
        return

    warmup_start = time.perf_counter()
    rank = pp_group.rank_in_group
    payload = torch.empty(1, dtype=torch.uint8, device=pp_group.device)

    # The scheduler output ring uses exactly these directed physical edges.
    # ProcessGroupNCCL caches lazy P2P communicators by directed peer pair, so
    # exercise last->first as well as every forward physical edge.
    for src in range(pp_group.world_size):
        dst = (src + 1) % pp_group.world_size
        work = None
        if rank == src:
            payload.fill_(src + 1)
            work = dist.isend(
                payload,
                dst=pp_group.ranks[dst],
                group=pp_group.device_group,
            )
        elif rank == dst:
            payload.zero_()
            work = dist.irecv(
                payload,
                src=pp_group.ranks[src],
                group=pp_group.device_group,
            )
        if work is not None:
            work.wait()
        current_platform.synchronize()
        pp_group.barrier()

    pynccl_comm = pp_group.pynccl_comm
    if pynccl_comm is None or not pynccl_comm.available:
        raise RuntimeError(
            "Elastic PP activation routing requires an available full-group "
            "PyNCCL communicator"
        )

    # Layer partitions preserve rank order, so every possible direct
    # activation edge has src < dst.  These probes reuse one full
    # communicator; they do not create O(PP^2) persistent process groups.
    with pynccl_comm.change_state(enable=True):
        for src in range(pp_group.world_size):
            for dst in range(src + 1, pp_group.world_size):
                if rank == src:
                    payload.fill_(src + 1)
                    pynccl_comm.send(payload, dst)
                elif rank == dst:
                    payload.zero_()
                    pynccl_comm.recv(payload, src)
                current_platform.synchronize()
                pp_group.barrier()

    del payload
    current_platform.empty_cache()
    warmup_elapsed = time.perf_counter() - warmup_start
    logger.info(
        "Elastic PP data-plane warmup completed in %.3fs "
        "(resident_pp_size=%d, output_edges=%d, activation_edges=%d)",
        warmup_elapsed,
        pp_group.world_size,
        pp_group.world_size,
        pp_group.world_size * (pp_group.world_size - 1) // 2,
    )


def _prewarm_tp_lm_head_all_to_all() -> None:
    """Materialize PyNCCL P2P resources before model-memory accounting."""
    warmup_start = time.perf_counter()
    tp_group = get_tp_group()
    pynccl_comm = tp_group.pynccl_comm
    if pynccl_comm is None or not pynccl_comm.available:
        raise RuntimeError(
            "--enable-tp-lm-head-all-to-all requires an available PyNCCL "
            "communicator for CUDA graph capture."
        )

    numel = tp_group.world_size * _TP_ALL_TO_ALL_WARMUP_BYTES_PER_PEER
    warmup_input = torch.empty(numel, dtype=torch.uint8, device=tp_group.device)
    warmup_output = torch.empty_like(warmup_input)

    # PyNCCL is disabled outside graph-capture contexts by default. Enable it
    # explicitly so eager startup does not fall back to ProcessGroupNCCL and
    # miss the P2P resources required by the captured all-to-all.
    with pynccl_comm.change_state(enable=True):
        pynccl_comm.all_to_all_single(warmup_output, warmup_input)
    current_platform.synchronize()

    del warmup_input, warmup_output
    current_platform.empty_cache()
    warmup_elapsed = time.perf_counter() - warmup_start
    logger.info(
        "TP LM-head PyNCCL all-to-all warmup completed in %.3fs "
        "(tp_size=%d, bytes_per_peer=%d)",
        warmup_elapsed,
        tp_group.world_size,
        _TP_ALL_TO_ALL_WARMUP_BYTES_PER_PEER,
    )


def _check_tp_memory_balance(
    *, pre_model_load_memory: float, local_gpu_memory: float
) -> None:
    if pre_model_load_memory < local_gpu_memory * 0.9:
        msg = "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
        msg += (
            f"{pre_model_load_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
        )
        if envs.SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK.get():
            raise RuntimeError(msg)
        else:
            logger.warning(msg)
