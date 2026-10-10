#include "configs.cuh"
#include "exception.cuh"
#define DEEPEP_LL_CPU_POST_BATCH 8
#include "ibgda_device.cuh"
#include "launch.cuh"

namespace deep_ep {

namespace internode_ll {

// Internal SM80-only launch phase. No public ABI or SM90 behavior changes.
constexpr int kRDMAFirstPayloadPhase = 4;

// A preceding send kernel has exited, so no GPU writer is still reserving or
// filling these QPs. Flush every ready tail (including batches shorter than
// eight), then poll precisely that completion index. The existing quiet helper
// can see only the CPU proxy's already posted index, which is insufficient here.
__global__ __launch_bounds__(32, 1) void drain_rdma_before_pcie(int rank, int num_ranks) {
    const auto state = ibgda_get_state();
    const auto qps_per_rank = state->num_rc_per_pe * state->num_devices_initialized;
    for (int i = threadIdx.x; i < qps_per_rank * (num_ranks - 1); i += blockDim.x) {
        const auto dst_rank = (rank + 1 + i / qps_per_rank) % num_ranks;
        const auto qp_id = i % qps_per_rank;
        auto qp = ibgda_get_rc(dst_rank, qp_id);
        const auto ready = ld_na_relaxed(&qp->mvars.tx_wq.ready_head);
        if (state->use_async_postsend)
            ibgda_proxy_post_send(qp, ready);
        else
            ibgda_post_send(qp, ready);
        ibgda_poll_cq(qp->tx_wq.cq, ready);
    }
    __syncthreads();
    // All ranks must drain before any rank starts its local PCIe writes;
    // per-rank completion alone still permits another rank's NIC/PCIe overlap.
    nvshmemx_barrier_all_block();
}

void launch_drain_rdma_before_pcie(int rank, int num_ranks, cudaStream_t stream) {
    SETUP_LAUNCH_CONFIG(1, 32, stream);
    LAUNCH_KERNEL(&cfg, drain_rdma_before_pcie, rank, num_ranks);
}

__forceinline__ __device__ bool is_cross_pair_pcie(uint64_t ptr, int rank, int dst_rank,
                                                bool nvlink_pair_only) {
    return dst_rank != rank and dst_rank != (rank ^ 1) and
           nvshmemi_get_p2p_ptr(ptr, rank, dst_rank, nvlink_pair_only) != 0;
}

template <bool use_warp_sync = false>
__forceinline__ __device__ bool is_rank_masked(int* mask_buffer_ptr, int rank) {
    if (mask_buffer_ptr == nullptr) {
        return false;
    }
    if constexpr (use_warp_sync) {
        return __shfl_sync(0xffffffff, ld_acquire_global(mask_buffer_ptr + rank), 0) != 0;
    } else {
        return ld_acquire_global(mask_buffer_ptr + rank) != 0;
    }
}

// Experimental SM80 transport split: payload keeps its configured P2P
// eligibility, but cross-pair completion/control messages use IBGDA.
// Self and rank^1 NVLink notifications retain their original fast path.
__forceinline__ __device__ uint64_t control_p2p_ptr(const uint64_t& ptr, const int& rank,
                                                  const int& dst_rank, bool nvlink_pair_only) {
#ifdef DISABLE_SM90_FEATURES
    return nvshmemi_get_p2p_ptr(ptr, rank, dst_rank, true);
#else
    return nvshmemi_get_p2p_ptr(ptr, rank, dst_rank, nvlink_pair_only);
#endif
}


// Experimental BF16 EP3/EP4/EP8 backend. Routing/slot allocation stay in the original
// kernels. Only the late cross-pair writes become local dynamically packed
// frames. Neither routes nor transfer lengths are provided by the benchmark.
struct CEDevice {
    uint64_t send, recv, peers[8];
    int* counts;
    int* notifications;
    uint64_t frame_bytes, ready_offset, ack_offset;
    int capacity, rank, world, local_experts, token_capacity;
    int dispatch_dedup;
};
__device__ __constant__ CEDevice ce_device;
static CEDevice ce_host{};
static int* ce_host_counts = nullptr;
static bool ce_configured = false;
static int ce_epoch = 0;
// calls, memcpy calls, wire bytes, payload copies, dispatch calls, combine calls
static uint64_t ce_metrics[6]{};
static bool ce_matched_peer_order=false;
static bool ce_skip_local_drain=false;

// Every payload peer must be mapped locally. No RDMA data is issued in
// the early phase, so draining payload QPs and a global barrier are redundant.
// Existing completion/ack and per-expert notification ordering remain intact.
extern "C" __attribute__((visibility("default"))) int deepep_ce_set_skip_local_drain(int enabled) {
    if (not ce_configured or ce_host.world!=4 or (enabled!=0 and enabled!=1))
        return static_cast<int>(cudaErrorInvalidValue);
    for (int i=0; i<ce_host.world; ++i) if (ce_host.peers[i]==0)
        return static_cast<int>(cudaErrorInvalidValue);
    auto status=cudaDeviceSynchronize();
    if (status!=cudaSuccess) return static_cast<int>(status);
    ce_skip_local_drain=enabled;
    return 0;
}

// Pairwise matching avoids both source GPUs targeting the same PCIe receiver.
// Set only between completed operations; payload and flags are unchanged.
extern "C" __attribute__((visibility("default"))) int deepep_ce_set_matched_peer_order(int enabled) {
    if (not ce_configured or ce_host.world!=4 or (enabled!=0 and enabled!=1))
        return static_cast<int>(cudaErrorInvalidValue);
    auto status=cudaDeviceSynchronize();
    if (status!=cudaSuccess) return static_cast<int>(status);
    ce_matched_peer_order=enabled;
    return 0;
}

extern "C" __attribute__((visibility("default"))) int deepep_ce_configure_v2(
    uint64_t send, uint64_t recv, const uint64_t* peers, int* counts,
    int* notifications, int* host_counts, int capacity, int rank, int world, int token_capacity) {
    if (ce_configured or (world != 3 and world != 4 and world != 8) or
        token_capacity <= 0 or token_capacity > 2048 or
        capacity <= 0 or capacity % token_capacity != 0 or capacity / token_capacity > 256 or rank < 0 or rank >= world)
        return static_cast<int>(cudaErrorInvalidValue);
    ce_host.send=send; ce_host.recv=recv; ce_host.counts=counts;
    ce_host.notifications=notifications; ce_host.capacity=capacity; ce_host.rank=rank;
    ce_host.world=world; ce_host.local_experts=capacity/token_capacity;
    ce_host.token_capacity=token_capacity;
    ce_host.frame_bytes=16ull + capacity * (16ull + 16 + 5120 * 2);
    ce_host.ready_offset=ce_host.frame_bytes * world;
    ce_host.ack_offset=ce_host.ready_offset + world * 128;
    for (int i=0; i<world; ++i) ce_host.peers[i]=peers[i];
    ce_host_counts=host_counts;
    auto status=cudaMemcpyToSymbol(ce_device, &ce_host, sizeof(ce_host));
    ce_configured=status==cudaSuccess;
    return static_cast<int>(status);
}
extern "C" __attribute__((visibility("default"))) int deepep_ce_snapshot(uint64_t* out) {
    if (not ce_configured) return -1;
    for (int i=0; i<6; ++i) out[i]=ce_metrics[i];
    return ce_epoch;
}

// Experimental runtime switch; caller changes it only between completed pairs.
// Kept off by default. The initial validation scope is BF16 EP4/top-k <= 8.
extern "C" __attribute__((visibility("default"))) int deepep_ce_set_dispatch_dedup(int enabled) {
    if (not ce_configured or ce_host.world != 4 or (enabled != 0 and enabled != 1))
        return static_cast<int>(cudaErrorInvalidValue);
    auto status=cudaDeviceSynchronize();
    if (status!=cudaSuccess) return static_cast<int>(status);
    ce_host.dispatch_dedup=enabled;
    return static_cast<int>(cudaMemcpyToSymbol(ce_device,&ce_host,sizeof(ce_host)));
}

// Header (16), eight destination offsets (64), then one source-token payload.
// Every route still gets its original expert slot. Only PCIe bytes are shared.
constexpr int kCEDedupHeaderBytes=80;
__forceinline__ __device__ uint64_t ce_allocate_dedup(int peer, int bytes) {
    int slot=get_lane_id()==0 ? atomicAdd(ce_device.counts+peer,1) : 0;
    slot=__shfl_sync(0xffffffff,slot,0);
    EP_DEVICE_ASSERT(slot<ce_device.token_capacity and bytes%16==0);
    auto packet=ce_device.send+peer*ce_device.frame_bytes+16ull+slot*(bytes+kCEDedupHeaderBytes);
    return packet;
}

__forceinline__ __device__ uint64_t ce_allocate(int peer, uint64_t offset, int bytes) {
    int slot=get_lane_id()==0 ? atomicAdd(ce_device.counts+peer,1) : 0;
    slot=__shfl_sync(0xffffffff,slot,0);
    EP_DEVICE_ASSERT(slot < ce_device.capacity and bytes % 16 == 0);
    auto packet=ce_device.send+peer*ce_device.frame_bytes+16ull+slot*(bytes+16ull);
    if (get_lane_id()==0) {
        *reinterpret_cast<uint64_t*>(packet)=offset;
        reinterpret_cast<int*>(packet)[2]=bytes;
        reinterpret_cast<int*>(packet)[3]=0;
    }
    return packet+16;
}

__global__ __launch_bounds__(32,1) void ce_finalize_frames(int payload_bytes, bool dedup) {
    const int peer=threadIdx.x;
    if (peer<ce_device.world) {
        auto header=reinterpret_cast<int4*>(ce_device.send+peer*ce_device.frame_bytes);
        *header=make_int4(ce_device.counts[peer],payload_bytes+(dedup ? kCEDedupHeaderBytes : 16),dedup,0);
    }
}

template <bool kAck>
__global__ __launch_bounds__(32,1) void ce_exchange_completion(int epoch) {
    int peer=threadIdx.x;
    const bool eligible=peer<ce_device.world and ce_device.peers[peer]!=0 and
                        peer!=ce_device.rank and peer!=(ce_device.rank^1);
    const auto flags=kAck ? ce_device.ack_offset : ce_device.ready_offset;
    if (eligible)
        st_release_sys_global(reinterpret_cast<int*>(ce_device.peers[peer]+flags+ce_device.rank*128),epoch);
    __syncwarp();
    if (eligible) {
        auto flag=reinterpret_cast<int*>(ce_device.recv+flags+peer*128);
        const auto begin=clock64();
        while (ld_acquire_sys_global(flag)!=epoch) {
            if (clock64()-begin>10000000000ull) {
                printf("CE_COMPLETION_TIMEOUT rank=%d peer=%d epoch=%d ack=%d\n",ce_device.rank,peer,epoch,kAck);
                trap();
            }
        }
    }
}

__global__ __launch_bounds__(1024,1) void ce_unpack(void* output, uint64_t output_bytes, bool dedup) {
    const int warp=blockIdx.x*32+threadIdx.x/32;
    for (int peer=0; peer<ce_device.world; ++peer) {
        if (ce_device.peers[peer]==0 or peer==ce_device.rank or peer==(ce_device.rank^1)) continue;
        const auto frame=ce_device.recv+peer*ce_device.frame_bytes;
        const auto header=*reinterpret_cast<const int4*>(frame);
        EP_DEVICE_ASSERT(header.x>=0 and header.x<=ce_device.capacity);
        for (int slot=warp; slot<header.x; slot+=gridDim.x*32) {
            const auto packet=frame+16ull+slot*static_cast<uint64_t>(header.y);
            EP_DEVICE_ASSERT(header.z==static_cast<int>(dedup));
            if (dedup) {
                const auto metadata=*reinterpret_cast<const int4*>(packet);
                const int count=metadata.x, bytes=metadata.y;
                EP_DEVICE_ASSERT(count>0 and count<=8 and bytes==5120*2+16 and
                                 header.y==bytes+kCEDedupHeaderBytes and header.x<=ce_device.token_capacity);
                const auto* offsets=reinterpret_cast<const uint64_t*>(packet+16);
                const auto* src=reinterpret_cast<const int4*>(packet+kCEDedupHeaderBytes);
                for (int i=0; i<count; ++i) {
                    const auto offset=offsets[i];
                    EP_DEVICE_ASSERT(offset%16==0 and offset+bytes<=output_bytes);
                    auto* dst=reinterpret_cast<int4*>(static_cast<uint8_t*>(output)+offset);
                    UNROLLED_WARP_COPY(4,get_lane_id(),bytes/16,dst,src,ld_nc_global,st_na_global);
                }
            } else {
                const auto offset=*reinterpret_cast<const uint64_t*>(packet);
                const auto bytes=reinterpret_cast<const int*>(packet)[2];
                EP_DEVICE_ASSERT(bytes>0 and bytes%16==0 and header.y==bytes+16 and offset+bytes<=output_bytes);
                const auto* src=reinterpret_cast<const int4*>(packet+16);
                auto* dst=reinterpret_cast<int4*>(static_cast<uint8_t*>(output)+offset);
                UNROLLED_WARP_COPY(4,get_lane_id(),bytes/16,dst,src,ld_nc_global,st_na_global);
            }
        }
    }
}

__global__ __launch_bounds__(32,1) void ce_notify(bool dispatch_mode, int* flags, int* mask) {
    const int nlocal=ce_device.local_experts;
    // One warp retains the original notification SM budget for any expert count.
    for (int i=threadIdx.x; i<nlocal*ce_device.world; i+=blockDim.x) {
    const int dst=i/nlocal, local_expert=i%nlocal, rank=ce_device.rank;
    const auto ptr=reinterpret_cast<uint64_t>(flags+(dispatch_mode ? local_expert*ce_device.world+rank : rank*nlocal+local_expert));
    const auto p2p=control_p2p_ptr(ptr,rank,dst,false);
    const int value=dispatch_mode ? ce_device.notifications[i] : 1;
    if (not is_rank_masked(mask,dst)) {
        if (p2p==0) nvshmemi_ibgda_rma_p(reinterpret_cast<int*>(ptr),value,dst,local_expert);
        else st_release_sys_global(reinterpret_cast<int*>(p2p),value);
    }
    }
}

void ce_begin(int rank, int world, int hidden, int experts, int capacity, cudaStream_t stream) {
    EP_HOST_ASSERT(ce_configured and rank==ce_host.rank and world==ce_host.world and hidden==5120 and experts==ce_host.local_experts*ce_host.world and capacity==ce_host.token_capacity);
    cudaStreamCaptureStatus capture;
    EP_HOST_ASSERT(cudaStreamIsCapturing(stream,&capture)==cudaSuccess and capture==cudaStreamCaptureStatusNone);
    EP_HOST_ASSERT(cudaMemsetAsync(ce_host.counts,0,ce_host.world*sizeof(int),stream)==cudaSuccess);
}

void ce_finish(bool dispatch_mode, void* recv, int* flags, int* mask, cudaStream_t stream) {
    const int payload_bytes=5120*2+(dispatch_mode ? 16 : 0);
    const uint64_t output_bytes=dispatch_mode ? static_cast<uint64_t>(ce_host.local_experts)*ce_host.world*ce_host.token_capacity*payload_bytes : static_cast<uint64_t>(ce_host.local_experts)*ce_host.world*ce_host.token_capacity*(5120*2+160);
    const bool dedup=dispatch_mode and ce_host.dispatch_dedup;
    ce_finalize_frames<<<1,32,0,stream>>>(payload_bytes,dedup);
    // Dynamic GPU counts become CPU memcpy sizes. This synchronization is
    // deliberately included in the real dispatch/combine timing.
    EP_HOST_ASSERT(cudaMemcpyAsync(ce_host_counts,ce_host.counts,ce_host.world*sizeof(int),cudaMemcpyDeviceToHost,stream)==cudaSuccess);
    EP_HOST_ASSERT(cudaStreamSynchronize(stream)==cudaSuccess);
    for (int index=0; index<ce_host.world; ++index) {
        const int peer=ce_matched_peer_order ? (ce_host.rank ^ index) : index;
        const bool eligible=ce_host.peers[peer]!=0 and peer!=ce_host.rank and peer!=(ce_host.rank^1);
        const int count=ce_host_counts[peer];
        EP_HOST_ASSERT(count>=0 and count<=ce_host.capacity and (eligible or count==0));
        if (not eligible) continue;
        const size_t bytes=16ull+static_cast<size_t>(count)*(payload_bytes+(dedup ? kCEDedupHeaderBytes : 16));
        EP_HOST_ASSERT(bytes<=ce_host.frame_bytes);
        EP_HOST_ASSERT(cudaMemcpyAsync(reinterpret_cast<void*>(ce_host.peers[peer]+ce_host.rank*ce_host.frame_bytes),
            reinterpret_cast<const void*>(ce_host.send+peer*ce_host.frame_bytes),bytes,cudaMemcpyDeviceToDevice,stream)==cudaSuccess);
        ++ce_metrics[1]; ce_metrics[2]+=bytes; ce_metrics[3]+=count;
    }
    const int epoch=++ce_epoch;
    ce_exchange_completion<false><<<1,32,0,stream>>>(epoch);
    ce_unpack<<<24,1024,0,stream>>>(recv,output_bytes,dedup);
    // Ack only after the target has unpacked: scratch cannot be overwritten by
    // a fast peer's next dispatch/combine. All flag traffic and waits are timed.
    ce_exchange_completion<true><<<1,32,0,stream>>>(epoch);
    ce_notify<<<1,32,0,stream>>>(dispatch_mode,flags,mask);
    EP_HOST_ASSERT(cudaGetLastError()==cudaSuccess);
    ++ce_metrics[0]; ++ce_metrics[dispatch_mode ? 4 : 5];
}

template <int kNumThreads>
__forceinline__ __device__ void barrier(int thread_id, int rank, int num_ranks, int* mask_buffer_ptr, int* sync_buffer_ptr,
                                      bool nvlink_pair_only) {
    EP_DEVICE_ASSERT(kNumThreads >= num_ranks);

    // Quiet all QPs
    auto qps_per_rank = ibgda_get_state()->num_rc_per_pe * ibgda_get_state()->num_devices_initialized;

    for (int i = thread_id; i < qps_per_rank * (num_ranks - 1); i += kNumThreads) {
        auto dst_rank = (rank + 1 + i / qps_per_rank) % num_ranks;
        auto qp_id = i % qps_per_rank;
        nvshmemi_ibgda_quiet(dst_rank, qp_id);
    }

    // Update local counter
    if (thread_id == 0)
        atomicAdd(sync_buffer_ptr + rank, -1);
    __syncthreads();

    int cnt = sync_buffer_ptr[rank];
    // Update remote counter and wait for local counter to be updated
    if (thread_id < num_ranks && thread_id != rank) {
        const auto dst_rank = thread_id;
        const auto dst_ptr = reinterpret_cast<uint64_t>(sync_buffer_ptr + rank);
        const auto dst_p2p_ptr = control_p2p_ptr(dst_ptr, rank, dst_rank, nvlink_pair_only);

        if (not is_rank_masked(mask_buffer_ptr, dst_rank)) {
            if (dst_p2p_ptr == 0) {
                nvshmemi_ibgda_rma_p(reinterpret_cast<int*>(dst_ptr), cnt, dst_rank, 0);
            } else {
                st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), cnt);
            }

            auto start_time = clock64();
            uint64_t wait_recv_cost = 0;
            while (ld_acquire_sys_global(sync_buffer_ptr + dst_rank) != cnt            // remote is not ready
                   && (wait_recv_cost = clock64() - start_time) <= NUM_TIMEOUT_CYCLES  // not timeout
            )
                ;
            // Mask rank if timeout
            if (wait_recv_cost > NUM_TIMEOUT_CYCLES) {
                printf("Warning: DeepEP timeout for barrier, rank %d, dst_rank %d\n", rank, dst_rank);
                if (mask_buffer_ptr == nullptr)
                    trap();
                atomicExch(mask_buffer_ptr + dst_rank, 1);
            }
        }
    }
    __syncthreads();
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void clean_low_latency_buffer(int* clean_0,
                                                                           int num_clean_int_0,
                                                                           int* clean_1,
                                                                           int num_clean_int_1,
                                                                           int rank,
                                                                           int num_ranks,
                                                                           int* mask_buffer_ptr,
                                                                           int* sync_buffer_ptr,
                                                                           bool nvlink_pair_only) {
    auto thread_id = static_cast<int>(threadIdx.x);

    // Barrier before cleaning (in case of unfinished chunked EP)
    if (sync_buffer_ptr == nullptr)
        nvshmemx_barrier_all_block();
    else
        barrier<kNumThreads>(thread_id, rank, num_ranks, mask_buffer_ptr, sync_buffer_ptr, nvlink_pair_only);

    // Clean
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_0; i += kNumThreads)
        clean_0[i] = 0;
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_1; i += kNumThreads)
        clean_1[i] = 0;

    // Barrier after cleaning (make sure the low-latency mode works fine)
    if (sync_buffer_ptr == nullptr)
        nvshmemx_barrier_all_block();
    else
        barrier<kNumThreads>(thread_id, rank, num_ranks, mask_buffer_ptr, sync_buffer_ptr, nvlink_pair_only);
}

void clean_low_latency_buffer(int* clean_0,
                              int num_clean_int_0,
                              int* clean_1,
                              int num_clean_int_1,
                              int rank,
                              int num_ranks,
                              int* mask_buffer_ptr,
                              int* sync_buffer_ptr,
                              cudaStream_t stream,
                              bool nvlink_pair_only) {
    constexpr int kNumThreads = 256;

    SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);

    LAUNCH_KERNEL(&cfg,
                  clean_low_latency_buffer<kNumThreads>,
                  clean_0,
                  num_clean_int_0,
                  clean_1,
                  num_clean_int_1,
                  rank,
                  num_ranks,
                  mask_buffer_ptr,
                  sync_buffer_ptr,
                  nvlink_pair_only);
}

template <bool kUseFP8, bool kUseUE8M0, int kHidden, bool kUseCE = false>
__global__ __launch_bounds__(1024, 1) void dispatch(void* packed_recv_x,
                                                    void* packed_recv_x_scales,
                                                    int* packed_recv_src_info,
                                                    int64_t* packed_recv_layout_range,
                                                    int* packed_recv_count,
                                                    int* mask_buffer_ptr,
                                                    int* cumulative_local_expert_recv_stats,
                                                    int64_t* dispatch_wait_recv_cost_stats,
                                                    void* rdma_recv_x,
                                                    int* rdma_recv_count,
                                                    void* rdma_x,
                                                    const void* x,
                                                    const topk_idx_t* topk_idx,
                                                    int* atomic_counter_per_expert,
                                                    int* atomic_finish_counter_per_expert,
                                                    int* next_clean,
                                                    int num_next_clean_int,
                                                    int num_tokens,
                                                    int num_max_dispatch_tokens_per_rank,
                                                    int num_topk,
                                                    int num_experts,
                                                    int rank,
                                                    int num_ranks,
                                                    int num_warp_groups,
                                                    int num_warps_per_group,
                                                    bool round_scale,
                                                    int phases,
                                                    bool nvlink_pair_only) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;
#ifdef DISABLE_SM90_FEATURES
    const bool split_rdma_first = not nvlink_pair_only;
    const bool early_rdma_payload_only = (phases & kRDMAFirstPayloadPhase) != 0;
#else
    constexpr bool split_rdma_first = false, early_rdma_payload_only = false;
#endif

    // May extract UE8M0 from the scales
    using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
    using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;
    EP_STATIC_ASSERT(sizeof(packed_t) % sizeof(scale_t) == 0, "Invalid vector length");

    // FP8 staffs
    constexpr int kNumPerChannels = 128;
    const int num_scales = kHidden / kNumPerChannels;
    const size_t hidden_bytes = kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
    const size_t hidden_int4 = hidden_bytes / sizeof(int4);

    // Message package: index at source (int), 3 reserved int fields, hidden data, FP8 scales
    // NOTES: currently we have 3 reserved int fields for future use
    using vec_t = std::conditional_t<kUseFP8, int2, int4>;
    const size_t num_bytes_per_msg = sizeof(int4) + (kUseFP8 ? (kHidden + num_scales * sizeof(float)) : (kHidden * sizeof(nv_bfloat16)));
    const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
    EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);

    // Expert counts
    constexpr int kNumMaxWarpGroups = 32;
    __shared__ int shared_num_tokens_sent_per_expert[kNumMaxWarpGroups];

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_DISPATCH_RECV;

    // There are 2 kinds of warps in this part:
    // 1. The first-kind warps for FP8 cast and sending top-k tokens
    // 2. The last warp for reading `topk_idx` and count for per-expert information
    if (warp_id < num_warps - 1) {
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
        EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerRead) == 0, "Invalid hidden");
        EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0, "Invalid vectorization");
        const auto num_threads = (num_warps - 1) * 32;
        const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;

        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            const auto x_int4 = static_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
            const auto rdma_x_src_idx = reinterpret_cast<int*>(static_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
            const auto rdma_x_vec = reinterpret_cast<vec_t*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
            const auto rdma_x_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

            // Overlap top-k index read and source token index writes
            auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;
            // Preserve RDMA source storage while the NIC drains the early
            // submissions. The late PCIe stage only reads the prepared payload.
            if (not split_rdma_first or early_rdma_payload_only) {
                thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;

                // FP8 cast
                EP_STATIC_ASSERT(hidden_bf16_int4 % 32 == 0, "Must use the full warp to reduce");
                #pragma unroll
                for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
                    // Read
                    auto int4_value = __ldg(x_int4 + i);

                    if constexpr (kUseFP8) {
                        // Calculate local amax
                        auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
                        float fp32_values[kNumElemsPerRead];
                        float amax = kFP8Margin, scale, scale_inv;
                        #pragma unroll
                        for (int j = 0; j < kNumElemsPerRead; ++j) {
                            fp32_values[j] = static_cast<float>(bf16_values[j]);
                            amax = fmaxf(amax, fabsf(fp32_values[j]));
                        }

                        // Reduce amax and scale
                        EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");
                        amax = warp_reduce_max<16>(amax);
                        calculate_fp8_scales(amax, scale, scale_inv, round_scale);
                        if (lane_id == 0 or lane_id == 16)
                            rdma_x_scales[i * kNumElemsPerRead / 128] = scale_inv;

                        // Cast into send buffer
                        vec_t int2_value;
                        auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
                        #pragma unroll
                        for (int j = 0; j < kNumElemsPerRead; j += 2) {
                            float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
                            fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
                        }
                        rdma_x_vec[i] = int2_value;
                    } else {
                        // Reinterpret-cast is for C++14 compatibility
                        rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
                    }
                }
            }
            asm volatile("bar.sync 1, %0;" ::"r"(num_threads));

            // A single leader warp per (token, destination GPU) allocates all
            // expert slots, packs their offsets, and copies the token once.
            // No cross-warp spinlock, extra kernel, or larger launch geometry.
            if constexpr (kUseCE) {
                if (ce_device.dispatch_dedup) {
                    EP_DEVICE_ASSERT(not kUseFP8 and num_topk<=8 and not early_rdma_payload_only);
                    if (dst_expert_idx>=0) {
                        const int dst_rank=dst_expert_idx/num_local_experts;
                        bool leader=true;
                        for (int i=0; i<warp_id; ++i) {
                            const int expert=static_cast<int>(__ldg(topk_idx+token_idx*num_topk+i));
                            if (expert>=0 and expert/num_local_experts==dst_rank) leader=false;
                        }
                        if (leader and is_cross_pair_pcie(reinterpret_cast<uint64_t>(rdma_recv_x),
                                                         rank,dst_rank,nvlink_pair_only)) {
                            const int expert=lane_id<num_topk ?
                                static_cast<int>(__ldg(topk_idx+token_idx*num_topk+lane_id)) : -1;
                            const bool selected=expert>=0 and expert/num_local_experts==dst_rank;
                            const unsigned selected_mask=__ballot_sync(0xffffffff,selected);
                            const int destinations=__popc(selected_mask);
                            const auto packet=ce_allocate_dedup(dst_rank,num_bytes_per_msg);
                            if (lane_id==0)
                                *reinterpret_cast<int4*>(packet)=make_int4(destinations,num_bytes_per_msg,0,0);
                            if (lane_id<8) reinterpret_cast<uint64_t*>(packet+16)[lane_id]=0;
                            __syncwarp();
                            if (selected) {
                                const int slot=atomicAdd(atomic_counter_per_expert+expert,1);
                                EP_DEVICE_ASSERT(slot<num_max_dispatch_tokens_per_rank);
                                const uint64_t offset=(static_cast<uint64_t>(expert%num_local_experts)*num_ranks*
                                    num_max_dispatch_tokens_per_rank+rank*num_max_dispatch_tokens_per_rank+slot)*num_bytes_per_msg;
                                const int index=__popc(selected_mask & ((1u<<lane_id)-1));
                                reinterpret_cast<uint64_t*>(packet+16)[index]=offset;
                            }
                            __syncwarp();
                            if (not is_rank_masked<true>(mask_buffer_ptr,dst_rank)) {
                                const auto* src=reinterpret_cast<const int4*>(rdma_x_src_idx);
                                auto* dst=reinterpret_cast<int4*>(packet+kCEDedupHeaderBytes);
                                UNROLLED_WARP_COPY(8,lane_id,num_int4_per_msg,dst,src,ld_nc_global,st_na_global);
                            }
                            __syncwarp();
                            if (selected) atomic_add_release_global(atomic_finish_counter_per_expert+expert,1);
                        }
                    }
                    continue;
                }
            }

            // Issue IBGDA sends
            if (dst_expert_idx >= 0) {
                const auto dst_rank = dst_expert_idx / num_local_experts;
                const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;
                // Allocate each destination slot exactly once across the two
                // launches. Phase 4 prepares the source once and sends self/NVLink/RDMA;
                // the ordinary send copies PCIe and publishes all notifications.
                if (split_rdma_first and
                    is_cross_pair_pcie(reinterpret_cast<uint64_t>(rdma_recv_x), rank, dst_rank,
                                       nvlink_pair_only) == early_rdma_payload_only)
                    continue;
                int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
                slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
                const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                    dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                    rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg + slot_idx * num_bytes_per_msg;
                auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank, nvlink_pair_only);
                if constexpr (kUseCE) {
                    EP_DEVICE_ASSERT(dst_p2p_ptr!=0 and dst_rank!=rank and dst_rank!=(rank^1));
                    dst_p2p_ptr=ce_allocate(dst_rank,dst_ptr-reinterpret_cast<uint64_t>(rdma_recv_x),
                                           num_bytes_per_msg);
                }
                if (not is_rank_masked<true>(mask_buffer_ptr, dst_rank)) {
                    if (dst_p2p_ptr == 0) {
                        nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, num_bytes_per_msg, dst_rank, dst_expert_local_idx, lane_id, slot_idx);
                    } else {
                        // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
                        const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                        const auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_p2p_ptr);
                        UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                    }
                }

                // Increase counter after finishing
                __syncwarp();
                lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
            }
        }
        // Each writer flushes once after its selected payloads. The early
        // launch completes before the late one; late writers globally synchronize
        // before any completion count can be issued by another block.
        if (split_rdma_first)
            __threadfence_system();
    } else if (warp_id == num_warps - 1 and not early_rdma_payload_only) {
        EP_DEVICE_ASSERT(num_sms > 1);
        if (sm_id == 0) {
            // The first SM is also responsible for checking QPs
            EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts);

            // The first SM is also responsible for cleaning the next buffer
            #pragma unroll
            for (int i = lane_id; i < num_next_clean_int; i += 32)
                next_clean[i] = 0;

            // Notify before executing `int_p`
            __syncwarp();
            #pragma unroll
            for (int i = lane_id; i < num_experts; i += 32)
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG);
        }

        // This SM should be responsible for some destination experts, read `topk_idx` for them
        int expert_count[kNumMaxWarpGroups] = {0};
        const auto expert_begin_idx = sm_id * num_warp_groups;
        const auto expert_end_idx = min(expert_begin_idx + num_warp_groups, num_experts);

        // Per lane count
        #pragma unroll 8
        for (int i = lane_id; i < num_tokens * num_topk; i += 32) {
            auto idx = static_cast<int>(__ldg(topk_idx + i));
            if (idx >= expert_begin_idx and idx < expert_end_idx)
                expert_count[idx - expert_begin_idx]++;
        }

        // Warp reduce
        #pragma unroll
        for (int i = expert_begin_idx; i < expert_end_idx; ++i) {
            auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
            if (lane_id == 0) {
                shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - sum);
            }
        }
    }
    __syncthreads();
    if (early_rdma_payload_only)
        return;
    if (split_rdma_first)
        cg::this_grid().sync();

    // Issue count sends
    if (responsible_expert_idx < num_experts and sub_warp_id == 0 and lane_id == 0) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto dst_expert_local_idx = responsible_expert_idx % num_local_experts;
        const auto num_tokens_sent = shared_num_tokens_sent_per_expert[responsible_expert_idx - sm_id * num_warp_groups];

        // Wait local sends issued and send expert counts
        while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) != FINISHED_SUM_TAG * 2)
            ;
        auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_count + dst_expert_local_idx * num_ranks + rank);
        auto dst_p2p_ptr = control_p2p_ptr(dst_ptr, rank, dst_rank, nvlink_pair_only);
        if constexpr (kUseCE) {
            ce_device.notifications[responsible_expert_idx]=-num_tokens_sent-1;
        } else if (not is_rank_masked(mask_buffer_ptr, dst_rank)) {
            if (dst_p2p_ptr == 0) {
                #ifdef DISABLE_SM90_FEATURES
                nvshmemi_ibgda_rma_p(reinterpret_cast<int*>(dst_ptr), -num_tokens_sent - 1, dst_rank, dst_expert_local_idx);
#else
                nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), -num_tokens_sent - 1, dst_rank, dst_expert_local_idx);
#endif
            } else {
                st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), -num_tokens_sent - 1);
            }
        }

        // Clean workspace for next use
        atomic_counter_per_expert[responsible_expert_idx] = 0;
        atomic_finish_counter_per_expert[responsible_expert_idx] = 0;

        // Clean `packed_recv_count`
        if (dst_rank == 0)
            packed_recv_count[dst_expert_local_idx] = 0;
    }
    __syncwarp();

// Receiving phase
LOW_LATENCY_DISPATCH_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // For send-and-recv kernels, we need a grid sync for making `packed_recv_count` visible
    if (phases & LOW_LATENCY_SEND_PHASE)
        cg::this_grid().sync();

    // Receiving and packing
    if (responsible_expert_idx < num_experts) {
        const auto src_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto rdma_recv_x_uint8 = static_cast<uint8_t*>(rdma_recv_x) +
            local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
            src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
        const auto recv_x_int4 =
            static_cast<int4*>(packed_recv_x) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_int4;
        const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;
        const auto num_aligned_scales = align_up<int>(num_scales, sizeof(float) / sizeof(scale_t));
        const auto recv_x_scales = static_cast<scale_t*>(packed_recv_x_scales) +
            local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_aligned_scales;

        // Shared between sub-warps in warp groups
        __shared__ int shared_num_recv_tokens[kNumMaxWarpGroups], shared_recv_token_begin_idx[kNumMaxWarpGroups];

        // Wait tokens to arrive
        // NOTES: using sub-warp 1 to overlap with sub-warp 0
        int num_recv_tokens = 0, recv_token_begin_idx;
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 15);
        if (sub_warp_id == 1 and lane_id == 0) {
            auto start_time = clock64();
            uint64_t wait_recv_cost = 0;
            if (not is_rank_masked(mask_buffer_ptr, src_rank)) {
                while ((num_recv_tokens = ld_acquire_sys_global(rdma_recv_count + local_expert_idx * num_ranks + src_rank)) ==
                           0                                                               // data not arrived
                       && (wait_recv_cost = clock64() - start_time) <= NUM_TIMEOUT_CYCLES  // not timeout
                )
                    ;
            }
            // Do not receive tokens if rank timeout or masked
            if (num_recv_tokens == 0)
                num_recv_tokens = -1;
            // Mask rank if timeout
            if (wait_recv_cost > NUM_TIMEOUT_CYCLES) {
                printf("Warning: DeepEP timeout for dispatch receive, rank %d, local_expert_idx %d, src_rank %d\n",
                       rank,
                       local_expert_idx,
                       src_rank);
                if (mask_buffer_ptr == nullptr)
                    trap();
                atomicExch(mask_buffer_ptr + src_rank, 1);
            }

            num_recv_tokens = -num_recv_tokens - 1;
            recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
            shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
            shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
            recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);

            // Add stats for diagnosis
            if (cumulative_local_expert_recv_stats != nullptr)
                atomicAdd(cumulative_local_expert_recv_stats + local_expert_idx, num_recv_tokens);
            if (dispatch_wait_recv_cost_stats != nullptr)
                atomicAdd(reinterpret_cast<unsigned long long*>(dispatch_wait_recv_cost_stats + src_rank), wait_recv_cost);
        }
        asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 2), "r"(num_warps_per_group * 32));
        num_recv_tokens = shared_num_recv_tokens[warp_group_id];
        recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

        // Copy tokens
        EP_DEVICE_ASSERT(num_scales <= 64);
        for (int i = sub_warp_id; i < num_recv_tokens; i += num_warps_per_group) {
            // Copy source info
            const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);
            if (lane_id == 0)
                recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
            __syncwarp();

            // Copy data
            // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
            const auto src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
            const auto dst_data = recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
            UNROLLED_WARP_COPY(7, lane_id, hidden_int4, dst_data, src_data, ld_nc_global, st_na_global);

            // Copy scales
            if constexpr (kUseFP8) {
                // Equivalent CuTe layout:
                //   (num_tokens, (num_packed, num_elems_per_pack)):(num_elems_per_pack, (num_tokens * num_elems_per_pack, 1))
                const auto src_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
                const auto num_elems_per_pack = static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
                const auto token_idx = recv_token_begin_idx + i;
                const auto token_stride = num_elems_per_pack;
                const auto pack_stride = num_ranks * num_max_dispatch_tokens_per_rank * num_elems_per_pack;
                if (lane_id < num_scales) {
                    const auto pack_idx = lane_id / num_elems_per_pack;
                    const auto elem_idx = lane_id % num_elems_per_pack;
                    auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id));
                    recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                }
                if (lane_id + 32 < num_scales) {
                    const auto pack_idx = (lane_id + 32) / num_elems_per_pack;
                    const auto elem_idx = (lane_id + 32) % num_elems_per_pack;
                    auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id + 32));
                    recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                }
            }
        }
    }
}

void dispatch(void* packed_recv_x,
              void* packed_recv_x_scales,
              int* packed_recv_src_info,
              int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* mask_buffer_ptr,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x,
              int* rdma_recv_count,
              void* rdma_x,
              const void* x,
              const topk_idx_t* topk_idx,
              int* next_clean,
              int num_next_clean_int,
              int num_tokens,
              int hidden,
              int num_max_dispatch_tokens_per_rank,
              int num_topk,
              int num_experts,
              int rank,
              int num_ranks,
              bool use_fp8,
              bool round_scale,
              bool use_ue8m0,
              void* workspace,
              int num_device_sms,
              cudaStream_t stream,
              int phases,
              bool nvlink_pair_only) {
    constexpr int kNumMaxTopK = 11;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 32 / num_warp_groups;
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);
    EP_HOST_ASSERT(kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms = ceil_div(num_experts, num_warp_groups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

    // Workspace checks
    auto atomic_counter_per_expert = static_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

    // FP8 checks
    if (use_ue8m0)
        EP_HOST_ASSERT(round_scale and "UE8M0 SF requires `round_scale=True`");

#define DISPATCH_LAUNCH_CASE(hidden)                         \
    {                                                        \
        auto dispatch_func = dispatch<false, false, hidden>; \
        if (use_fp8 and not use_ue8m0)                       \
            dispatch_func = dispatch<true, false, hidden>;   \
        if (use_fp8 and use_ue8m0)                           \
            dispatch_func = dispatch<true, true, hidden>;    \
        auto launch_dispatch_phase = [&](int launch_phases, bool stage_ce = false) { \
        auto chosen_func = stage_ce ? dispatch<false,false,hidden,true> : dispatch_func; \
        LAUNCH_KERNEL(&cfg,                                  \
                      chosen_func,                           \
                      packed_recv_x,                         \
                      packed_recv_x_scales,                  \
                      packed_recv_src_info,                  \
                      packed_recv_layout_range,              \
                      packed_recv_count,                     \
                      mask_buffer_ptr,                       \
                      cumulative_local_expert_recv_stats,    \
                      dispatch_wait_recv_cost_stats,         \
                      rdma_recv_x,                           \
                      rdma_recv_count,                       \
                      rdma_x,                                \
                      x,                                     \
                      topk_idx,                              \
                      atomic_counter_per_expert,             \
                      atomic_finish_counter_per_expert,      \
                      next_clean,                            \
                      num_next_clean_int,                    \
                      num_tokens,                            \
                      num_max_dispatch_tokens_per_rank,      \
                      num_topk,                              \
                      num_experts,                           \
                      rank,                                  \
                      num_ranks,                             \
                      num_warp_groups,                       \
                      num_warps_per_group,                   \
                      round_scale,                           \
                      launch_phases,                         \
                      nvlink_pair_only);                     \
        };                                                   \
        if (split_rdma_first and (phases & LOW_LATENCY_SEND_PHASE)) { \
            launch_dispatch_phase(LOW_LATENCY_SEND_PHASE | kRDMAFirstPayloadPhase); \
            if (not ce_configured or not ce_skip_local_drain) launch_drain_rdma_before_pcie(rank, num_ranks, stream); \
        }                                                    \
        if (ce_configured and split_rdma_first and (phases & LOW_LATENCY_SEND_PHASE)) { \
            EP_HOST_ASSERT(not use_fp8); \
            ce_begin(rank,num_ranks,hidden,num_experts,num_max_dispatch_tokens_per_rank,stream); \
            launch_dispatch_phase(LOW_LATENCY_SEND_PHASE,true); \
            ce_finish(true,rdma_recv_x,rdma_recv_count,mask_buffer_ptr,stream); \
            if (phases & LOW_LATENCY_RECV_PHASE) launch_dispatch_phase(LOW_LATENCY_RECV_PHASE); \
        } else launch_dispatch_phase(phases);                \
    }                                                        \
    break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
#ifdef DISABLE_SM90_FEATURES
    const bool split_rdma_first = not nvlink_pair_only;
#else
    constexpr bool split_rdma_first = false;
#endif
    SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

#ifndef DISABLE_SM90_FEATURES
template <int kNumSendUnrolls>
__forceinline__ __device__ int logfmt_encode(void* buffer, nv_bfloat162* shared_amaxmin, const int& lane_id) {
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr float kLogThreshold = 0;
    constexpr float kMinClip = 32;  // `== log_2(2 ^ (2 ^ 5))`
    constexpr int kNumBits = 10;
    constexpr int kNumValues = 1 << (kNumBits - 1);

    int4 int4_values[kNumSendUnrolls];
    const auto& uint32_values = reinterpret_cast<uint32_t*>(int4_values);
    const auto& bf162_values = reinterpret_cast<nv_bfloat162*>(int4_values);

    // Calculate lane offset
    const auto& ld_buffer = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(buffer) + lane_id * (kNumSendUnrolls * sizeof(int4)));
    const auto& st_buffer =
        reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(buffer) + lane_id * (kNumSendUnrolls * sizeof(int4) * 10 / 16));

    // Local log amax
    auto bf162_amax = __nv_bfloat162(CUDART_ZERO_BF16, CUDART_ZERO_BF16);
    auto bf162_amin = __nv_bfloat162(CUDART_INF_BF16, CUDART_INF_BF16);
    uint32_t local_signs = 0;
    #pragma unroll
    for (int k = 0; k < kNumSendUnrolls * kNumElemsPerInt4 / 2; ++k) {
        // TODO: eliminate bank conflicts
        uint32_values[k] = ld_buffer[k];
        local_signs |= ((uint32_values[k] >> 15) & 1) << (k * 2);
        local_signs |= ((uint32_values[k] >> 31) & 1) << (k * 2 + 1);
        uint32_values[k] &= 0x7fff7fff;

        bf162_amax = __hmax2(bf162_amax, bf162_values[k]);
        bf162_amin = __hmin2(bf162_amin, bf162_values[k]);
    }

    // Reduce per 128 channels
    // TODO: figure out how hardware do 2-byte min/max
    auto amax = std::max(static_cast<float>(bf162_amax.x), static_cast<float>(bf162_amax.y));
    auto amin = std::min(static_cast<float>(bf162_amin.x), static_cast<float>(bf162_amin.y));
    constexpr static int kNumLanesToReduce = 128 * sizeof(nv_bfloat16) / (kNumSendUnrolls * sizeof(int4));
    amax = warp_reduce_max<kNumLanesToReduce>(amax);
    amin = warp_reduce_min<kNumLanesToReduce>(amin);

    // Write min/max into the shared memory
    if (shared_amaxmin != nullptr)
        *shared_amaxmin = __nv_bfloat162(amax, amin);
    __syncwarp();

    // Calculate log amin/amax float
    const auto& log_amax = log2f_approx(amax);
    const auto& log_amin = fmaxf(log2f_approx(amin), log_amax - kMinClip);
    const bool& enable_cast = warp_reduce_and<kNumLanesToReduce, true>(log_amax < kLogThreshold and log_amin < log_amax);

    // Case into LogFMT-10 if satisfied
    if (enable_cast) {
        const auto step = (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
        const auto step_inv = 1.0f / step;
        const auto rounding = 2.0f - log2f_approx((1.0f + exp2f_approx(step)) * 0.5f) * step_inv;
        const auto fused_rounding = rounding - log_amin * step_inv;

        // Pack every 256 bits into 160 bits
        EP_STATIC_ASSERT(kNumSendUnrolls == 2 or kNumSendUnrolls == 4, "kNumSendUnrolls == 2 or 4 only");
        uint32_t encoded[kNumElemsPerInt4 * 2];
        #pragma unroll 1
        for (int i = 0; i < kNumSendUnrolls / 2; ++i) {
            #pragma unroll
            for (int k = 0; k < kNumElemsPerInt4; ++k) {
                const auto& [x, y] = __bfloat1622float2(bf162_values[i * kNumElemsPerInt4 + k]);
                encoded[k * 2 + 0] = __float2uint_rd(fmaxf(log2f_approx(x) * step_inv + fused_rounding, 0));
                encoded[k * 2 + 1] = __float2uint_rd(fmaxf(log2f_approx(y) * step_inv + fused_rounding, 0));
            }
            st_buffer[i * 5 + 0] = (encoded[0] >> 0) | (encoded[1] << 9) | (encoded[2] << 18) | (encoded[3] << 27);
            st_buffer[i * 5 + 1] = (encoded[3] >> 5) | (encoded[4] << 4) | (encoded[5] << 13) | (encoded[6] << 22) | (encoded[7] << 31);
            st_buffer[i * 5 + 2] = (encoded[7] >> 1) | (encoded[8] << 8) | (encoded[9] << 17) | (encoded[10] << 26);
            st_buffer[i * 5 + 3] =
                (encoded[10] >> 6) | (encoded[11] << 3) | (encoded[12] << 12) | (encoded[13] << 21) | (encoded[14] << 30);
            st_buffer[i * 5 + 4] = (encoded[14] >> 2) | (encoded[15] << 7) | ((i == 0) ? (local_signs << 16) : (local_signs & 0xffff0000u));
        }
        tma_store_fence();
        __syncwarp();
    }

    // Return TMA copy bytes
    return enable_cast ? (32 * (kNumSendUnrolls * sizeof(int4) * 8 * 10 / 16 / 8)) : (32 * (kNumSendUnrolls * sizeof(int4)));
}
#endif

template <int kNumLanes, int kNumSendUnrolls, int kNumRecvUnrolls>
__forceinline__ __device__ void logfmt_check_amaxmin(
    uint8_t* meta_buffer, float2* shared_log_amax, float2* shared_log_amin, int* shared_cast_info, const int lane_id) {
    constexpr float kLogThreshold = 0;
    constexpr float kMinClip = 32;  // `== log_2(2 ^ (2 ^ 5))`

    bool enable_cast = true;
    if (lane_id < kNumLanes) {
        // Calculate log amin/amax float
        auto amaxmin2 = reinterpret_cast<uint64_t*>(meta_buffer)[lane_id];
        const auto& bf162_amaxmin = reinterpret_cast<__nv_bfloat162*>(&amaxmin2);
        float log_amax[2], log_amin[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            auto amax = static_cast<float>(bf162_amaxmin[i].x);
            auto amin = static_cast<float>(bf162_amaxmin[i].y);
            log_amax[i] = log2f_approx(amax);
            log_amin[i] = amin == 0 ? log_amax[i] - kMinClip : fmaxf(log2f_approx(amin), log_amax[i] - kMinClip);
            enable_cast = enable_cast and log_amax[i] < kLogThreshold and log_amin[i] < log_amax[i];
        }
        shared_log_amax[lane_id] = make_float2(log_amax[0], log_amax[1]);
        shared_log_amin[lane_id] = make_float2(log_amin[0], log_amin[1]);
    }

    const auto& casted = warp_reduce_and<kNumSendUnrolls>(enable_cast) ? 1u << (lane_id / kNumRecvUnrolls) : 0u;
    const auto& num_casted_prefix = __popc(warp_reduce_or<kNumRecvUnrolls, true>(casted) & ((1u << (lane_id / kNumRecvUnrolls)) - 1));

    if (lane_id < kNumLanes and lane_id % kNumRecvUnrolls == 0)
        shared_cast_info[lane_id / kNumRecvUnrolls] = (num_casted_prefix << 1) | (casted ? 1u : 0u);
    __syncwarp();
}

template <int kNumRecvUnrolls>
__forceinline__ __device__ void decode_and_accumulate(
    uint32_t* ld_buffer, float* accum, const float& log_amax, const float& log_amin, const bool& enable_cast, const float& weight) {
    if (enable_cast) {
        constexpr int kNumBits = 10;
        constexpr int kNumValues = 1 << (kNumBits - 1);

        const auto& step = (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
        auto decode = [=](const uint32_t& encoded, const uint32_t& sign) {
            const auto decoded = encoded == 0 ? .0f : exp2f_approx((encoded - 1) * step + log_amin);
            return sign ? -decoded : decoded;
        };

        EP_STATIC_ASSERT(kNumRecvUnrolls == 2 or kNumRecvUnrolls == 4, "kNumRecvUnrolls == 2 or 4 only");
        #pragma unroll
        for (int i = 0; i < kNumRecvUnrolls / 2; ++i) {
            uint32_t concat[6];
            concat[0] = ld_buffer[i * 5];
            #pragma unroll
            for (int k = 1; k < 5; ++k)
                concat[k] = (ld_buffer[i * 5 + k - 1] >> (32 - k * 5)) | (ld_buffer[i * 5 + k] << (k * 5));
            concat[5] = ld_buffer[i * 5 + 4] >> 7;

            const uint32_t& local_signs = ld_buffer[i * 5 + 4] >> 16;
            #pragma unroll
            for (int k = 0; k < 5; ++k) {
                accum[i * 16 + k * 3 + 0] += decode((concat[k] >> 0) & 0x1ff, (local_signs >> (k * 3 + 0)) & 1) * weight;
                accum[i * 16 + k * 3 + 1] += decode((concat[k] >> 9) & 0x1ff, (local_signs >> (k * 3 + 1)) & 1) * weight;
                accum[i * 16 + k * 3 + 2] += decode((concat[k] >> 18) & 0x1ff, (local_signs >> (k * 3 + 2)) & 1) * weight;
            }
            accum[i * 16 + 15] += decode(concat[5] & 0x1ff, (local_signs >> 15) & 1) * weight;
        }
    } else {
        #pragma unroll
        for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
            auto bf16_pack = *reinterpret_cast<__nv_bfloat162*>(ld_buffer + k);
            accum[k * 2 + 0] += static_cast<float>(bf16_pack.x) * weight;
            accum[k * 2 + 1] += static_cast<float>(bf16_pack.y) * weight;
        }
    }
}

template <bool kUseLogFMT, int kHidden, int kNumMaxTopk, int kNumMaxUnrolls, bool kUseCE = false>
__global__ __launch_bounds__(1024, 1) void combine(void* combined_x,
                                                   void* rdma_recv_x,
                                                   int* rdma_recv_flag,
                                                   void* rdma_send_x,
                                                   const void* x,
                                                   const topk_idx_t* topk_idx,
                                                   const float* topk_weights,
                                                   const int* src_info,
                                                   const int64_t* layout_range,
                                                   int* mask_buffer_ptr,
                                                   int64_t* combine_wait_recv_cost_stats,
                                                   int* next_clean,
                                                   int num_next_clean_int,
                                                   int* atomic_clean_flag,
                                                   int num_combined_tokens,
                                                   int hidden,
                                                   int num_topk,
                                                   int num_max_dispatch_tokens_per_rank,
                                                   int num_experts,
                                                   int rank,
                                                   int num_ranks,
                                                   int num_warp_groups,
                                                   int num_warps_per_group,
                                                   int phases,
                                                   bool zero_copy,
                                                   bool nvlink_pair_only) {
    const auto sm_id = __shfl_sync(0xffffffff, static_cast<int>(blockIdx.x), 0);
    const auto num_sms = __shfl_sync(0xffffffff, static_cast<int>(gridDim.x), 0);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = __shfl_sync(0xffffffff, static_cast<int>(blockDim.x), 0);
    const auto warp_id = __shfl_sync(0xffffffff, thread_id / 32, 0), lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;
#ifdef DISABLE_SM90_FEATURES
    const bool split_rdma_first = not nvlink_pair_only;
    const bool early_rdma_payload_only = (phases & kRDMAFirstPayloadPhase) != 0;
#else
    constexpr bool split_rdma_first = false, early_rdma_payload_only = false;
#endif

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr int64_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

    // Use different unroll factors for send and recv phases
    constexpr int kNumSendUnrolls = kHidden % (32 * 4 * sizeof(int4) / sizeof(nv_bfloat16)) == 0 ? 4 : 2;
    constexpr int kNumRecvUnrolls = 2;
    constexpr int hidden_bf16_int4_pad = align_up(static_cast<int>(hidden_bf16_int4), 32 * kNumSendUnrolls);
    EP_STATIC_ASSERT(kHidden % (32 * 2 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls <= kNumMaxUnrolls and kNumRecvUnrolls <= kNumMaxUnrolls, "Invalid unrolls");
    EP_STATIC_ASSERT(hidden_bf16_int4 % kNumSendUnrolls == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls >= kNumRecvUnrolls, "Invalid unroll factors");

    // Message package
    EP_STATIC_ASSERT(kHidden % 128 == 0, "Invalid hidden");
    constexpr int kNumDivisions = kHidden / 128;
    constexpr int kNumMetaBytes = kNumDivisions * sizeof(nv_bfloat162);
    constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16) + kNumMetaBytes;
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_COMBINE_RECV;

    // Clean up next buffer
    if (not early_rdma_payload_only and sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
        #pragma unroll
        for (int i = lane_id; i < num_next_clean_int; i += 32)
            next_clean[i] = 0;

        // Notify before executing `int_p`
        __syncwarp();
        if (lane_id == 0)
            atomic_add_release_global(atomic_clean_flag, num_experts);
    }

    // Issue IBGDA sends
    if (responsible_expert_idx < num_experts) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
        const auto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
        const auto local_x =
            static_cast<const int4*>(x) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
        const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto rdma_send_x_vec =
            static_cast<uint8_t*>(rdma_send_x) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;

        // Unpack layout
        int offset, num_tokens_to_send;
        unpack2(layout, num_tokens_to_send, offset);
#ifndef DISABLE_SM90_FEATURES
        // TMA stuffs
        constexpr int kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls;
        constexpr int kNumStages = 3;
        constexpr int kNumPrefetch = 1;
        EP_STATIC_ASSERT(kNumStages == 3 and kNumPrefetch == 1, "Invalid stages");

        auto smem_ptr = smem_buffer + warp_id * (kNumStages * (kNumTMABufferBytes + 16) + kNumMetaBytes);
        uint32_t tma_phase = 0;
        auto tma_buffers = PatternVisitor([=](const int& i) { return reinterpret_cast<int4*>(smem_ptr + i * (kNumTMABufferBytes + 16)); });
        auto full_barriers = PatternVisitor(
            [=](const int& i) { return reinterpret_cast<uint64_t*>(smem_ptr + i * (kNumTMABufferBytes + 16) + kNumTMABufferBytes); });
        auto meta_buffers = kUseLogFMT ? reinterpret_cast<nv_bfloat162*>(smem_ptr + kNumStages * (kNumTMABufferBytes + 16)) : nullptr;
        EP_STATIC_ASSERT(kNumSendUnrolls * kNumStages <= 12, "TMA buffer size exceed limit");

        // Initialize m-barriers
        if (lane_id < kNumStages) {
            mbarrier_init(full_barriers[lane_id], 1);
            fence_barrier_init();
        }
        __syncwarp();

        constexpr int kNumIters = hidden_bf16_int4_pad / (32 * kNumSendUnrolls);
        auto tma_load_and_arrive = [&](const int& stage_idx, const int4* gmem_ptr, const int& num_bytes) {
            tma_load_1d(tma_buffers[stage_idx], gmem_ptr, full_barriers[stage_idx], num_bytes);
            mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_bytes);
        };
        auto get_num_tma_bytes = [&](const int& offset_int4) {
            return min(kNumTMABufferBytes, static_cast<int>((hidden_bf16_int4 - offset_int4) * sizeof(int4)));
        };
#endif

        // Issue IBGDA send
        if (not is_rank_masked<true>(mask_buffer_ptr, dst_rank) and
            (not split_rdma_first or
             is_cross_pair_pcie(reinterpret_cast<uint64_t>(rdma_recv_x), rank, dst_rank,
                                nvlink_pair_only) != early_rdma_payload_only)) {
            for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += num_warps_per_group) {
                const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
                const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
                const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);

                // Copy directly to local rank, or copy to buffer and issue RDMA
                const auto src_idx = __shfl_sync(0xffffffff, __ldg(local_src_info + token_idx), 0);
                const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                    (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot;
                auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank, nvlink_pair_only);
                if constexpr (kUseCE) {
                    EP_DEVICE_ASSERT(dst_p2p_ptr!=0 and dst_rank!=rank and dst_rank!=(rank^1));
                    dst_p2p_ptr=ce_allocate(dst_rank,dst_ptr-reinterpret_cast<uint64_t>(rdma_recv_x),hidden*2);
                }
                int num_send_bytes = hidden * sizeof(nv_bfloat16);

                if (not zero_copy or dst_p2p_ptr != 0) {
                    // Read from `cpy_src_int4_ptr` and copy into `cpy_dst_int4_ptr`
                    const auto cpy_src_int4_ptr = zero_copy ? reinterpret_cast<int4*>(buf_ptr) : x_int4;
                    const auto cpy_dst_int4_ptr =
                        dst_p2p_ptr == 0 ? reinterpret_cast<int4*>(buf_ptr) : reinterpret_cast<int4*>(dst_p2p_ptr);
#ifndef DISABLE_SM90_FEATURES
                    // Prefetch
                    if (elect_one_sync())
                        tma_load_and_arrive(0, cpy_src_int4_ptr, get_num_tma_bytes(0));
                    __syncwarp();

                    int tma_offset_bytes = kNumMetaBytes;
                    #pragma unroll
                    for (int i = lane_id * kNumSendUnrolls, iter_idx = 0; i < hidden_bf16_int4_pad; i += 32 * kNumSendUnrolls, ++iter_idx) {
                        // Load the next iteration
                        const int& stage_idx = iter_idx % kNumStages;
                        const int& next_stage_idx = (iter_idx + 1) % kNumStages;
                        if (iter_idx + 1 < kNumIters and elect_one_sync()) {
                            tma_store_wait<kNumStages - kNumPrefetch - 1>();
                            const auto& offset_int4 = i + 32 * kNumSendUnrolls;
                            tma_load_and_arrive(next_stage_idx, cpy_src_int4_ptr + offset_int4, get_num_tma_bytes(offset_int4));
                        }
                        __syncwarp();

                        // Wait the current TMA arrival
                        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
                        mbarrier_wait<true>(full_barriers[stage_idx], tma_phase, stage_idx);
                        if constexpr (kUseLogFMT) {
                            // Cast if possible
                            constexpr int kNumInt4PerDivision = 128 / kNumElemsPerInt4;
                            int num_tma_bytes = logfmt_encode<kNumSendUnrolls>(
                                tma_buffers[stage_idx],
                                // NOTES: only the leader lane will write the result
                                (i % kNumInt4PerDivision == 0) ? meta_buffers + i / kNumInt4PerDivision : nullptr,
                                lane_id);
                            if (elect_one_sync())
                                tma_store_1d(
                                    tma_buffers[stage_idx], reinterpret_cast<uint8_t*>(cpy_dst_int4_ptr) + tma_offset_bytes, num_tma_bytes);
                            tma_offset_bytes += num_tma_bytes;
                        } else {
                            // BF16 original values
                            if (elect_one_sync())
                                tma_store_1d(tma_buffers[stage_idx], cpy_dst_int4_ptr + i, get_num_tma_bytes(i));
                        }
                        __syncwarp();
                    }

                    // Store metadata (min/max values) for LogFMT
                    if constexpr (kUseLogFMT) {
                        num_send_bytes = tma_offset_bytes;
                        if (elect_one_sync())
                            tma_store_1d(meta_buffers, cpy_dst_int4_ptr, kNumMetaBytes);
                    }

                    // Flush all stores
                    tma_store_wait<0>();
                    __syncwarp();
#else
                    constexpr int kNumUnrolls = 4;
                    // Keep the existing warp/SM budget and four-vector register
                    // pipeline. Only ordinary BF16 writes to a remote mapped
                    // GPU use lane-contiguous vectors, as dispatch already does.
                    // Local/RDMA staging and LogFMT retain their original path.
                    if (not kUseLogFMT and dst_p2p_ptr != 0 and dst_rank != rank) {
                        UNROLLED_WARP_COPY(kNumUnrolls, lane_id, hidden_bf16_int4,
                                           cpy_dst_int4_ptr, cpy_src_int4_ptr, ld_nc_global, st_na_global);
                    } else {
                    constexpr int hidden_bf16_int4_pad = align_up(static_cast<int>(hidden_bf16_int4), 32 * kNumUnrolls);
                    #pragma unroll
                    for (int i = lane_id * kNumUnrolls; i < hidden_bf16_int4_pad; i += 32 * kNumUnrolls) {
                        // Read
                        int4 int4_values[kNumUnrolls];
                        if (i < hidden_bf16_int4) {
                            #pragma unroll
                            for (int k = 0; k < kNumUnrolls; ++k)
                                int4_values[k] = ld_nc_global(cpy_src_int4_ptr + i + k);
                        }
                        auto bf16_values = reinterpret_cast<nv_bfloat16*>(int4_values);
                        auto uint32_values = reinterpret_cast<uint32_t*>(int4_values);

                        // Simulated cast
                        if constexpr (kUseLogFMT) {
                            constexpr float kThreshold = 1;
                            constexpr float kMinClip = 32;  // `== log_2(2 ^ (2 ^ 5))`
                            constexpr int kNumBits = 10;
                            constexpr int kNumValues = 1 << (kNumBits - 1);
                            EP_STATIC_ASSERT(kHidden % (kNumElemsPerInt4 * 32) == 0 and kNumElemsPerInt4 == 8, "Invalid hidden");

                            // Local log amax
                            float log_abs_values[kNumElemsPerInt4 * kNumUnrolls], log_amax, log_amin, amax;
                            #pragma unroll
                            for (int j = 0; j < kNumElemsPerInt4 * kNumUnrolls; ++j) {
                                auto value = static_cast<float>(bf16_values[j]);
                                log_abs_values[j] = log2f_approx(fabsf(value));
                                amax = j == 0 ? value : fmaxf(amax, fabsf(value));
                                log_amax = j == 0 ? log_abs_values[j] : fmaxf(log_amax, log_abs_values[j]);
                                log_amin = value != 0 ? (j == 0 ? log_abs_values[j] : fminf(log_amin, log_abs_values[j])) : log_amin;
                            }

                            // Reduce per 128 channels
                            amax = warp_reduce_max<(16 / kNumUnrolls)>(amax);
                            log_amax = warp_reduce_max<(16 / kNumUnrolls)>(log_amax);
                            log_amin = fmaxf(warp_reduce_min<(16 / kNumUnrolls)>(log_amin), log_amax - kMinClip);

                            const auto step = (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
                            const auto step_inv = 1.0f / step;
                            const auto rounding = 2.0f - log2f_approx((1.0f + exp2f_approx(step)) * 0.5f) * step_inv;

                            // Use LogFMT only with `amax <= kThreshold` (maybe not all quarter-warps)
                            if (amax <= kThreshold and log_amin < log_amax) {
                                // Transform
                                auto transform = [=](const float& log_abs_value) -> nv_bfloat16 {
                                    const auto encoded = floorf((log_abs_value - log_amin) * step_inv + rounding);
                                    const auto decoded = exp2f_approx((encoded - 1) * step + log_amin);
                                    return decoded;
                                };
                                #pragma unroll
                                for (int j = 0; j < kNumElemsPerInt4 * kNumUnrolls; j += 2) {
                                    auto bf162_pack = __nv_bfloat162(transform(log_abs_values[j]), transform(log_abs_values[j + 1]));
                                    auto uint32_pack = *reinterpret_cast<uint32_t*>(&bf162_pack);
                                    uint32_values[j / 2] = (uint32_values[j / 2] & 0x80008000) | uint32_pack;
                                }
                            }
                            __syncwarp();
                        }

                        // Store
                        EP_STATIC_ASSERT(hidden_bf16_int4 % kNumUnrolls == 0, "Invalid hidden");
                        if (i < hidden_bf16_int4) {
                            #pragma unroll
                            for (int k = 0; k < kNumUnrolls; ++k)
                                st_na_global(cpy_dst_int4_ptr + i + k, int4_values[k]);
                        }
                    }
                    }
#endif
                }

                // Issue RDMA
                // NOTES: for zero-copy mode, we assume the data is already in the send buffer
                if (dst_p2p_ptr == 0)
                    nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, num_send_bytes, dst_rank, local_expert_idx, lane_id, token_idx - offset);
            }
        }

        // One system fence per sending writer (after its token loop), before
        // the warp-group barrier and the separate RDMA finishing-flag issuer.
        // A fence in only the issuer would not cover other writers' P2P stores.
#ifdef DISABLE_SM90_FEATURES
        if (split_rdma_first and dst_rank != rank and
            nvshmemi_get_p2p_ptr(reinterpret_cast<uint64_t>(rdma_recv_x), rank, dst_rank, nvlink_pair_only) != 0)
            __threadfence_system();
#endif

        // Put the finishing flag
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 16);
        asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 1), "r"(num_warps_per_group * 32));
        if (not early_rdma_payload_only and sub_warp_id == 1 and lane_id == 0) {
            while (ld_acquire_global(atomic_clean_flag) == 0)
                ;
            auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_flag + global_expert_idx);
            auto dst_p2p_ptr = control_p2p_ptr(dst_ptr, rank, dst_rank, nvlink_pair_only);
            if (not kUseCE and not is_rank_masked(mask_buffer_ptr, dst_rank)) {
                if (dst_p2p_ptr == 0) {
                    #ifdef DISABLE_SM90_FEATURES
                nvshmemi_ibgda_rma_p(reinterpret_cast<int*>(dst_ptr), 1, dst_rank, local_expert_idx);
#else
                nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), 1, dst_rank, local_expert_idx);
#endif
                } else {
                    st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), 1);
                }
            }
            atomic_add_release_global(atomic_clean_flag, -1);
        }
        __syncwarp();
#ifndef DISABLE_SM90_FEATURES
        // Destroy m-barriers
        if (lane_id < kNumStages) {
            mbarrier_inval(full_barriers[lane_id]);
            fence_barrier_init();
        }
#endif
        __syncwarp();
    }

// Receiving phase
LOW_LATENCY_COMBINE_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Wait all ranks to arrive
    if (responsible_expert_idx < num_experts) {
        EP_DEVICE_ASSERT(num_warps_per_group > 1);
        if (sub_warp_id == 0 and lane_id == 0) {
            const auto src_rank = responsible_expert_idx / num_local_experts;
            auto start_time = clock64();
            uint64_t wait_recv_cost = 0;
            if (not is_rank_masked(mask_buffer_ptr, src_rank)) {
                while (ld_acquire_sys_global(rdma_recv_flag + responsible_expert_idx) == 0  // recv not ready
                       && (wait_recv_cost = clock64() - start_time) <= NUM_TIMEOUT_CYCLES   // not timeout
                )
                    ;
            }
            // Mask rank if timeout
            if (wait_recv_cost > NUM_TIMEOUT_CYCLES) {
                printf("Warning: DeepEP timeout for combine receive, rank %d, local_expert_idx %d, src_rank %d\n",
                       rank,
                       responsible_expert_idx % num_local_experts,
                       src_rank);
                if (mask_buffer_ptr == nullptr)
                    trap();
                atomicExch(mask_buffer_ptr + src_rank, 1);
            }

            if (combine_wait_recv_cost_stats != nullptr) {
                atomicAdd(reinterpret_cast<unsigned long long*>(combine_wait_recv_cost_stats + src_rank), wait_recv_cost);
            }
        }
    }
    cg::this_grid().sync();

#ifndef DISABLE_SM90_FEATURES
    // Reassign warp groups
    constexpr int kMaxNumGroups = 2;
    const int num_decode_warps = hidden_bf16_int4_pad / (kNumRecvUnrolls * 32);
    const int num_groups = min(kMaxNumGroups, (num_threads / 32) / (num_decode_warps + 1));
    const int decode_warp_idx = __shfl_sync(0xffffffff, warp_id % (num_decode_warps + 1), 0);
    const int group_idx = __shfl_sync(0xffffffff, warp_id / (num_decode_warps + 1), 0);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(num_groups > 0);

    if (group_idx < num_groups) {
        constexpr int kNumStages = 3;
        constexpr int kNumTMABufferBytes = 16 * 2 + kHidden * 2;
        constexpr int kNumBF16PerWarpBytes = 32 * kNumRecvUnrolls * kNumElemsPerInt4 * 2;
        constexpr int kNumLogFMTPerWarpBytes = kNumBF16PerWarpBytes / 16 * 10;
        constexpr int kNumDivisionBytes = kNumDivisions * sizeof(uint32_t);
        constexpr int kNumBytesPerGroup = kNumStages * kNumTMABufferBytes + kHidden * 2 + kNumStages * kNumDivisionBytes * 3;

        // Reallocate shared memory
        const auto smem_group_buffer = smem_buffer + kNumBytesPerGroup * group_idx;
        auto full_barriers =
            PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_group_buffer + i * kNumTMABufferBytes); });
        auto empty_barriers =
            PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_group_buffer + i * kNumTMABufferBytes + 8); });
        auto tma_ld_buffers =
            PatternVisitor([=](const int& i) { return reinterpret_cast<uint8_t*>(smem_group_buffer + i * kNumTMABufferBytes + 16); });
        auto tma_st_buffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<uint32_t*>(smem_group_buffer + kNumStages * kNumTMABufferBytes + i * kNumBF16PerWarpBytes);
        });

        // Redundant when logfmt is disabled
        const auto smem_group_ptr = smem_group_buffer + kNumStages * kNumTMABufferBytes + kHidden * 2;
        auto log_amax_buffers =
            PatternVisitor([=](const int& i) { return reinterpret_cast<float*>(smem_group_ptr + i * kNumDivisionBytes); });
        auto log_amin_buffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<float*>(smem_group_ptr + kNumStages * kNumDivisionBytes + i * kNumDivisionBytes);
        });
        auto cast_info_buffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<int*>(smem_group_ptr + kNumStages * kNumDivisionBytes * 2 + i * kNumDivisionBytes);
        });

        uint32_t tma_phase = 0;
        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
        if (decode_warp_idx == num_decode_warps)
            tma_phase = (1 << kNumStages) - 1;

        // Initialize m-barriers
        if (decode_warp_idx == num_decode_warps and lane_id < kNumStages) {
            mbarrier_init(full_barriers[lane_id], 1);
            mbarrier_init(empty_barriers[lane_id], num_decode_warps);
        }
        asm volatile("bar.sync %0, %1;" ::"r"(group_idx + 1), "r"((num_decode_warps + 1) * 32));

        int stage_idx = 0, topk_idx_by_lane = 0;
        EP_STATIC_ASSERT(kNumMaxTopk <= 32, "Invalid number of topks");
        if (decode_warp_idx == num_decode_warps) {
            // TMA load warp
            for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
                if (lane_id < num_topk)
                    topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
                for (int i = 0; i < num_topk; ++i) {
                    int topk_idx_reg = __shfl_sync(0xffffffff, topk_idx_by_lane, i);
                    if (topk_idx_reg < 0)
                        continue;
                    if (is_rank_masked(mask_buffer_ptr, topk_idx_reg / num_local_experts))
                        continue;

                    mbarrier_wait<true>(empty_barriers[stage_idx], tma_phase, stage_idx);
                    auto buffer = static_cast<uint8_t*>(rdma_recv_x) +
                        (topk_idx_reg * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot;
                    if constexpr (kUseLogFMT) {
                        logfmt_check_amaxmin<kNumDivisions / 2, kNumSendUnrolls, kNumRecvUnrolls>(
                            buffer,
                            reinterpret_cast<float2*>(log_amax_buffers[stage_idx]),
                            reinterpret_cast<float2*>(log_amin_buffers[stage_idx]),
                            cast_info_buffers[stage_idx],
                            lane_id);
                    }
                    if (elect_one_sync()) {
                        int num_casted = 0;
                        if constexpr (kUseLogFMT) {
                            const auto& info = cast_info_buffers[stage_idx][num_decode_warps - 1];
                            num_casted = (info >> 1) + (info & 1);
                        }
                        int num_tma_bytes = num_casted * kNumLogFMTPerWarpBytes + (num_decode_warps - num_casted) * kNumBF16PerWarpBytes;
                        tma_load_1d(
                            tma_ld_buffers[stage_idx], buffer + (kUseLogFMT ? kNumMetaBytes : 0), full_barriers[stage_idx], num_tma_bytes);
                        mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_tma_bytes);
                    }
                    __syncwarp();
                    stage_idx = (stage_idx + 1) % kNumStages;
                }
            }
        } else {
            // Reduction warps
            float topk_weights_by_lane;
            for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
                if (lane_id < num_topk) {
                    topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
                    topk_weights_by_lane = __ldg(topk_weights + token_idx * num_topk + lane_id);
                }
                __syncwarp();

                float combined_values[kNumElemsPerInt4 * kNumRecvUnrolls] = {0.0f};
                for (int i = 0; i < num_topk; ++i) {
                    int topk_idx_reg = __shfl_sync(0xffffffff, topk_idx_by_lane, i);
                    if (topk_idx_reg < 0)
                        continue;
                    if (is_rank_masked(mask_buffer_ptr, topk_idx_reg / num_local_experts))
                        continue;
                    const auto& topk_weight = __shfl_sync(0xffffffff, topk_weights_by_lane, i);

                    mbarrier_wait<true>(full_barriers[stage_idx], tma_phase, stage_idx);
                    if constexpr (kUseLogFMT) {
                        const auto& info = cast_info_buffers[stage_idx][decode_warp_idx];
                        bool enable_cast = info & 1;
                        int num_casted_prefix = info >> 1;
                        int tma_offset =
                            kNumLogFMTPerWarpBytes * num_casted_prefix + kNumBF16PerWarpBytes * (decode_warp_idx - num_casted_prefix);
                        int division_idx = decode_warp_idx * (kNumRecvUnrolls * 2) + lane_id * kNumRecvUnrolls / 16;
                        decode_and_accumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tma_ld_buffers[stage_idx] + tma_offset +
                                                        (enable_cast ? kNumLogFMTPerWarpBytes : kNumBF16PerWarpBytes) / 32 * lane_id),
                            combined_values,
                            log_amax_buffers[stage_idx][division_idx],
                            log_amin_buffers[stage_idx][division_idx],
                            enable_cast,
                            topk_weight);
                    } else {
                        int tma_offset = kNumBF16PerWarpBytes * decode_warp_idx;
                        decode_and_accumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tma_ld_buffers[stage_idx] + tma_offset + kNumBF16PerWarpBytes / 32 * lane_id),
                            combined_values,
                            0,
                            0,
                            false,
                            topk_weight);
                    }

                    if (elect_one_sync())
                        mbarrier_arrive(empty_barriers[stage_idx]);
                    stage_idx = (stage_idx + 1) % kNumStages;
                }
                tma_store_wait<0>();

                #pragma unroll
                for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
                    auto combined_pack = __nv_bfloat162(combined_values[k * 2], combined_values[k * 2 + 1]);
                    tma_st_buffers[decode_warp_idx][kNumRecvUnrolls * 4 * lane_id + k] = *reinterpret_cast<uint32_t*>(&combined_pack);
                }
                tma_store_fence();
                if (elect_one_sync()) {
                    tma_store_1d(tma_st_buffers[decode_warp_idx],
                                 static_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4 + decode_warp_idx * kNumRecvUnrolls * 32,
                                 kNumBF16PerWarpBytes);
                }
                __syncwarp();
            }
        }
    }
#else
    // Reduce tokens
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    for (int hidden_idx = thread_id; hidden_idx < hidden_bf16_int4; hidden_idx += num_threads) {
        for (int token_idx = sm_id; token_idx < num_combined_tokens; token_idx += num_sms) {
            // Read top-k indices and weights
            int reg_topk_idx[kNumMaxTopk];
            float reg_topk_weights[kNumMaxTopk];
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) {
                reg_topk_idx[i] = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + i));
                reg_topk_weights[i] = __ldg(topk_weights + token_idx * num_topk + i);
            }

            float combined_values[kNumElemsPerInt4] = {0.0f};
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) if (reg_topk_idx[i] >= 0) {
                // Read from sources
                auto rdma_buffer_type = reinterpret_cast<const int*>(static_cast<uint8_t*>(rdma_recv_x) + (reg_topk_idx[i] * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot);
                auto rdma_buffer_row = reinterpret_cast<const uint8_t*>(rdma_buffer_type);

                // Reduce
                auto x_vec = ld_nc_global(reinterpret_cast<const int4*>(rdma_buffer_row) + hidden_idx);
                const auto x_bf16 = reinterpret_cast<nv_bfloat16*>(&x_vec);
                #pragma unroll
                for (int j = 0; j < kNumElemsPerInt4; ++ j)
                    combined_values[j] += static_cast<float>(x_bf16[j]) * reg_topk_weights[i];
            }

            // Write results
            int4& combined_int4 = *reinterpret_cast<int4*>(combined_values);
            auto combined_bf16 = reinterpret_cast<nv_bfloat16*>(&combined_values);
            #pragma unroll
            for (int j = 0; j < kNumElemsPerInt4; ++ j)
                combined_bf16[j] = static_cast<nv_bfloat16>(combined_values[j]);
            (static_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4)[hidden_idx] = combined_int4;
        }
    }
#endif
}

void combine(void* combined_x,
             void* rdma_recv_x,
             int* rdma_recv_flag,
             void* rdma_send_x,
             const void* x,
             const topk_idx_t* topk_idx,
             const float* topk_weights,
             const int* src_info,
             const int64_t* layout_range,
             int* mask_buffer_ptr,
             int64_t* combine_wait_recv_cost_stats,
             int* next_clean,
             int num_next_clean_int,
             int num_combined_tokens,
             int hidden,
             int num_max_dispatch_tokens_per_rank,
             int num_topk,
             int num_experts,
             int rank,
             int num_ranks,
             bool use_logfmt,
             void* workspace,
             int num_device_sms,
             cudaStream_t stream,
             int phases,
             bool zero_copy,
             bool nvlink_pair_only) {
    constexpr int kNumMaxTopk = 11;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 32 / num_warp_groups;
    const int num_recv_per_sm = ceil_div(num_combined_tokens, num_device_sms);
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0 and num_recv_per_sm >= 0);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms =
        max(ceil_div(num_experts, num_warp_groups), num_recv_per_sm == 0 ? 1 : ceil_div(num_combined_tokens, num_recv_per_sm));

    // Check workspace
    auto atomic_clean_flag = static_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

    // Online cast cannot use zero-copy
    EP_HOST_ASSERT(not(zero_copy and use_logfmt));

    constexpr int kNumStages = 3;
    constexpr int kNumMaxUnrolls = 4;
    constexpr int kMaxNumGroups = 2;

    // Send buffer size
    const int num_meta_bytes = hidden / 128 * 4;
    const int num_send_tma_bytes = 32 * sizeof(int4) * kNumMaxUnrolls + 16;
    const int smem_send_size = num_warps * (kNumStages * num_send_tma_bytes + num_meta_bytes);

    // Receive buffer size
    const int num_recv_tma_bytes = 16 + hidden * 2;
    const int smem_recv_size = kMaxNumGroups * (kNumStages * num_recv_tma_bytes + hidden * 2 + kNumStages * num_meta_bytes * 3);

    // Total requirement
    const int smem_size = max(smem_send_size, smem_recv_size);

#define COMBINE_LAUNCH_CASE(hidden)                                                                                                \
    {                                                                                                                              \
        auto combine_func =                                                                                                        \
            use_logfmt ? combine<true, hidden, kNumMaxTopk, kNumMaxUnrolls> : combine<false, hidden, kNumMaxTopk, kNumMaxUnrolls>; \
        SET_SHARED_MEMORY_FOR_TMA(combine_func);                                                                                   \
        auto launch_combine_phase = [&](int launch_phases, bool stage_ce = false) { \
        auto chosen_func = stage_ce ? combine<false,hidden,kNumMaxTopk,kNumMaxUnrolls,true> : combine_func;                                                                        \
        LAUNCH_KERNEL(&cfg,                                                                                                        \
                      chosen_func,                                                                                                 \
                      combined_x,                                                                                                  \
                      rdma_recv_x,                                                                                                 \
                      rdma_recv_flag,                                                                                              \
                      rdma_send_x,                                                                                                 \
                      x,                                                                                                           \
                      topk_idx,                                                                                                    \
                      topk_weights,                                                                                                \
                      src_info,                                                                                                    \
                      layout_range,                                                                                                \
                      mask_buffer_ptr,                                                                                             \
                      combine_wait_recv_cost_stats,                                                                                \
                      next_clean,                                                                                                  \
                      num_next_clean_int,                                                                                          \
                      atomic_clean_flag,                                                                                           \
                      num_combined_tokens,                                                                                         \
                      hidden,                                                                                                      \
                      num_topk,                                                                                                    \
                      num_max_dispatch_tokens_per_rank,                                                                            \
                      num_experts,                                                                                                 \
                      rank,                                                                                                        \
                      num_ranks,                                                                                                   \
                      num_warp_groups,                                                                                             \
                      num_warps_per_group,                                                                                         \
                      launch_phases,                                                                                               \
                      zero_copy,                                                                                                   \
                      nvlink_pair_only);                                                                                           \
        };                                                                                                                         \
        if (split_rdma_first and (phases & LOW_LATENCY_SEND_PHASE)) {                                                               \
            launch_combine_phase(LOW_LATENCY_SEND_PHASE | kRDMAFirstPayloadPhase);                                                   \
            if (not ce_configured or not ce_skip_local_drain) launch_drain_rdma_before_pcie(rank, num_ranks, stream);                                                                \
        }                                                                                                                        \
        if (ce_configured and split_rdma_first and (phases & LOW_LATENCY_SEND_PHASE)) { \
            EP_HOST_ASSERT(not use_logfmt and not zero_copy); \
            ce_begin(rank,num_ranks,hidden,num_experts,num_max_dispatch_tokens_per_rank,stream); \
            launch_combine_phase(LOW_LATENCY_SEND_PHASE,true); \
            ce_finish(false,rdma_recv_x,rdma_recv_flag,mask_buffer_ptr,stream); \
            if (phases & LOW_LATENCY_RECV_PHASE) launch_combine_phase(LOW_LATENCY_RECV_PHASE); \
        } else launch_combine_phase(phases);                                                                                              \
    }                                                                                                                              \
    break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
#ifdef DISABLE_SM90_FEATURES
    const bool split_rdma_first = not nvlink_pair_only;
#else
    constexpr bool split_rdma_first = false;
#endif
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* mask_tensor) {
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_threads = num_sms * kNumThreads;
    const auto thread_id = sm_id * kNumThreads + static_cast<int>(threadIdx.x);
    for (int rank_id = thread_id; rank_id < num_ranks; rank_id += num_threads) {
        mask_tensor[rank_id] = mask_buffer_ptr[rank_id];
    }
}

void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* mask_tensor, cudaStream_t stream) {
    constexpr int num_sms = 1;
    constexpr int kNumThreads = 1024;
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, query_mask_buffer<kNumThreads>, mask_buffer_ptr, num_ranks, mask_tensor);
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void update_mask_buffer(int* mask_buffer_ptr, int rank_to_mask, bool mask) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    if (sm_id == 0 && thread_id == 0) {
        atomicExch(mask_buffer_ptr + rank_to_mask, mask ? 1 : 0);
    }
}

void update_mask_buffer(int* mask_buffer_ptr, int rank, bool mask, cudaStream_t stream) {
    constexpr int num_sms = 1;
    constexpr int kNumThreads = 32;
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, update_mask_buffer<kNumThreads>, mask_buffer_ptr, rank, mask);
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks) {
    auto thread_id = static_cast<int>(threadIdx.x);
    #pragma unroll
    for (int i = thread_id; i < num_ranks; i += kNumThreads)
        mask_buffer_ptr[i] = 0;
}

void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks, cudaStream_t stream) {
    constexpr int num_sms = 1;
    constexpr int kNumThreads = 32;
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, clean_mask_buffer<kNumThreads>, mask_buffer_ptr, num_ranks);
}

}  // namespace internode_ll

}  // namespace deep_ep
