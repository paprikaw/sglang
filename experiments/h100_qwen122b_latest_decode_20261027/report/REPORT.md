# Qwen3.5-122B-A10B FP8: latest EP2 / PP2 Decode comparison

同一 H100178 NVLink GPU0/1，完整 48 层模型。EP2 为 Attention DP2/TP1、MoE EP2，每卡 48 层、每层 128 个本地专家；PP2 为 24+24 层、每层 256 个本地专家，两个固定 B/2 流水槽。

B 是全局请求数：8、16、32 个重复语料请求，以及 32 个不同主题请求。沿用前一轮 35B 的输入构造；每请求 4096 个真实 GPU 缓存上下文 token，加 1 个新输入 token，生成 128 个输出 token。请求标记和输入 SHA256 校验相同，EP 两个 DP rank 与 PP 两个槽均使用相同的偶数/奇数分组。

两边保留原生 FP8 W8A8/DeepGEMM、BF16 KV 和残差、FP32 GDN 状态。35B 的 BF16 专家 Triton 替换不适用于这些 FP8 权重，没有对任何一边混用该替换。PP 开启 policy9 提前发射、独立消息通道、下一槽准备与发射；EP 保留共享专家重叠，并单独测试新两分块的同层 MoE 通信/计算重叠。原生全模型 TBO 未启用。

两边都开启原生 SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK；EP 使用原生 lazy ping-pong，PP 使用其原生非重叠调度的单 extra buffer。两边实际均按每请求 3 个 FP32 状态槽准入，保留 page64、EP 原生 overlap schedule 和相同的请求输入；未降低 GDN 状态精度。统一 mem_fraction_static=0.945；每卡 BF16 KV 池为 1.875GiB，EP 每卡 81,920 个全模型 KV token，PP 每卡 163,840 个半模型 KV token。池容量均足以容纳完整 B32 和输出余量。

每点先完成真实前缀准备及编译/Graph/通信格式预热，再做 3 轮完整预热和 5 轮关闭 profiler 的正式计时。另有独立两 rank profile；每 rank 丢弃前 20 个实际本地 B/2 Decode Graph 后选择 16 个。前缀准备、数值/输出一致性检查和 profile 均不计入性能结果，无 CPU KV restore。

吞吐为完整 cohort 输出 token 总量 / 从第一请求发出到最后输出到达的时间；TPOT 为每请求第一至最后输出到达间隔 /127，再对正式请求取平均。TTFT 为发出请求至第一个输出；E2E 为发出请求至最后一个输出。吞吐 SD 是 5 个 cohort 的样本标准差。

| Workload | Configuration | Output tok/s ↑ | SD | TPOT ms ↓ | TTFT ms | E2E ms |
|---|---|---:|---:|---:|---:|---:|
| b8_repeated | EP2 native + shared overlap | 600.12 | 7.97 | 11.358 | 263.20 | 1705.71 |
| b8_repeated | EP2 two-chunk overlap | 479.35 | 14.85 | 14.464 | 300.18 | 2137.04 |
| b8_repeated | PP2 lookahead | 599.62 | 0.55 | 12.360 | 133.62 | 1703.38 |
| b16_repeated | EP2 native + shared overlap | 1,081.46 | 8.21 | 12.574 | 295.26 | 1892.20 |
| b16_repeated | EP2 two-chunk overlap | 861.08 | 20.20 | 16.369 | 298.93 | 2377.77 |
| b16_repeated | PP2 lookahead | 972.21 | 0.90 | 15.358 | 150.24 | 2100.66 |
| b32_repeated | EP2 native + shared overlap | 1,742.06 | 6.79 | 16.094 | 303.82 | 2347.70 |
| b32_repeated | EP2 two-chunk overlap | 1,351.81 | 15.93 | 21.400 | 308.96 | 3026.78 |
| b32_repeated | PP2 lookahead | 1,585.96 | 0.98 | 18.920 | 171.42 | 2574.26 |
| b32_diverse | EP2 native + shared overlap | 1,323.02 | 2.25 | 21.900 | 311.21 | 3092.48 |
| b32_diverse | EP2 two-chunk overlap | 1,030.46 | 5.26 | 28.846 | 308.07 | 3971.51 |
| b32_diverse | PP2 lookahead | 1,219.14 | 0.75 | 25.017 | 172.92 | 3350.09 |

| Workload | PP vs native EP throughput | PP vs overlapping EP throughput | EP overlap vs native |
|---|---:|---:|---:|
| b8_repeated | -0.08% | +25.09% | -20.13% |
| b16_repeated | -10.10% | +12.91% | -20.38% |
| b32_repeated | -8.96% | +17.32% | -22.40% |
| b32_diverse | -7.85% | +18.31% | -22.11% |

## Expert, Attention and communication

单位为 µs/卡/实际层调用；EP 每卡 48 层，PP 每卡 24 层。Expert 仅 routed GEMM，Attention 仅 FA3/GDN core；量化、排列、共享专家及其他运算留在原始 profile。Dispatch/Combine kernel 包含等待。交集按同一 GPU 时间轴的区间求交，分母是该配置 Dispatch/Combine 的时间并集；这些分项不能相加作为端到端 TPOT。

| Workload | Configuration | Expert GEMM | Attention core | Dispatch | Combine | Comm ∩ GEMM | Comm overlapped |
|---|---|---:|---:|---:|---:|---:|---:|
| b8_repeated | EP2 native + shared overlap | 82.35 | 17.14 | 15.50 | 22.28 | 0.14 | 0.4% |
| b16_repeated | EP2 native + shared overlap | 85.12 | 27.44 | 16.13 | 23.28 | 0.18 | 0.5% |
| b32_repeated | EP2 native + shared overlap | 125.27 | 49.30 | 17.67 | 27.88 | 0.13 | 0.3% |
| b32_diverse | EP2 native + shared overlap | 244.86 | 49.30 | 17.42 | 31.57 | 0.13 | 0.3% |
| b8_repeated | EP2 two-chunk overlap | 127.57 | 16.74 | 32.39 | 39.08 | 11.50 | 15.9% |
| b16_repeated | EP2 two-chunk overlap | 159.68 | 27.31 | 33.36 | 44.12 | 17.38 | 22.2% |
| b32_repeated | EP2 two-chunk overlap | 211.68 | 49.14 | 37.16 | 48.94 | 26.49 | 30.3% |
| b32_diverse | EP2 two-chunk overlap | 374.76 | 49.24 | 44.16 | 56.70 | 38.18 | 37.1% |
| b8_repeated | PP2 lookahead | 88.98 | 15.66 | 0.00 | 0.00 | 0.00 | 0.0% |
| b16_repeated | PP2 lookahead | 115.53 | 26.25 | 0.00 | 0.00 | 0.00 | 0.0% |
| b32_repeated | PP2 lookahead | 147.34 | 48.12 | 0.00 | 0.00 | 0.00 | 0.0% |
| b32_diverse | PP2 lookahead | 316.62 | 48.18 | 0.00 | 0.00 | 0.00 | 0.0% |

## Observations

B8 的完整 cohort 吞吐基本持平；EP 原路径的稳态 TPOT 更低，但首 token 更慢，两者在 128-token 输出下的完整 cohort 吞吐接近。B16、B32 重复和 B32 多样输入，PP 相对 EP 原路径吞吐分别低 10.1%, 9.0%, 7.9%。

新增两分块重叠使四组吞吐下降 20.1%–22.4%。B32 重复时实际通信与 routed GEMM 的重叠比例达 30.3%，但 routed GEMM 从 125.3 增至 211.7 µs/层（+69.0%）。两分块将每层 routed GEMM 从 2 次变为 4 次；Attention core 基本相同。这一实现的额外专家执行和通信 kernel 时间超过了可用重叠收益，应保留原生 EP 路径作为当前更快选择。

相同全局 B32，重复改为多样语料时，EP 原路径 Expert GEMM 从 125.3 升至 244.9 µs/层；PP 从 147.3 升至 316.6 µs/层。Attention core 仍约 48–49 µs/层。该现象支持在部署决策中额外考虑输入与专家执行特性，仅用 batch 大小不能预测性能。

PP 的 Expert 与图间等待均需保留在成本模型中。以下图间空隙表提供测量证据，目前没有把所有剩余差距单独归因于 CPU、网络或某个调度步骤。

## Graph execution and gaps in independent profiles

同一 rank 相邻完整本地 B/2 Decode Graph 的平均 GPU span 和图间空隙。EP 每个 Graph 是 48 层，PP 每个 Graph 是单 stage 的 24 层，两个固定槽交替执行；不能直接把单个 PP Graph 当作请求 TPOT。图间空隙包含调度、阶段依赖和边界通信等等待，未细分为 CPU 或网络原因；这些值来自独立 profile，不与关闭 profiler 的 TPOT 直接相减。

| Workload | Configuration | Graph GPU span ms | Between-graph gap ms | Graph start interval ms |
|---|---|---:|---:|---:|
| b8_repeated | EP2 native + shared overlap | 11.625 | 0.094 | 11.713 |
| b16_repeated | EP2 native + shared overlap | 12.265 | 0.095 | 12.363 |
| b32_repeated | EP2 native + shared overlap | 15.701 | 0.100 | 15.790 |
| b32_diverse | EP2 native + shared overlap | 21.617 | 0.100 | 21.696 |
| b8_repeated | EP2 two-chunk overlap | 14.594 | 0.093 | 14.679 |
| b16_repeated | EP2 two-chunk overlap | 16.710 | 0.095 | 16.810 |
| b32_repeated | EP2 two-chunk overlap | 20.442 | 0.098 | 20.537 |
| b32_diverse | EP2 two-chunk overlap | 28.426 | 0.098 | 28.501 |
| b8_repeated | PP2 lookahead | 5.379 | 0.676 | 6.019 |
| b16_repeated | PP2 lookahead | 6.283 | 0.778 | 7.020 |
| b32_repeated | PP2 lookahead | 7.631 | 0.625 | 8.219 |
| b32_diverse | PP2 lookahead | 11.715 | 0.346 | 12.067 |

## Validation and provenance

每个 EP 配置、每种 workload，双 rank 各 48 个真实 MoE 层均比较同输入、同路由的原生/串行分块/重叠结果；原生/分块 relative L2≤1.5%，串行/重叠要求逐元素相同。PP 在计时前将原生调度与提前发射的完整 128-token 输出逐请求对照，正式轮次另检查固定分组、提前发射计数增加和所有权排空。

正式请求逐一验证 4096 个缓存 token、128 个输出 ID、SSE 计数 1–128 及由事件重算的 TPOT。23 项调度/通信所有权 CPU 测试和 12 项原生 GDN 分配/锁测试通过。完整模型 FP8 dtype、48/24 层、DeepGEMM backend、输入 SHA256、冻结文件、CPU 权重缓存使用和 Guard 清理/释放均已检查。

EP 首个探索运行在 B32 时被缓存检查拒绝，先前 B8/B16 部分数据未进入本表；no_buffer 尝试在参数校验阶段被拒绝，没有正式样本；后续低 KV 容量的 EP 运行也在 B32 缓存校验时被拒绝。PP 首次使用过大的自动 KV 池，在 typed NCCL 首次 P2P 缓冲分配时失败，无正式样本。最终统一显式 KV 预算并保留 FP32 状态、page64 和 EP overlap schedule；这些探索失败均未混入本表。

- native: [ep2_native_r5](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_qwen122b_latest_decode_20261027/runs/ep2_native_r5/launch_manifest.json)
- overlap: [ep2_overlap_r5](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_qwen122b_latest_decode_20261027/runs/ep2_overlap_r5/launch_manifest.json)
- pp: [pp2_lookahead_r6](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_qwen122b_latest_decode_20261027/runs/pp2_lookahead_r6/launch_manifest.json)

Engine commit: `0f2f1990994774472b1089d2b0682f0bf926a6b4`.

[Structured results](results.json) · [Throughput / TPOT](decode_comparison.png) · [B32 component costs](component_costs.png)
