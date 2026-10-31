# H100 EP Prefill: MoE-only communication / Expert overlap

主结果：同为8192-token预算，B32重复输入的重叠EP达到 90,066.4 input tok/s，相对原生EP 88,448.4 提升 1.83%；请求E2E从 990.63 降至 970.08 ms。同预算PP2原生为 103,701.9 input tok/s，仍更快。时间线证实了通信和Expert计算的重叠，但分块导致的GEMM效率损失和资源争用抵消了大部分收益。

完整 Qwen3.6-35B-A3B，BF16 权重/激活/KV、FP32 GDN 状态；H100178 同一 NVLink GPU0/1。EP2 是 Attention DP2×TP1、MoE EP2；PP2 是两个20层阶段、两个流水槽。本轮每个 Attention 实例的 Prefill 预算：EP 8192、PP 8192 token。Attention/GDN 保持整批执行，仅在 EP 的 MoE 内分成两组。

每个请求缓存前缀为0、新输入4096 token、生成1 token；每策略预热3轮，正式5轮交错比较，profiler关闭，每轮清空KV/前缀缓存。吞吐为整批新输入 token 数除以整批完成时间；E2E为从客户端发起请求到首个且唯一输出 token 的请求均值。因此本实验没有 Decode TPOT。独立 profile 的单位是每卡、每次实际 Prefill 层调用的 GPU kernel 微秒，不能加总成请求 E2E。

p0 保留原生 EP；p1 将 MoE token 分两组串行执行；p2 使用相同两组并交错 Dispatch/GEMM/Combine；p3 在 p2 上将Expert GEMM限制为112个SM，为使用20个SM的DeepEP通信留出资源，H100共132个SM。两组各有独立通信 handle，两个输入就绪事件都在第一个 GEMM 前记录。原有 shared-expert overlap 保持开启；SGLang 全模型 TBO 开关保持关闭。实验路径要求两侧每步输入均至少128 token；混合Prefill/Decode和空闲尾部保留原生路径。所有EP策略在每次完整forward开始时通过CPU广播固定策略，开销计入各策略端到端时间。

## b8_q4096_repeated: 8 requests × 4096 new tokens

| Configuration | Input tok/s ↑ | Sample SD | Request E2E ms ↓ | vs native EP |
|---|---:|---:|---:|---:|
| EP2 native | 69,739.7 | 407.6 | 431.83 | +0.00% |
| EP2 serial split | 69,052.8 | 77.8 | 436.85 | -0.98% |
| EP2 overlap | 70,093.4 | 311.2 | 429.57 | +0.51% |
| EP2 overlap / reserved SM | 71,572.1 | 155.3 | 420.06 | +2.63% |
| PP2 native | 89,807.5 | 317.3 | 255.59 | +28.78% |
| PP2 early send | 91,177.1 | 231.4 | 253.52 | +30.74% |

| Configuration | Routed GEMM µs ↓ | Dispatch µs | Combine µs | Attention µs | Comm ∩ GEMM µs | Comm overlapped |
|---|---:|---:|---:|---:|---:|---:|
| EP2 native | 628.01 | 620.32 | 2098.53 | 1472.70 | 0.00 | 0.0% |
| EP2 serial split | 740.81 | 593.93 | 476.85 | 1469.37 | 0.00 | 0.0% |
| EP2 overlap | 808.90 | 689.31 | 518.44 | 1469.86 | 189.46 | 15.7% |
| EP2 overlap / reserved SM | 778.19 | 685.78 | 524.02 | 1468.82 | 285.65 | 23.6% |
| PP2 native | 1180.90 | 0.00 | 0.00 | 1505.02 | — | — |
| PP2 early send | 1116.87 | 0.00 | 0.00 | 1481.91 | — | — |

EP2 overlap：相对串行分块吞吐 +1.51%；相对原生 EP +0.51%。实际时间轴上，15.7% 的通信 kernel 时间与 routed GEMM 相交；每层 GEMM 相对原生变化 +28.8%。

EP2 overlap / reserved SM：相对串行分块吞吐 +3.65%；相对原生 EP +2.63%。实际时间轴上，23.6% 的通信 kernel 时间与 routed GEMM 相交；每层 GEMM 相对原生变化 +23.9%。

## b32_q4096_repeated: 32 requests × 4096 new tokens

| Configuration | Input tok/s ↑ | Sample SD | Request E2E ms ↓ | vs native EP |
|---|---:|---:|---:|---:|
| EP2 native | 88,448.4 | 234.8 | 990.63 | +0.00% |
| EP2 serial split | 82,856.3 | 616.8 | 1058.85 | -6.32% |
| EP2 overlap | 86,105.9 | 440.3 | 1014.70 | -2.65% |
| EP2 overlap / reserved SM | 90,066.4 | 74.8 | 970.08 | +1.83% |
| PP2 native | 103,701.9 | 247.4 | 715.10 | +17.25% |
| PP2 early send | 102,825.5 | 3,651.6 | 732.87 | +16.25% |

| Configuration | Routed GEMM µs ↓ | Dispatch µs | Combine µs | Attention µs | Comm ∩ GEMM µs | Comm overlapped |
|---|---:|---:|---:|---:|---:|---:|
| EP2 native | 727.13 | 598.71 | 528.65 | 1474.61 | 0.00 | 0.0% |
| EP2 serial split | 926.27 | 606.23 | 511.16 | 1473.79 | 0.00 | 0.0% |
| EP2 overlap | 1042.52 | 754.39 | 582.95 | 1473.46 | 326.15 | 24.4% |
| EP2 overlap / reserved SM | 992.00 | 755.69 | 595.06 | 1473.94 | 503.06 | 37.2% |
| PP2 native | 1192.07 | 0.00 | 0.00 | 1517.19 | — | — |
| PP2 early send | 1119.44 | 0.00 | 0.00 | 1499.45 | — | — |

EP2 overlap：相对串行分块吞吐 +3.92%；相对原生 EP -2.65%。实际时间轴上，24.4% 的通信 kernel 时间与 routed GEMM 相交；每层 GEMM 相对原生变化 +43.4%。

EP2 overlap / reserved SM：相对串行分块吞吐 +8.70%；相对原生 EP +1.83%。实际时间轴上，37.2% 的通信 kernel 时间与 routed GEMM 相交；每层 GEMM 相对原生变化 +36.4%。

## b32_q4096_diverse: 32 requests × 4096 new tokens

| Configuration | Input tok/s ↑ | Sample SD | Request E2E ms ↓ | vs native EP |
|---|---:|---:|---:|---:|
| EP2 native | 85,846.3 | 5,398.7 | 1030.45 | +0.00% |
| EP2 serial split | 81,918.5 | 2,589.2 | 1077.18 | -4.58% |
| EP2 overlap | 86,115.2 | 56.3 | 1016.00 | +0.31% |
| EP2 overlap / reserved SM | 89,740.1 | 123.8 | 974.00 | +4.54% |
| PP2 native | 100,549.1 | 5,658.3 | 754.22 | +17.13% |
| PP2 early send | 101,445.9 | 6,494.5 | 751.66 | +18.17% |

| Configuration | Routed GEMM µs ↓ | Dispatch µs | Combine µs | Attention µs | Comm ∩ GEMM µs | Comm overlapped |
|---|---:|---:|---:|---:|---:|---:|
| EP2 native | 737.97 | 599.46 | 508.66 | 1476.66 | 0.00 | 0.0% |
| EP2 serial split | 942.49 | 607.50 | 504.87 | 1474.08 | 0.00 | 0.0% |
| EP2 overlap | 1066.55 | 759.65 | 579.74 | 1473.56 | 344.27 | 25.7% |
| EP2 overlap / reserved SM | 1007.49 | 758.48 | 594.76 | 1474.00 | 513.58 | 37.9% |
| PP2 native | 1217.49 | 0.00 | 0.00 | 1519.96 | — | — |
| PP2 early send | 1141.90 | 0.00 | 0.00 | 1504.74 | — | — |

EP2 overlap：相对串行分块吞吐 +5.12%；相对原生 EP +0.31%。实际时间轴上，25.7% 的通信 kernel 时间与 routed GEMM 相交；每层 GEMM 相对原生变化 +44.5%。

EP2 overlap / reserved SM：相对串行分块吞吐 +9.55%；相对原生 EP +4.54%。实际时间轴上，37.9% 的通信 kernel 时间与 routed GEMM 相交；每层 GEMM 相对原生变化 +36.5%。

## Why the first 4096-token split was slower

独立第一轮仍使用双方4096-token Prefill预算，EP拆成两组2048 token。完整5轮、双rank profile、Guard与清理审计均通过；未与8192预算样本混合。

| Configuration | Input tok/s | Mean request E2E ms | GEMM µs/layer/card | Dispatch+Combine µs | Comm∩GEMM µs |
|---|---:|---:|---:|---:|---:|
| EP2 native | 76,017.6 | 1014.28 | 485.51 | 589.28 | 0.00 |
| EP2 serial split | 60,694.9 | 1279.95 | 675.98 | 635.27 | 0.00 |
| EP2 overlap | 63,408.5 | 1226.31 | 714.90 | 734.55 | 144.60 |
| EP2 overlap / reserved SM | 66,601.9 | 1166.35 | 705.73 | 747.25 | 232.75 |
| PP2 native | 81,372.2 | 896.68 | 881.27 | 0.00 | 0.00 |
| PP2 early send | 84,447.0 | 835.00 | 880.70 | 0.00 | 0.00 |

预留SM重叠相对原生EP吞吐 -12.39%。真实重叠已发生，但更小的Expert输入增加GEMM时间，通信和计算同时执行也增加了各自的kernel耗时。因此第二轮增大双方的通用Prefill预算，使EP每个Expert子批仍有4096个源token。8192预算下应与同预算原生EP比较，不能把通用批次增大的收益全部归因于重叠。

## Independent diverse-input recheck

主矩阵多主题输入的原生EP有一个慢样本，因此另起独立实验，仅比较原生EP与预留SM重叠。8192预算、B32、相同输入/精度/路由，每策略3次冷预热及10次交错正式计时，profiler关闭；未与主矩阵5轮合并，复用主矩阵的独立profile作为机制证据。

| Configuration | Input tok/s mean | Sample SD | Median | E2E ms |
|---|---:|---:|---:|---:|
| EP2 native | 88,431.8 | 2,351.3 | 89,189.6 | 986.33 |
| EP2 overlap / reserved SM | 89,325.9 | 2,301.7 | 90,298.5 | 983.42 |

独立10轮均值提升 +1.01%，中位数提升 +1.24%；全部样本保留。主矩阵的4.54%均值增益受原生EP慢样本影响，不能作为稳定收益；当前证据支持约1%的小幅收益。

## Validation and artifacts

两rank各40层真实权重验证通过：同分块串行/重叠输出逐元素相同=True；原生/分块最大 relative L2=0，串行/预留SM最大 relative L2=0。数值验证在计时外执行；GEMM 形状变化使用此前相同的1.5%逐层 L2 阈值。这验证了当前输入上的执行等价性，不是完整语言模型质量评测。

| Workload | Configuration | All formal first tokens equal to native EP |
|---|---|---|
| b8_q4096_repeated | EP2 native | True |
| b8_q4096_repeated | EP2 serial split | True |
| b8_q4096_repeated | EP2 overlap | True |
| b8_q4096_repeated | EP2 overlap / reserved SM | True |
| b8_q4096_repeated | PP2 native | True |
| b8_q4096_repeated | PP2 early send | True |
| b32_q4096_repeated | EP2 native | True |
| b32_q4096_repeated | EP2 serial split | True |
| b32_q4096_repeated | EP2 overlap | True |
| b32_q4096_repeated | EP2 overlap / reserved SM | True |
| b32_q4096_repeated | PP2 native | True |
| b32_q4096_repeated | PP2 early send | True |
| b32_q4096_diverse | EP2 native | True |
| b32_q4096_diverse | EP2 serial split | True |
| b32_q4096_diverse | EP2 overlap | True |
| b32_q4096_diverse | EP2 overlap / reserved SM | True |
| b32_q4096_diverse | PP2 native | True |
| b32_q4096_diverse | PP2 early send | True |

21项CPU检查、全部冻结输入SHA256、每轮输入/零缓存/输出计数、双rank通信handle排空、canonical CPU权重缓存使用、Guard postcheck、所有自有GPU进程清理和租约释放均通过。正式计时与独立 profile 分开；5轮标准差用于显示波动，不作为服务尾延迟统计。

[Frozen launch manifest](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_prefill_overlap_20261027/runs/overlap_budget8192_r5_fair/launch_manifest.json) · [source/config](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_prefill_overlap_20261027/runs/overlap_budget8192_r5_fair/config.json) · [raw metrics and trace manifests](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_prefill_overlap_20261027/runs/overlap_budget8192_r5_fair) · [recomputed results](./results.json)

先前失败包括缓存提供者源码声明不完整、空闲DP rank的分块边界，以及混合阶段收尾通信超时；没有可用性能点。保留其原始failed/queued状态与失败说明，未混入正式结果。

GPU验证源码标签：`experiment/sglang/qwen35b-prefill-ep-overlap-20261027-h100-qualified`，commit `7960005f41afe0d3b2bad9d69b0433f9f885c088`。[EP overlap implementation](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/qwen35b-prefill-ep-overlap-r3-20261027/python/sglang/srt/layers/moe/prefill_overlap.py) · [GPU qualification](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_prefill_overlap_20261027/runs/overlap_budget8192_r5_fair/gpu_qualification.json)
