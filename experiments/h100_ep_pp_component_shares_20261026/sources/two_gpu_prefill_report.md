# H100 EP2／PP2：Prefill 的 Attention、Expert 和通信分项

## 请求和测量边界

**已缓存上下文**是计时／profile 开始前由真实模型生成、已驻留 GPU 的前缀；**新增 Prefill token**是本轮真正处理的输入。每条只输出 1 token。没有 CPU HiCache 换入、合成 KV 或 MTP。

完整 Qwen3.5-122B-A10B-FP8、节点 178 的 4 张 H100 80GB。EP2 用 GPU0/1，Attention TP1×DP2；PP2 用 GPU2/3，每阶段 24 层、Attention TP1、Expert EP1，两个等大槽位。两对均为 NVLink NV6，**不是跨节点通信实验**。继续使用原生 FP8 W8A8 DeepGEMM、FA3 全 Attention、Triton 线性 Attention，EP Prefill 使用 DeepEP normal intranode；原有 Decode Graph 保留，但不拿它计作 Prefill。PP Prefill 继续使用已验收的发送就绪策略。

本轮只补采 7 个 Prefill cell，各 1 次形状预热、1 次完整 trace。完整权重从既有 CPU Bank 加载；基础前缀准备和预热在 profiler 外。14 个逐卡 trace 均核验实际新增 token 总数、全部 Prefill chunk、完整层范围及缓存命中。Guard Bank 使用、科学结果与退出清理通过，GPU 租约已释放。

## Prefill 分项：毫秒／每卡／整批请求

按 CUDA kernel 的实际 CPU launch correlation 归属模块，再累计其 GPU 运行时长，**不是 CPU Python scope 的墙钟时长**。Attention 包含各自输入输出投影和模块内操作；Routed Expert 包含 `Fp8MoEMethod.apply` 内的量化、重排、激活和 grouped GEMM，Shared Expert 是单独共享 MLP；外部 router、共享 gate 等保留在其他。Dispatch／Combine 包括相应 DeepEP 内核及通知／布局内核，可以包含等其他 rank 的时间，**不是纯 NVLink 载荷传输延迟**。

EP 每卡处理一半请求并执行 48 层；PP 每阶段执行 24 层、跨两个槽位处理整批请求。因此这里比较同一全局工作量下的每卡工作，**不能把两个 PP 阶段的表项直接相加当作 TPOT**。不同流的内核可重叠；下表累计值也不能直接除以 TTFT 作为独占占比。

| 请求组成 | 配置 | 全 Attention 含投影 | 线性 Attention 含投影 | Routed Expert 计算 | Shared Expert 计算 | Dispatch | Combine | 其他内核 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8 条请求，每条无旧缓存，新增 4096 token | EP2，两卡均值 | 56.686 | 195.587 | 205.214 | 17.305 | 60.035 | 68.997 | 39.070 |
| 8 条请求，每条无旧缓存，新增 4096 token | PP2，第 1 阶段 | 56.554 | 196.078 | 319.918 | 17.483 | 0.000 | 0.000 | 25.437 |
| 8 条请求，每条无旧缓存，新增 4096 token | PP2，第 2 阶段 | 56.178 | 194.592 | 325.967 | 17.234 | 0.000 | 0.000 | 30.126 |
| 8 条请求，每条 GPU 已缓存 26752 token，新增 512 token | EP2，两卡均值 | 34.561 | 29.387 | 35.788 | 5.229 | 82.974 | 17.084 | 8.194 |
| 8 条请求，每条 GPU 已缓存 26752 token，新增 512 token | PP2，第 1 阶段 | 34.039 | 29.380 | 65.168 | 5.240 | 0.000 | 0.000 | 5.417 |
| 8 条请求，每条 GPU 已缓存 26752 token，新增 512 token | PP2，第 2 阶段 | 33.896 | 29.234 | 81.871 | 5.187 | 0.000 | 0.000 | 7.058 |
| 16 条请求，每条 GPU 已缓存 13376 token，新增 512 token | EP2，两卡均值 | 40.282 | 56.890 | 53.930 | 5.692 | 22.775 | 23.534 | 14.277 |
| 16 条请求，每条 GPU 已缓存 13376 token，新增 512 token | PP2，第 1 阶段 | 39.404 | 52.711 | 83.730 | 4.962 | 0.000 | 0.000 | 9.873 |
| 16 条请求，每条 GPU 已缓存 13376 token，新增 512 token | PP2，第 2 阶段 | 39.160 | 52.621 | 98.152 | 4.961 | 0.000 | 0.000 | 11.662 |
| 仅 PP：16 条请求，每条 GPU 已缓存 42176 token，新增 512 token | PP2，第 1 阶段 | 102.364 | 53.588 | 83.950 | 5.103 | 0.000 | 0.000 | 10.016 |
| 仅 PP：16 条请求，每条 GPU 已缓存 42176 token，新增 512 token | PP2，第 2 阶段 | 102.071 | 53.508 | 98.456 | 5.049 | 0.000 | 0.000 | 11.804 |

**PP 的 Dispatch／Combine 为 0，只表示各阶段 Expert EP1 没有跨卡专家交换，不表示 PP 总通信为零。** 阶段激活、元数据和反馈仍要通信。

## PP 边界与其他 NCCL：单列驻留，不冒充纯传输

| 请求组成 | 配置 | trace 中 NCCL kernel 累计毫秒，含等待／控制 |
|---|---|---:|
| 8 条请求，每条无旧缓存，新增 4096 token | EP2，两卡均值 | 0.127 |
| 8 条请求，每条无旧缓存，新增 4096 token | PP2，第 1 阶段 | 11.270 |
| 8 条请求，每条无旧缓存，新增 4096 token | PP2，第 2 阶段 | 3.372 |
| 8 条请求，每条 GPU 已缓存 26752 token，新增 512 token | EP2，两卡均值 | 0.271 |
| 8 条请求，每条 GPU 已缓存 26752 token，新增 512 token | PP2，第 1 阶段 | 3.751 |
| 8 条请求，每条 GPU 已缓存 26752 token，新增 512 token | PP2，第 2 阶段 | 0.640 |
| 16 条请求，每条 GPU 已缓存 13376 token，新增 512 token | EP2，两卡均值 | 0.103 |
| 16 条请求，每条 GPU 已缓存 13376 token，新增 512 token | PP2，第 1 阶段 | 1.664 |
| 16 条请求，每条 GPU 已缓存 13376 token，新增 512 token | PP2，第 2 阶段 | 0.977 |
| 仅 PP：16 条请求，每条 GPU 已缓存 42176 token，新增 512 token | PP2，第 1 阶段 | 1.387 |
| 仅 PP：16 条请求，每条 GPU 已缓存 42176 token，新增 512 token | PP2，第 2 阶段 | 1.050 |

该列没有把 PP 边界有效传输、等上游数据、元数据和 profiler 控制完全拆开，不能全部加入模型通信成本，也不能据此计算 NVLink 带宽。

## 与之前的 Decode 判断核对

下面沿用已完成 r7 的真实 Decode Graph 数据，单位改为 **毫秒／每卡／一次 Decode Graph**。Attention 这里只含 FA3／GDN 核心，Expert 这里只含 routed grouped GEMM，**范围比上面的完整模块窄，不能直接跨两表作 Prefill／Decode 比例**。

| 请求组成，均为每条新增 1、输出 128 | EP2 Attention 核心 | EP2 Routed Expert GEMM | EP2 Dispatch | EP2 Combine | Dispatch＋Combine／Routed GEMM |
|---|---:|---:|---:|---:|---:|
| 8 条，每条 GPU 已缓存 4096 token | 0.808 | 2.368 | 0.810 | 1.103 | 80.8% |
| 8 条，每条 GPU 已缓存 26752 token | 1.585 | 2.026 | 0.753 | 0.912 | 82.2% |
| 16 条，每条 GPU 已缓存 13376 token | 1.891 | 2.923 | 0.805 | 1.209 | 68.9% |

**该 Decode 判断的方向成立：路由 Expert GEMM 比 DeepEP kernel 累计更长。** 但通信仍达到 routed GEMM 的约 69%–82%，不能说可以忽略；完整共享专家、投影、调度及重叠尚不在这个比值内。PP 避免专家交换后，每层专家集中，计算开销上升，还保留阶段传输、反馈和调度依赖，因此不保证其端到端 Decode 更快。

## 本次 trace 能支持什么

- 同一全局 Prefill 工作量下，两种布局每卡 Attention 总成本接近；PP 的 Routed Expert 成本明显更高。它的层内通信节省必须与这个计算代价一起看。
- 8 条长缓存请求的 EP trace，第一轮有一个 Attention DP rank 为 IDLE、另一个只进入 512 个新增 token，随后才处理剩余请求。Dispatch 含等另一 rank／源批次的驻留，这一格不能当成固定带宽的纯通信曲线。该 cell 只有一次 trace，不能据此判定硬件拓扑的稳定通信延迟。
- 16 条请求、同样每条新增 512 token 时，将 PP 的每条缓存从 13376 增到 42176，全 Attention 成本约从 39 增至 102 毫秒／阶段；线性 Attention 约 53 毫秒、Expert 基本不变。这符合旧 KV 扫描主要影响全 Attention、而不增加本轮 Expert token 行数的解释。

## 诊断限制和证据修正

- EP 原生 overlap 在输出 1 token 的请求末尾仍提前执行一轮 Decode。本报告按 CPU step／Graph correlation 把它排除出 Prefill 表，并保留其逐卡 kernel 累计成本：
  8 条请求，每条无旧缓存，新增 4096 token：11.603 毫秒／卡，1 个额外 Decode step。
  8 条请求，每条 GPU 已缓存 26752 token，新增 512 token：12.879 毫秒／卡，1 个额外 Decode step。
  16 条请求，每条 GPU 已缓存 13376 token，新增 512 token：13.668 毫秒／卡，1 个额外 Decode step。
- 部分准备／profile 过程出现可恢复 CUDA allocator OOM 重试和延迟装载提示；所有请求完成、命中与输出长度均通过。只采用实际 GPU kernel 分项作短诊断，**不使用本轮带 profiler 的墙钟时间替换正式吞吐或 TTFT**。特别是长缓存 8 请求的单次 Dispatch 驻留不可外推。
- 一份 PP rank1 导出文件刚创建时的快照哈希为空文件哈希；其实际导出在 1.70 秒后完成、早于模型退出。原 manifest 保持不变；在已验证生产者退出后，重新验证完整 gzip CRC／JSON、全部 token／层调用、两次一致的文件哈希，写入 `validation.json` 的 `export_reconciliation`。后续采集脚本已修正为等完整 gzip／JSON 再登记哈希。
- 两组仍不是所有优化开关一致：EP 开原生 overlap，PP 为引擎不支持 overlap 的生产路径。这是当前可用的完整配置比较，不是单变量硬件实验；各 cell 仅一次 trace，没有尾延迟推断。

[正式 Prefill／Decode 三轮性能](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_h100_20261025/runs/h100_ep2_pp2_r7/analysis/REPORT.md) · [原 Decode 分项范围](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_h100_20261025/runs/h100_ep2_pp2_r7/analysis/component_followup/REPORT.md) · [逐卡 CSV](per_rank.csv) · [分配置 CSV](comparison.csv) · [核验、原始路径与哈希](validation.json)。

![Prefill 分项](prefill_components.png)
