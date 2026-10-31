# PP2 Decode：瓶颈定位与调度优化

当前 PP2 慢于 EP2 的主要可消除部分在模型 Graph 之间的依赖与 CPU 退休路径。保持相同 Expert GEMM，提前准备并发射下一槽位的真实 GPU 计算，可以把旧输出的 CPU 处理移出关键路径。

最后五轮对照：PP2 TPOT **9.303 → 6.998 ms/token**（降低 24.8%）；吞吐 **804.9 → 1048.3 output tok/s**（提高 30.2%）。相同通用 GEMM 优化的 EP2 为 **7.657 ms/token、890.4 tok/s**。

![Formal measurements and independent profile](pp_decode_scheduling.png)

## 正式测量

同一 H100178 GPU0/1 NVLink 对、同一 BF16 模型；全局 8 请求，固定两组 B4。每请求已有真实 GPU 驻留的 4096-token 前缀，新增 1 token，生成 128 token；temperature=0，ignoreEOS，无推测解码。每配置五个正式 cohort。TPOT 为首末输出间隔 / 127；吞吐为 1024 / cohort makespan，包含首 token 的影响。

| 配置 | TPOT ms/token | 五轮均值范围 ms | Output tok/s | TTFT ms | 请求 E2E ms |
|---|---:|---:|---:|---:|---:|
| EP2 / shared GEMM | 7.657 | 7.642–7.683 | 890.4 | 177.01 | 1149.43 |
| PP2 / shared GEMM | 9.303 | 9.231–9.536 | 804.9 | 88.07 | 1269.54 |
| PP2 / early activation | 8.793 | 8.775–8.831 | 849.4 | 86.18 | 1202.94 |
| PP2 / lookahead | 7.184 | 7.173–7.201 | 1023.1 | 85.88 | 998.22 |
| PP2 / wider lookahead | 6.998 | 6.989–7.013 | 1048.3 | 85.49 | 974.20 |

EP2 为 Attention DP2/TP1、40 层与 MoE EP2；PP2 为 Attention TP1、每阶段 20 层与 MoE EP1，两个流水槽位。固定 B4 成员在策略之间保持不变；等待八请求到齐的实验 admission 时间包含在客户端指标内。

## 为什么 GEMM 差距解释不了原来的 TPOT 差距

下面是独立 profile 中每 GPU、每层、B4 的均值，单位 µs；两张卡取均值。Expert GEMM 不含 EP 的布局适配和 dispatch/combine。

| 配置 | Expert GEMM µs | Attention core µs | 跨 GPU dispatch + combine µs |
|---|---:|---:|---:|
| EP2 / shared GEMM | 53.17 | 12.82 | 33.68 |
| PP2 / shared GEMM | 66.30 | 12.31 | 0.00 |
| PP2 / early activation | 66.53 | 12.34 | 0.00 |
| PP2 / lookahead | 66.37 | 12.28 | 0.00 |
| PP2 / wider lookahead | 66.39 | 12.27 | 0.00 |

PP 单层 Expert 比 EP 多约 **13.13 µs**；沿 40 层串行相加仅约 **0.525 ms**。这是量级比较，不是正式 TPOT 的精确归因。PP 整个模型 Graph 还省去了 EP 的逐层通信，所以判断端到端速度必须继续查看 Graph 外的反馈闭环。

| 配置 | 模型 Graph 合计 ms | 前向交接间隔 ms | 反馈至下一同槽位间隔 ms | 独立 profile token cycle ms |
|---|---:|---:|---:|---:|
| EP2 / shared GEMM | 7.580 | — | 0.093（EP Graph 间隔） | 7.674 |
| PP2 / shared GEMM | 6.110 | 2.676 | 2.260 | 11.046 |
| PP2 / early activation | 6.116 | 2.336 | 1.947 | 10.399 |
| PP2 / lookahead | 6.107 | 0.556 | 0.179 | 6.842 |
| PP2 / wider lookahead | 6.106 | 0.585 | 0.142 | 6.833 |

本表每 rank 丢弃前 20 个 Graph，再选 16 个 Graph；保守与宽窗口 lookahead 均已进入启用区间。原始早期窗口解析仍保留在 `analysis/shared_gemm_final_r8`，早期窗口中保守版本尚未完全启用，不能作为两种 lookahead 的稳态对比。

这些区间由同一请求组的 stage0 Graph 结束、stage1 Graph 开始/结束及下一同组 stage0 Graph 开始配对而成，包含 CPU 调度、等待对端、通信与另一 microbatch 的工作。NCCL kernel 的驻留时间也包含等待，不能据此认定 NVLink 本身很慢。Profiler 会改变 CPU 路径，表内 cycle 与上面的正式 TPOT 属于不同执行；正式均值还覆盖启动和结束保护区间，不能与 profile 相减作精确分解。

## 已实现的 PP 专属优化

1. **提前提交 activation。** 原生 forward 返回后即用同一个 producer event 提交原有 payload；不等旧 microbatch 的 CPU 结果退休。保留发送完成与 Graph buffer 复用的依赖。
2. **提前准备下一步 metadata。** 已收到的真实 GPU token 继续通过原生 future_map 输入；KV 位置和长度提前更新，旧结果处理保留独立 snapshot，避免重复分配与重复递增。
3. **下一槽位提前发射。** 在 CPU 等待旧输出 D2H 前执行下个槽位的真实 forward。正常调度走到该槽位时复用同一 result/event，避免第二次计算、发送和 FIFO 入队。按消息类型隔离三个 NCCL channel；保留事件与 buffer 生命周期依赖。

lookahead 使用原生 sampling、token relay、CUDA Graph 和通用 Expert GEMM；不预测 token 值。保守版本在稳定输出区间启用，宽窗口版本提前到已生成 2 token 后，并保留结束前与 Mamba/GDN 快照边界的保护。

共同稳态窗口中，保守与宽窗口版本的 Graph 和反馈周期接近；结合其唯一实现差别，这支持宽窗口的额外正式收益主要来自让更多启动/结束附近 token 使用同一优化路径，而不是让单次 Expert 计算变快。

## 优化过程与验证

[逐轮数据](optimization_history.csv) 保留每轮自己的 EP/PP 对照和计时范围。提前发送有效；单独取消 tail echo、拆分 channel、增加 activation stream 没有稳定收益；提前 metadata 有小幅收益；进一步把真实 forward 放在 CPU 退休前带来主要改善。不同轮的均值不拼接成同一实验。

异步 depth1 + 无类型共享 channel 的候选因消息顺序歧义发生停顿，已终止并清理，没有有效结果；另一个 typed-depth1 目录仅准备未运行。失败和未运行目录保留，不计入成功测量。

最终对照中所有正式 PP 配置的完整 128-token 序列逐位置相同；另有 untimed payload hash、事件/消息通道检查以及 prepared/lookahead 生产与消费计数一致、pending 为空的检查。12 项 CPU 调度/边界测试通过。正式计时关闭 profiler，输出/通信验证与 profile 单独执行。EP 的相同 GEMM 优化保持开启；本次没有为 PP 再改 GEMM。

该原型目前针对静态 PP2、固定 B4、无 stop/penalty/logprob/speculation 的稳定 Decode 做过完整验证；没有据此宣称动态到达与任意 batch 均可启用。冷 Prefill4096→1 为每布局三轮基线控制，本轮 lookahead 的性能结论仅针对 Decode。

## 数据和复现入口

- [最终冻结实验](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_graph_gap_opt_20261027/runs/shared_gemm_final_r8)；[冻结配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_graph_gap_opt_20261027/runs/shared_gemm_final_r8/config.json)；[私有调度实现](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_graph_gap_opt_20261027/runs/shared_gemm_final_r8/code/gap_schedule.py)。
- [正式指标](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_graph_gap_opt_20261027/analysis/shared_gemm_final_r8_steady/metrics.csv)；[组件](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_graph_gap_opt_20261027/analysis/shared_gemm_final_r8_steady/components.csv)；[逐 token profile](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_graph_gap_opt_20261027/analysis/shared_gemm_final_r8_steady/profile_cycles.csv)；[完整解析及来源哈希](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_graph_gap_opt_20261027/analysis/shared_gemm_final_r8_steady/results.json)。
- 源码 commit `870269702285d94bdd926a50cc29ad1a20bcf60c`；模型 revision `995ad96eacd98c81ed38be0c5b274b04031597b0`；调度改变只在实验私有 adapter，冻结 engine 未改。
- 分析：`python3 /data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_graph_gap_opt_20261027/analyze_gap.py --run /data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_graph_gap_opt_20261027/runs/shared_gemm_final_r8 --out /data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_graph_gap_opt_20261027/analysis/shared_gemm_final_r8_steady --drop-graphs 20 --forwards 16`；绘图/报告：`/home/bxb1/conda/bin/python /data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_graph_gap_opt_20261027/build_report.py --analysis /data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_graph_gap_opt_20261027/analysis/shared_gemm_final_r8_steady`。
- 每个已完成实验均有 Guard postcheck、owned cleanup 和冻结输入哈希验证；大日志与原始 trace 在原实验目录保留，归档保存引用和 SHA256。
