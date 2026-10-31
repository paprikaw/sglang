# EP／PP 实验速览

更新：2026-10-31T15:54:53+11:00（墨尔本）。[全部实验索引](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/README.md)。本文只整理已有数据，没有启动新实验。

**目前最清楚的结论：固定 synthetic Decode 有“低并发单卡、高并发 EP4”的反转；AgentX 合并部署按平均请求 E2E 看，C8 偏单卡、C128 偏 PP2。P/D 分离和动态配置的净收益仍待同版本对照。**

| 需要关注的实验 | 用来回答什么 | 最短入口 |
|---|---|---|
| Synthetic 整模型 P／D | 同卡预算下，单卡、PP、EP 各适合什么工作区间？ | [固定阶段对照](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/ep4_fixed_phase_comparison_20261030/REPORT.md) |
| Expert／通信分项 | 计算粒度、专家数和通信域如何改变成本？ | [组件专题](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/ep_pp_tradeoff/REPORT.md) |
| AgentX 整模型 | 缓存、排队、分支和 P/D 干扰后，优势是否仍成立？ | [数据盘点与 benchmark 审计](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_data_audit_20261031/REPORT.md) · [详细结果](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_pd_dynamic_20261030/REPORT.md) |

口径：`P`＝每请求旧缓存，`Q`＝新增输入，`O`＝输出；`N`＝synthetic 全组请求数；`C`＝AgentX 根会话并发上限；`T`＝单卡收到的 token–expert 行数。**N、C、T 不能互换。** 下列整模型表均为三轮算术平均、绝对时间；组件源图保留单次事件及全部样本。

## 1. Synthetic：用来确定部署边界

主对照：完整 Qwen3.6-35B-A3B，BF16 权重／KV，FP32 GDN；每种布局固定单节点四张 H100。EP 的 Attention 为 TP1、对应 DP。下面只列 011 节点，不混合不同节点均值。

**Decode：固定并发、GPU 热缓存。** 每请求先建立 P4096，再以 Q1 生成 O128；重复文本带独立标识，忽略 EOS。旧 Prefill 不计时，HiCache 关闭，无 CPU 恢复，也无持续 Prefill 到达。N8 时四单卡各承担 2 请求，N64 时各承担 16 请求。

| 全组 N | EP4 TPOT ms | 2×EP2 | 4×单卡 | 2×优化 PP2 |
|---:|---:|---:|---:|---:|
| 8 | 6.122 | 6.534 | 4.752 | 5.495 |
| 16 | 6.740 | 7.435 | 5.996 | 6.700 |
| 32 | 7.781 | 8.735 | 7.890 | 8.625 |
| 64 | 9.220 | 10.683 | 10.310 | 11.181 |

低并发单卡更快，N32 接近，N64 偏 EP4；N8／N64 在另一节点五轮反向复测保留方向。完整扫描涵盖 P256–131072、N8–128，PP2 在 16 个工况中均未取得最低平均 TPOT。128K／N32 时单卡容量不足，PP2 整请求 E2E 2181.57 ms、EP4 2284.38 ms，但 PP2 的 TPOT 略慢。[完整 Decode 与容量审计](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/REPORT.md)

**Prefill：冷输入、O1。** P0，无缓存命中；下表为请求平均 TTFT，越低越好。PP2 使用可变批次原生 policy0，与上表固定 Decode 的优化 policy9 分开。

| N×每请求 Q＝总新输入 token | EP4 TTFT ms | 2×EP2 | 4×单卡 | 2×PP2 |
|---|---:|---:|---:|---:|
| 32×512＝16,384 | 219.11 | 205.88 | 163.39 | 146.06 |
| 16×8192＝131,072 | 720.38 | 687.48 | 501.32 | 424.94 |
| 8×16384＝131,072 | 834.77 | 786.91 | 567.40 | 483.94 |

三档均为 PP2 平均首 token 最快、单卡副本整批完成最快。例如 N16/Q8192，整批完成为单卡 706.53 ms、PP2 721.86 ms、EP4 1030.50 ms。首 token 与整批时间是两个目标。

另外保留两组背景数据：

- **Synthetic P/D 混合到达：** 64 请求，Q8192/O16 与 Q256/O256 按三种比例、16／32 请求每秒混合；各请求在同一实例完成 P 和 D。两节点各六档中四单卡平均 E2E 均最低，尚无布局混合的独有收益。[混合到达报告](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/pd_mix_pool_20261030/REPORT.md)
- **HiCache 自然／全 CPU 重放：** 35B 四卡五布局、P8256／130304，缓存起点会改变排序；两卡组拆分显示长前缀下单卡副本可能明显落后于 EP2／PP2。它与上面的纯 GPU 热 Decode 分开比较。[缓存实验及两卡拆分](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_five_layouts_hicache_20261028/two_gpu_three_layouts_analysis/REPORT.md)

GLM／HY3／122B、A100／H100 的早期 EP8↔PP2×EP4、EP2↔PP2 结果继续保存在[总索引历史表](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/README.md)，按原模型、精度、缓存和通信路径查看；本表的 35B 边界不直接外推。

## 2. 分项分析：用来解释，暂不用于确定切换阈值

| 数据 | 已有覆盖与用途 | 关键边界／入口 |
|---|---|---|
| 单层 Expert 二维图 | GLM-4.7、HY3、35B、122B；单卡 T×活跃专家数；两份独立测量，1584 格，每格 30 次事件 | 权重已驻留，含 GEMM、激活与 padding；不含 Attention、通信、CPU 权重加载。[数据](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_four_model_expert_surface_20261029/REPORT.md) |
| EP 通信二维图 | 四模型激活尺寸；EP2／4／8×全组原始 token 8–32768；624 条件，每点 30 事件 | BF16、DeepEP LL；EP2／4 节点内，EP8 跨两节点；超过 2048 token 分块，无计算 overlap。[数据](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_ep_token_comm_surface_20261029/REPORT.md) |
| 计算＋通信推导 | 估算“少通信时间－多算 Expert 时间”，定位待验证区域 | 旧图微批归一化曾让两侧复用同一 Expert 格；已核对，不能把预算正负视作实测胜负。[口径修正](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/ep_pp_tradeoff/REPORT.md) |

最重要的限制：35B 的 EP4 每卡应驻留 64 个专家，PP 负责层驻留全部 256 个；原 Expert 扫描始终驻留 256 个，尚缺真实分片对照。还缺实际路由、PP 交接／流水等待和暴露通信的联合测量。GLM／HY3 部署候选按最小 EP4 处理。

## 3. AgentX：用来检验实际回放中的端到端收益

[已有数据盘点与 benchmark 审计](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_data_audit_20261031/REPORT.md) 已重新核对主表21轮共34,371调用；另有7个历史点、21轮不重复测量（包括早期EP4混部／PD C2／C8的12轮）。早期480调用片段和仅准备的V4也独立索引，完整列出输入构造、回放方法、输出差异和协议限制。

完整 35B BF16，两个四卡 H100 节点、共 8 GPU；**各 rank 开 HiCache，CPU 32 decimal GB／rank，GPU KV 按 0.95 原生自动分配，会话内保留缓存。** 选取 8 个 AgentX 家族共 171 次调用；C128 用 16 份缓存隔离副本，共 2736 次调用。输入按 hash 重建前缀、保留长度，属于派生回放。

**已验收的合并部署基线：18 轮，E5 源码＋V3 回放。** P、D 在同一实例完成。以下 TPOT 为请求平均，延迟越低越好。

| 根会话并发 | 8 GPU 部署 | TTFT ms | TPOT ms/token | 请求 E2E ms | 压缩整轮 s |
|---|---|---:|---:|---:|---:|
| C8 | 2×EP4 | 431.71 | 9.640 | 3542.97 | 443.42 |
| C8 | 4×PP2 | 277.15 | 7.137 | 3266.10 | 441.65 |
| C8 | 8×单卡 | 348.46 | 4.990 | 2558.71 | 397.42 |
| C128 | 2×EP4 | 2271.32 | 73.435 | 22624.02 | 1168.21 |
| C128 | 4×PP2 | 1188.73 | 39.419 | 13128.90 | 835.80 |
| C128 | 8×单卡 | 27305.92 | 18.662 | 33312.39 | 1449.09 |

按平均 E2E，C8 单卡最好、C128 PP2 最好。C128 单卡虽 TPOT 最低，但 TTFT 很高；不能只按 Decode 速度选择布局。严格 SLO 下 EP4 仍有优势点，见[完整 SLO 与均值](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/matrix_analysis/REPORT.md)。

这些部署的缓存容量不同：GPU KV 总容量约单卡 271 万、PP2 1508 万、EP4 1781 万 token；单卡 GDN 预算 40、EP／PP 为完整模型等效 80。C128 的本地纯 Decode batch 均值仅约 EP4 5.62、PP2 5.04、单卡 2.21，不能把 C128 当成单步 N128。

**新完成的独立 P/D 点：E7 源码＋V3 回放，4P4D、P=EP4→D=EP4、C128。** 三轮共 8208 调用，完整验收及清理通过；GDN80、receive8，交接等待上限 300 s、整轮上限 7200 s。

| 配置 | TTFT ms | TPOT ms/token | 请求 E2E ms | 压缩整轮 s |
|---|---:|---:|---:|---:|
| EP4 P→EP4 D | 60765.86 | 11.170 | 65526.39 | 2409.73 |

该点请求平均 P 可运行队列等待 42.66 s，首 token 等待很长；P/D 等待有重叠，不相加，也不当纯网络时间。它验证了修复后能完整跑通，但缺同源码对照，不能与上表直接作受控性能归因。[三轮结果](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/e7_t300_b7200_analysis/REPORT.md) · [等待分解](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/eight_gpu_pd_ep4_e7_t300_b7200_r1/analysis/PD_WAITS.md)

**协议边界：** V3 有空闲压缩和分支时序限制；后续 V4 修正分支时序，须独立比较。当前尚无 V4 GPU 结果，原 24 轮交错队列已取消、准备文件保留。当前数据不是官方 AgentX 分数；不同布局的计时输出也未做到全部逐 token 相同。[协议与输出范围](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_pd_dynamic_20261030/REPORT.md)

## 4. 接下来只关注这三个问题

1. **合并部署的最好基线是什么？** 在同一 V4 回放、同一 8 GPU 预算下，先补全单卡、PP2、EP4 等固定配置；保留缓存容量、实际 batch、TTFT、TPOT、E2E 和任务完成时间。
2. **P/D 分离本身是否有净收益？** 再比较 P:D=6:2、4:4、2:6，每个配比内 P 侧保留单卡／PP2；D 侧先比较单卡副本，具备四卡时加入 EP4，长缓存保留 PP2／EP2 容量对照。均为待验证候选，单卡 D 尚无正式 AgentX 结果。
3. **是否需要动态配比？** 只有固定配置在同一目标下互有胜负，再测动态路由／配比相对最佳固定合并、固定 P/D、仅调整 P:D 的额外收益，计入缓存、排队、batch 分流与迁移成本。

后续新增结果优先更新本文对应表格，细节留在专题报告。[本文数据及来源清单](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/ep_pp_compact_20261031/manifest.json) · [重建脚本](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/ep_pp_compact_20261031/build_report.py)
