# 实验总览：性能、容量、组件特性与优化记录

本文件是全部实验的统一总结与检索入口，不限于 EP8 或 PP2×EP4。按实验类别持续收录完整推理、KV Cache 容量与缓存恢复、单层 Attention／Expert、通信拓扑及调度优化；每项保留负载、计时口径、简短报告和原始数据入口。当前已收录并核验的实验如下，后续实验继续加入对应表格。

## 专题报告索引

所有入口均为 Markdown，可直接查看结论、表格与原始数据来源。后续新增结果继续加入本表或下方对应实验类别。

| 专题 | 内容 | 文档 |
|---|---|---|
| AgentX 2P6D／GDN 状态槽 | PP2 P＋EP4/EP2 D 与 PP2 P＋3×EP2 D；槽预算、运行结果、C8 混部／4P4D 对照及单卡优势分解 | [实验方案与结果](agentx_2p6d_20261031/REPORT.md) · [C8 六配置对照](agentx_2p6d_20261031/C8_COLOCATED_COMPARISON.md) · [单卡为何更快](agentx_2p6d_20261031/SINGLE_VS_PD_DIAGNOSIS.md) |
| AgentX：HiCache 总命中为何不同 | 同一 E5/V3 的 18 轮逐请求核对；32 GB 实际拆分、KV/GDN 容量、子请求热点与整段前缀丢失 | [专项核查](agentx_hicache_hit_audit_20261031/REPORT.md) <!-- qwen35b_h100_agentx_hicache_hit_audit_20261031 --> |
| AgentX workload 画像与回放方法 | 官方 full／256k 与当前八族的覆盖差异；会话大小、分支、长尾、缓存命中口径及预取指标缺口 | [负载探索报告](agentx_workload_exploration_20261031/REPORT.md) |
| AgentX C8／C128 workload 与缓存分布 | 21 个已审计 episode；逐请求／逐对话 context、新增 Prefill、Decode、GPU／CPU 命中与延迟分布 | [统计报告](agentx_workload_profile_20261031/REPORT.md) |
| EP／PP 实验速览（建议先读） | Synthetic P／D、Expert／通信分项、AgentX E2E 三条实验线；关键绝对均值、缓存口径、已验收结果与待验证问题 | [Compact 文档](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/EP_PP_EXPERIMENTS_COMPACT.md) |
| AgentX 数据盘点与 benchmark 审计（建议先读） | 21轮、34,371调用请求级复算；已有点、实际回放方法、历史去重、输出差异及其他协议限制 | [盘点与审计](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_data_audit_20261031/REPORT.md) |
| AgentX修复与两节点验证（阶段记录） | 8卡EP4、PP2、单卡18轮合并基线；E7 EP4 4P4D另3轮已验收，同版本控制与动态收益待验证 | [进展报告](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_pd_dynamic_20261030/REPORT.md) |
| AgentX 数据与早期兼容性初筛（10/30） | 官方数据下载核验；480次HiCache片段请求；旧引擎的GDN P/D缓存限制及独立合并核查，后续修复见阶段记录 | [历史专题](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_pd_feasibility_20261030/REPORT.md) |
| Expert 计算与 EP／PP 通信权衡 | 四模型 H100 二维数据、EP2／4／8 通信、单层与微批比较口径核对 | [专题总结](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/ep_pp_tradeoff/REPORT.md) |
| EP4明确并列：固定P／D配置是否互有胜负 | 复核143条已有四卡均值；Prefill偏单卡／PP，P4096的Decode N8→N64由单卡／PP转向EP4，另一节点五轮复测；混合流量另列 | [EP4、EP2、单卡、PP2完整对照](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/ep4_fixed_phase_comparison_20261030/REPORT.md) |
| 研究口径修正：真实P/D分离＋布局类别配比 | 单卡／纯PP统一为EP=1类，与EP比较；先验证真实P→D交接收益，再验证动态配比增量。上轮请求类别隔离不是P/D分离 | [修正说明、已有证据与最小对照](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/pd_disaggregation_scope_20261030/REPORT.md) |
| 35B Prefill／Decode 混合：PP／EP 配比机会 | 两个四卡H100节点；六档到达流量、三档冷Prefill、混合池与请求隔离；135组有效三轮均值。PP首token可更快，四单卡平均E2E仍最优，关键SLO收益被单卡分流覆盖 | [简短结论、绝对数、完整表与审计](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/pd_mix_pool_20261030/REPORT.md) |
| 35B 单节点 Decode：动态配比是否必要 | N8–128、P256–131072；加入单卡和 EP2 基线，五轮反向复测；PP2 在16个工况均未取得最低平均 TPOT，长缓存整请求有约4.5%小收益 | [结论、绝对均值、理想选型收益与审计](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/REPORT.md) |
| 35B 小 batch 四卡验证（新增实测） | 两个 H100 节点各有 EP4 对照；优化 2×PP2、4×单卡、原生 PP4；N8／16／32，每点五轮平均 | [绝对延迟、反转幅度与原始轮次](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/small_batch_layout_verify_20261029/REPORT.md) |
| 35B 四卡五布局 | EP4、2×EP2、4×单卡、2×PP2、PP4；三轮平均绝对时间，缓存协议分别比较 | [结果表与逐轮来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/four_gpu_layout_retrieval_20261029/REPORT.md) |
| P／D 阶段内布局反转 | 已有证据、生成尾段与整批指标、Prefill 证据缺口 | [证据核查](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/phase_reversal_audit_20261029/REPORT.md) |
| 优化后回归 Benchmark | 三模型、A100／H100、P／D；固定矩阵与逐项覆盖状态，以文档日期为准 | [Benchmark 清单](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/BENCHMARK.md) |
| 优化实现记录 | 通信库、Attention、PP 调度、缓存加载的改动、收益及限制 | [优化记录](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/OPTIMIZATIONS.md) |
| 研究规划与相关工作 | 实例路由机制、PP／EP 张力的文献核查、原地切换可行性 | [机制规划](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/MOE_INSTANCE_ROUTING_DESIGN_20261029.md) · [文献核查](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/PP_EP_TENSION_LITERATURE_AUDIT_20261029.md) · [切换评估](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/EP_PP_INPLACE_SWITCH_FEASIBILITY_20261029.md) |

索引更新：2026-10-31；下列历史表保留原实验日期与口径。历史请求结果见第 2–5 节；通信扫描见第 6 节（含本轮 T01）；单层 Attention／Expert 扫描见第 7 节；单节点双卡补充见第 8 节 Q01–Q09；H100／A100跨规模通信域对照见第9节Q10–Q11。

**先说清楚：不是同一个 workload 测出了互相矛盾的结果，而是输入长度、缓存起点、模型、CUDA Graph 和 MTP 都有变化。** 下表每行是一组可追踪的历史条件，不能跨行直接归因。

## 1. 我们到底测过哪几类东西？

| 类型 | 用大白话说是在测什么 | 对应数据 |
|---|---|---|
| 从头 Prefill | 没有旧缓存，读完 8192 个输入 token，只生成 1 个 token | P01–P07、P10 |
| 短输入 Decode | 先读 128 个输入 token，再生成 512 个；不是长缓存实验 | D01–D06、D04b |
| 有缓存的 Prefill／Decode | CPU HiCache 里已有每条请求的旧 context；计时后搬到 GPU，再处理新输入／生成 | Hy3：D07–D08、P08–P09；GLM 长缓存：D09 |
| 语义缓存 Decode／MTP | 有真实语义的旧 context，生成 64 个 token；比较不开 MTP、旧 Verify、修复 Verify，以及 PP 调度优化 | D10–D12、O01–O04 |
| 纯通信／单层 Attention、Expert | 不运行完整推理，只测 Dispatch／Combine、Attention 或 Expert 内核及其维度扫描 | 通信见第 6 节；单层扫描见第 7 节；不能当成请求 TPOT |

`B`＝全局请求数；`P`＝**每条请求已经缓存的** context token 数；`Q`＝每条请求本次新增输入；`O`＝生成 token 数；总旧 context `M=B×P`。**P2304 不是又做了一次 2304-token Prefill。**

第 2–4 节的 EP8／PP2×EP4 对照均为两节点／8 GPU：EP8 的 Attention 在节点内 TP4×DP2，Expert 跨节点 EP8；PP2×EP4 每节点负责一段层，每段 Attention TP4、Expert EP4。**两个独立 EP4 副本（2×EP4）是第三种配置，不能叫 EP8。**

## 2. Decode：已有的 EP8 与 PP2×EP4 性能

每个数字均按 **EP8 / PP2×EP4** 排列。TTFT、E2E 单位秒，TPOT 单位毫秒，吞吐为 output token/s。延迟取请求均值；D10–D12 展示两个正式 cohort 的均值。点击 ID 看完整来源，详细分位数在 [metrics.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/metrics.csv)。

| ID／模型、GPU | workload：B；P；Q→O | 运行条件 | 吞吐 | TTFT | TPOT | E2E | 有分项 profile 吗？ |
|---|---|---|---:|---:|---:|---:|---|
| [D01：GLM / H100](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_h100_expert_tp8_ep8_pp2ep4_20261007/metrics.csv) | 4；0；128→512 | Graph 开；早期两组 native 库不同 | 130.60 / 166.37 | 0.246 / 0.208 | 29.94 / 23.44 | 15.55 / 12.18 | 没有完整 Decode 分项；F02 是 Prefill |
| [D02：GLM / H100](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_h100_expert_tp8_ep8_pp2ep4_20261007/metrics.csv) | 16；0；128→512 | 同 D01 | 429.39 / 518.09 | 0.487 / 0.255 | 35.47 / 29.58 | 18.61 / 15.37 | 同上 |
| [D03：GLM / A100](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_ep8_pp2ep4_optimized_20261008_r4/full_metrics.csv) | 4；0；128→512 | Graph 开；旧 SM-PCIe 小包策略 | 106.26 / 111.20 | 0.441 / 0.411 | 36.65 / 35.04 | 19.17 / 18.32 | 无 D/C/Expert 分项 |
| [D04：GLM / A100](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_ep8_pp2ep4_pair_rdma_20261010_r1/full_metrics.csv) | 4；0；128→512 | Graph 开；pair NVLink＋其余 RDMA | 100.10 / 101.88 | 0.436 / 0.410 | 39.03 / 38.34 | 20.38 / 20.00 | 无 |
| [D04b：GLM / A100](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_pcie_prefill_20261010_r2/five_conditions.csv) | 4；0；128→512 | Graph 开；RDMA；统一 local chunk 后的另一轮 | 100.06 / 101.92 | 0.438 / 0.413 | 39.05 / 38.29 | 20.39 / 19.98 | 无 |
| [D05：GLM / A100](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_ep8_pp2ep4_ce_decode_20261010_r1/full_metrics.csv) | 4；0；128→512 | **Graph 关**；CE-PCIe | 18.66 / 19.71 | 0.572 / 0.338 | 213.52 / 202.54 | 109.68 / 103.84 | 有路径证明，无分项耗时 |
| [D06：GLM / A100](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_ep8_pp2ep4_ce_decode_20261010_r1/full_metrics.csv) | 4；0；128→512 | **Graph 关**；与 D05 同 native 的 RDMA 对照 | 22.02 / 22.12 | 0.433 / 0.405 | 180.97 / 180.25 | 92.91 / 92.51 | 同上 |
| [D07：Hy3 BF16 / A100](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/hy3_stream_cached_measure_v4_20261020/comparison.csv) | 8；2304；1→128 | Graph 关；合成 HiCache；两组容量够用 | 46.74 / 45.03 | 0.334 / 0.302 | 169.83 / 176.28 | 21.90 / 22.69 | 没找到此完整模型的 D/C/Expert 分项 |
| [D08：Hy3 BF16 / A100](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/hy3_stream_cached_measure_v4_20261020/comparison.csv) | 8；3712；1→128 | 同上；**EP8 每节点只能同时驻留 3 条，发生排队** | 23.49 / 45.81 | 5.747 / 0.315 | 170.04 / 173.15 | 27.34 / 22.30 | 同上；不能把吞吐差全算作通信差 |
| [D09：GLM / A100](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_ep8_pp2ep4_m655360_b8_decode_20261022_r1/full_metrics.csv) | 8；81920；1→128 | Graph 关；合成 HiCache；**EP8 和 PP 都放得下** | 38.88 / 39.10 | 2.439 / 1.746 | 188.16 / 192.06 | 26.34 / 26.14 | 有另跑的 GPU 热缓存短诊断 F03，不是同一 cohort |
| [D10：GLM / A100](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_semantic_bank_fourway_20261023_v5/runs/fourway_reuse_r3) | 8；4096；1→64 | Graph 开；语义 HiCache；MTP 关 | 84.94 / 83.53 | 0.533 / 0.496 | 87.21 / 89.06 | 6.03 / 6.11 | 有另跑的 F04：A、routed/shared Expert、D、C、TP |
| [D11：GLM / A100](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_semantic_bank_fourway_20261023_v5/runs/fourway_reuse_r3) | 同 D10 | MTP 开；**旧 Verify 后端，保留作修复前记录** | 96.50 / 88.28 | 0.553 / 0.502 | 69.72 / 77.18 | 4.95 / 5.36 | F04；包含旧的 Verify Attention 高耗时 |
| [D12：GLM / A100](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_mtp_verify_fixed_ep8_pp2ep4_20261023/RESULTS.md) | 同 D10 | MTP 开；修复 Verify；PP 原始 46/46 分层 | 131.59 / 128.56 | 0.557 / 0.504 | 46.74 / 52.79 | 3.50 / 3.83 | F05：目标验证、草稿分别拆开 |

**怎么读这张表：** D05/D06 才是同一 native、同为无 Graph 的 CE/RDMA 对照；不能拿 D05 与 Graph 已开启的 D03 相减来证明“PCIe 慢了几倍”。D09 的 P81920 不能与 D10 的 P4096 混为同一长 context workload。

## 3. Prefill：同样有 EP8 与 PP2×EP4 对比，但不能拿来解释 Decode 占比

吞吐为本次**新输入** token/s；TTFT 单位秒；仍按 **EP8 / PP**。以下均为 eager Prefill、输出 1 token，因此 TPOT 不适用，E2E 与 TTFT 几乎相同，精确值保留在 CSV。

| ID／模型、GPU | B；P；Q | 策略／差别 | 新输入吞吐 | TTFT | 分项 profile |
|---|---|---|---:|---:|---|
| [P01：GLM / H100](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_h100_expert_tp8_ep8_pp2ep4_20261007/metrics.csv) | 4；0；8192 | 早期 native 不同；EP/PP local chunk=512/1024 | 3706.71 / 12336.44 | 6.520 / 1.683 | 后续 F02；local Q 不同 |
| [P02：GLM / H100](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_h100_expert_tp8_ep8_pp2ep4_20261007/metrics.csv) | 16；0；8192 | 同上 | 3840.99 / 12480.09 | 19.136 / 5.605 | 无对应 B16 分项 |
| [P03：GLM / A100](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_ep8_pp2ep4_optimized_20261008_r4/full_metrics.csv) | 4；0；8192 | 旧小包 PCIe 策略；大 Prefill 实际走 RDMA | 1326.41 / 1937.77 | 18.372 / 10.740 | 无 |
| [P04：GLM / A100](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_ep8_pp2ep4_pair_rdma_20261010_r1/full_metrics.csv) | 同上 | 关闭非 NVLink 的 PCIe；local chunk 仍不同 | 1324.78 / 1946.36 | 18.369 / 10.691 | 无 |
| [P05：GLM / A100](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_pcie_prefill_20261010_r2/five_conditions.csv) | 同上 | 同 local chunk1024；**强制旧 SM-PCIe 大包** | 206.20 / 485.59 | 118.932 / 42.835 | 仅路径证明 |
| [P06：GLM / A100](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_pcie_prefill_20261010_r2/five_conditions.csv) | 同上 | 同 local chunk1024；RDMA 对照 | 1322.77 / 1943.53 | 18.254 / 10.710 | 无 |
| [P07：GLM / A100](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_ep8_pp2ep4_ce_prefill_20261010_r4/full_metrics.csv) | 同上 | **CE Copy Engine PCIe**；native c3405cad | 1635.45 / 4092.66 | 14.731 / 5.079 | CE 计数证明启用，不是分项耗时 |
| [P08：Hy3 / A100](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/hy3_stream_cached_measure_v4_20261020/comparison.csv) | 8；2304；256 | 合成 HiCache；计入 H2D | 1381.21 / 1769.84 | 1.066 / 0.806 | 无完整模型分项 |
| [P09：Hy3 / A100](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/hy3_stream_cached_measure_v4_20261020/comparison.csv) | 8；3712；256 | 同上 | 1325.83 / 1616.17 | 1.104 / 0.845 | 同上 |
| [P10：GLM / A100](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_ep8_pp_attribution_20261010_r1/full_metrics.csv) | 4；0；8192 | 统一 local chunk1024 的归因轮；RDMA | 1326.03 / 1936.28 | 18.203 / 10.749 | **有 F01；生产性能与 profile 分开采样** |

旧“05d 优化版”不能解释成 CE：D03/P03 实际 native 为 b0c15ccb；D04/P04/P05/P06/P10 为 9dc34887；CE 轮为 c3405cad。所谓“RDMA”仍保留 pair 内 NVLink，不是所有流量都走网卡。P07 的原始吞吐包含归档尾部，CSV 另存 HTTP 完成边界的吞吐，未偷换口径。

## 4. 想看 Dispatch、Combine、Expert 各花多久，打开哪份？

`Dispatch`＝把 token 发给专家；`Combine`＝把专家结果送回并合并；`routed Expert`＝被路由命中的专家计算；`shared Expert`＝共享专家计算。**不是所有报告里的“Expert”都只包含 routed Expert。**

| Profile ID／直接入口 | 测的东西 | Dispatch / Combine 能分开看？ | Expert 能看什么？ | 数字的单位／限制 |
|---|---|---|---|---|
| [F01：A100 真 Prefill 单层](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_ep8_pp_attribution_20261010_r1/analysis/root_layer_comparison.csv) | P10 的独立诊断；local Q1024，P1024/5120/7168 | **能** | routed、shared 分开，另有 Attention、TP | ms／每卡／一次 MoE 层；不同 P 单列 |
| [F02：H100 真 Prefill 单层](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_h100_ep8_pp_attribution_20261008/layer_metrics.csv) | 层20/50、P2048；EP local Q512，PP Q1024 | 已发布层表只有 **D+C 合计**；未核验 trace 能否再拆 | routed 单列；router/shared 在合并项 | ms／采样层／rank；不是相同 local Q 对照 |
| [F03：A100 长缓存 Decode](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_a100_cached_decode_gpu_activity_20261022/analysis_output/gpu_r1/summary.json) | B8、P81920、Q1、**O32，GPU 已热** | **能**；`mean_phase_union_ms.deep_ep.dispatch/combine` | `expert_ffn` 合计，没有 routed/shared 拆分 | 按实际 GPU 活动分类；分项可重叠；不是 D09 的 O128 冷恢复 |
| [F04：语义 Decode 四组](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_semantic_bank_fourway_20261023_v8/runs/event_profile_followup_r1) | D10/D11 的补采；MTP 关／旧 MTP 开 | **能**；`exclusive_ms.deepep_dispatch/combine` | routed、shared、dense 分开 | ms／实际 Graph；target、draft 分开；PP 按 stage 分开 |
| [F05：修复 Verify 后](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_mtp_verify_fixed_ep8_pp2ep4_20261023/runs/paired_r1) | D12 的目标验证／草稿 | **能** | 同 F04 | 同上；不是 ms／输出 token |
| [F06：PP50/42 层平衡后](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_pp_stage_balance_20261023/runs/balance_r1/pp2ep4_balanced/stage_activity_summary.json) | O01；PP 两槽 B4；目标验证／草稿 | **能** | 同 F04 | EP8 参考来自 D12，不是这轮重测 |

上述分项已汇总到 [profiles.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/profiles.csv)：49 行，每行都带原始路径、哈希和计量单位。F04–F06 的逐卡 JSON 文件名均为 `stage_activity_summary.json`。

**两个防误读规则：** Prefill 的“13.26 / 7.66 ms 通信”是平均单 MoE 层；语义 Decode 的“89.91 ms Graph”包含整段模型，不能放在一列比较。PP 两 stage 的耗时也不能直接相加就叫 TPOT，MTP 一轮更可能产出多个 token。

## 5. 还有一些常被混到一起的历史数据

| 数据／入口 | 实际比较了什么 | 当前应怎样使用 |
|---|---|---|
| [H100 九点容量表](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_h100_hicache_grid_strict9_20261016_v2/strict9_metrics.csv) | **2×EP4 vs PP2×EP4，不含 EP8**；M=229376/393216/655360 × B=8/32/128；P=M/B；Prefill Q256→1，Decode Q1→128 | 36 条配置×workload 记录，含全套均值/中位数/P99；合成 CPU HiCache 满、GPU 空；没有九点逐点 D/C/Expert profile |
| [A100 长缓存容量点](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_m655360_b8_hicache_cold_20261015_r2/full_metrics.csv) | **2×EP4 vs PP**；B8、P81920；Prefill/Decode | 与 D09 有相同请求形状，但对手不同；不能把 2×EP4 的容量瓶颈自动套到 EP8 |
| [A100 冷输入 32K 容量对照](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_pp_kv_crossover_20261008_r2/full_metrics.csv) | 2×EP4 vs PP 单槽/双槽；B16；先处理长输入再生成 | 冷启动容量实验；不是预先准备好 HiCache 的纯 Decode |
| [H01：早期到达流实验](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_2xep4_heavy_20261004_r2/comparison.csv) | Prefill：65条到达，P8192/Q4096；Decode：18条到达，P8128/Q64→256；含 EP8/PP/2×EP4 | Prefill EP8仅完成58/65、PP65/65；吞吐1274.31/1865.96，E2E77.68/42.56s。**Decode PP仅完成13/18（censored）**；不能作完整 cohort 排名；固定窗口吞吐也不是饱和容量 |
| [H02：修正 cap8 后](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_pp2ep4_decode_cap8_20261004_r1/ep8_comparison.csv) | 12 条到达请求；P8128、Q64→256；EP8 vs PP；非一次 B12 同时入场 | 完整结果吞吐29.75/29.94，TPOT189.26/189.15 ms；无已核验的 D/C/Expert 分项 |
| [旧网页汇总及历史口径](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiment-report-hub/dist/reports/decode-ep8-pp2ep4-cross-model-20261022-v1/metrics.csv) | 还收录 10/13–14 的旧缓存恢复协议、H100 短输入 2×EP4，以及部分单槽 PP | 这些行也保留在本目录 CSV，标为相关/历史；统一容量结论优先用 strict9；网页不含后来的 D09–D12 |

### 同一个语义 workload 上的 PP 优化，不是新增 workload

以下始终是 B8、逻辑 P4096、Q1→64、MTP 开。EP8 参考 D12：**131.59 tok/s、46.74 ms TPOT**；下列多数只重测 PP，不能冒充新一轮 EP8/PP 配对。

| 优化记录 | PP 吞吐 / TPOT | 有什么可看 |
|---|---:|---|
| [O01：原始 → 本地反馈 → 50/42 分层](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_pp_stage_balance_20261023/RESULTS.md) | 128.60/52.82 → 132.59/50.98 → 140.73/47.40 | 完整 Graph 分项 F06、反馈和执行时间线 |
| [O02：反馈包合并，同进程 A/B/B/A](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_pp_feedback_payload_ab_20261024/REPORT.md) | 141.01/47.29 → 141.88/46.99 | 未加外层 fence 的调度诊断；不是再采一次完整 A/E/D/C 消融 |
| [O03：复用 CPU 序列信息，同进程 A/B/B/A](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_pp_seq_cpu_reuse_20261024/REPORT.md) | 141.47/47.13 → 142.86/46.64 | 同上；原始逐 cohort 均值已收录，未伪造合并 P99 |
| [O04：Prefill layer Graph 尝试](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_pp_prefill_layer_graph_20261024_v13/runs/layer_event_ab_r1/pp2ep4_prefill_graph_ab/arm_result.json) | TPOT46.67→46.58 ms；有性能文件 | **profile 写盘因 EDQUOT 失败，整轮未验收**；不纳入正式 CSV 主结果，不能说启动等待已解决 |

这里的逻辑 P4096，实际 CPU Bank 缓存4032，首轮还重算64个旧 token，加上 Q1 共65个新计算位置。因此“有旧 context”不等于首轮完全没有 Prefill/Extend。所有 MTP 数据都是性能验证，**尚未通过严格的无损生成等价性资格验证**。

## 6. 只想研究通信／Expert 曲线，别误拿完整推理结果

| ID／入口 | 测了什么 | 没有测什么 |
|---|---|---|
| [T01：A100 · GLM／Hy3 · EP4／EP8 · CE／RDMA](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/single_layer_ep4_ep8_matched_20261024/ce_rdma_r1/summary.md) · [80 行比较表](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/single_layer_ep4_ep8_matched_20261024/ce_rdma_r1/transport_comparison.csv)／[Dispatch、Combine 分项](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/single_layer_ep4_ep8_matched_20261024/ce_rdma_r1/phase_metrics.csv)／[曲线](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/single_layer_ep4_ep8_matched_20261024/ce_rdma_r1/transport_curves.png) | 本轮新增：整批唯一 token 从 8 逐次翻倍到 4096；分别比较 EP8 整批／EP4 每次半批，以及单次 token 相同。065／167 同型 ConnectX-6 Dx；eager，每点正式 3 次，共 280 点。整批 4096、EP4 每次 2048 token 时，065 上 GLM 的通信 RDMA／CE 为 17.7161／2.4376 毫秒；全部形状与节点见报告。 | **只有通信，没有 Expert GEMM、专家权重、Attention 或 KV Cache。** 专家数／top-8 只用于构造路由。CE 在节点内大消息上明显受益，但跨节点 EP8 收益不稳定；不能直接解释成完整 PP 加速或 Prefill／Decode 延迟。 |
| [T02：A100 固定总量 EP2→EP4](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/deepep_ep2_ep4_a100_20261027/REPORT.md) · [通信图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/deepep_ep2_ep4_a100_20261027/latency.png)／[D、C 分项](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/deepep_ep2_ep4_a100_20261027/components.png) <!-- begin deepep_ep2_ep4_a100_20261027 --><!-- end deepep_ep2_ep4_a100_20261027 --> | A100065，固定全组 8/16/32/256/1024/2048 token，BF16 H5120、160 experts、top8；双卡 NVLink／四卡混合、RDMA-only 对照；小消息 eager/Graph、大消息 eager，同一 low-latency API。72 点，两轮每点 30 次。实际扩卡小消息 1.64–3.19 倍，大消息 9.91–21.64 倍；统一 RDMA 后大消息约 1.47–1.50 倍。 | 只有 Dispatch／Combine，无 Expert GEMM、Attention、模型或请求 TPOT。四卡跨出 NVLink 对。GPU 测量及清理均完成；原 Guard 汇总失败原样保留，离线重算 6480 rank samples 通过。 |
| [T03：H100 固定总量 EP2→EP4](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/deepep_ep2_ep4_h100_20261027/REPORT.md) · [NVLink 图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/deepep_ep2_ep4_h100_20261027/latency.png)／[D、C 分项](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/deepep_ep2_ep4_h100_20261027/components.png)／[A100 对照](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/deepep_ep2_ep4_h100_20261027/a100_h100_domains.png) <!-- begin deepep_ep2_ep4_h100_20261027 --><!-- end deepep_ep2_ep4_h100_20261027 --> | H100178 四卡两两 NV6；全局 token、路由和协议匹配 T02。小消息 Graph EP4/EP2 为 1.233/1.236/1.073 倍；大消息 eager 为 0.578/0.394/0.361 倍。统一 RDMA 后大消息为 1.265/1.251/1.235 倍。72 点，两轮每点 30 次。 | 纯 Dispatch／Combine，无模型或 Expert GEMM。H100 扩卡仍在 NVLink 域内，A100 T02 跨 NVLink 对；SM90/SM80 native 不同。独立审计、Guard postcheck、清理及释放通过。 |
| [H100 EP4/EP8 纯通信表](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_h100_ep8_pp_attribution_20261008/pure_communication/summary.md) | T4/8/16 的 Graph 通信、T512/1024 的 eager 通信；D+C、RS/AG 和独立 PP 边界 | 没有 Attention/Expert 计算、CPU 调度或真实流水气泡 |
| [Hy3 A100 EP4/EP8 通信归档](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/hy3_a100_ep4_ep8_comm_20261020_r1) | 与 Hy3 形状关联的通信微基准；以该目录协议为准 | 不是 D07/D08 的完整模型分项 profile |
| [Decode trade-off 派生分析](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_decode_tradeoff_20261024/REPORT.md) | 引用已测 Expert 曲线和 Graph 分项，估算通信节省与 Expert 代价 | **CPU 离线计算，没有新跑 GPU，也不是新的吞吐观测**；原始曲线来源见 provenance |

本轮及后续纯通信扫描以表内 ID 检索；[通信扫描清单](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/communication_sweeps.json)保存 T01 的数据路径、哈希、token 范围、后端和计时边界。**T01 不包含 Expert 计算；本轮按相同路由补测的真实 Expert 计算见第 7.2 节 E06，历史数据见 E02／E04／E05。不同计时边界的组件不能直接相加冒充完整层观测。**

## 7. 补充：单层 Attention / Expert 的维度扫描

这里保留此前的 **14 组归档**，并补入本轮单层 Expert 扫描 **E06**；这些均不是第 2/3 节的完整推理结果。除特别注明，下面的延迟均为 **ms、median**，Graph 表示 CUDA Graph；已缓存前缀长度和本轮新增输入量都按每条请求计算。表中的扫描范围不表示所有变量做了完整笛卡尔积，未测格子不能插值当实测。

### 7.1 Attention：context、新 token、batch、布局与请求组成

以下都测 post-RoPE BF16 Attention core，**不含 QKV/O 投影、TP/NCCL 通信、Expert、HiCache 换入和调度**。单卡实验使用 GLM 的 TP4 单 rank 几何（24 Q heads、2 KV heads、D128）；四卡实验按对齐迭代的 rank 最大值统计。

| ID／GPU、入口 | 扫描了哪些维度 | 可以直接引用的结果 | 计时和解释边界 |
|---|---|---|---|
| [简短报告 A01](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/A01.md) · A100 单卡；[CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_attention_context_prefix_profile_20261005_r1/full_metrics.csv)／[热图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_attention_context_prefix_profile_20261005_r1/cold_latency_heatmap.png) | 每次只测一个请求的一层 Attention。处理后的总长度取 512、2048、4096、8192、16384 token；其中一部分前缀已经缓存在 GPU，剩余部分是本轮需要处理的输入。两者相加等于处理后的总长度，所以不是两个独立扫描轴。共 30 个数据点。 | 例如：处理后总长度为 16384 token，GPU 已缓存前缀为 15360 token，则本轮处理 1024 个新增输入 token，Attention 核心耗时中位数为 2.8247 毫秒；完全没有已缓存前缀、处理全部 16384 个输入 token 时为 17.4090 毫秒。 | 剩余输入多于 1 token 时测 Prefill/Extend 内核；只剩 1 token 时测 Decode 内核。这里只测一次单层调用，不是生成一串输出。计时前清理的是 L2 缓存，不是删除 GPU 中的前缀 KV。 |
| [简短报告 A02](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/A02.md) · A100 单卡；[CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_attention_cached_context_prefill_profile_20261005_r1/full_metrics.csv)／[缓存前缀与新增输入热图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_attention_cached_context_prefill_profile_20261005_r1/latency_heatmap.png) | 每次测一个请求。GPU 已缓存前缀分别为 0、1024、4096、8192、12288 token；本轮新增的 Prefill 输入分别为 16、64、128、256、512、1024、2048、4096 token。两个维度独立组合，共 40 个点，全部使用 Prefill/Extend 内核，不测 Decode。 | 已缓存前缀固定为 12288 token 时，新增输入从 16 增到 512 token，耗时从 1.0957 增到 1.1561 毫秒：输入量增加 32 倍，耗时增加约 5.5%。 | 前缀 KV 在计时前已经位于 GPU；没有 CPU→GPU 换入。计时前清理 L2，不删除前缀 KV。CUDA Graph 关闭；仅测 Attention 核心。 |
| [简短报告 A03](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/A03.md) · A100 四卡；[CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_attention_tpdp_efficiency_20261006_r2/full_metrics.csv)／[对比图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_attention_tpdp_efficiency_20261006_r2/core_latency_by_workload.png) | 同一组请求分别使用四个单卡 Attention 副本、两个双卡 Attention 副本、一个四卡 Attention 副本。测试组合覆盖 1、2、4 个请求，已缓存前缀 1024、12288、131072 token/请求，以及本轮处理 1、16、512 token/请求；只有表中实际列出的 8 种请求组合，不是全部组合。 | 一个请求、GPU 已缓存前缀 12288 token、新增 Prefill 输入 512 token 时，三种布局的耗时依次为 4.4892、2.2533、1.1571 毫秒。四请求的已测组合中，布局差异不超过 2.55%。 | 本轮处理 1 token 的点使用 Decode 内核，16/512 token 使用 Prefill/Extend 内核。CUDA Graph 开启。一个请求分配给一个副本时，其他副本确实没有请求；计时不含张量并行通信。 |
| [简短报告 A04](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/A04.md) · A100 四卡；[CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_attention_long_context_batch_20261006/full_metrics.csv)／[batch 曲线](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_attention_long_context_batch_20261006/attention_batch_scaling.png) | 每个请求在 GPU 已缓存 131072 token 的前缀；本轮每个请求只做一次 Decode，处理 1 个当前 token。不做长输入 Prefill。比较三种节点内布局，第一轮测 4/8 个请求，第二轮重新测 8 个请求再测 16 个请求。 | 同一轮内，请求数从 4 增到 8，核心处理率增加 77.7%–82.1%；从 8 增到 16 再增加 57.5%–62.6%。到 16 个请求仍未确认处理率平台。 | CUDA Graph 开启。两轮的 8 请求基线分别测量，不能跨轮混算。只保留一层 KV，不代表完整模型能驻留同样数量的长请求。 |
| [简短报告 A05](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/A05.md) · A100 TP4；[CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_attention_pair_batch_20261016_r1/full_metrics.csv)／[报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_attention_pair_batch_20261016_r1/summary.md) | 比较 GPU 已缓存前缀为 10000 token 的单请求、已缓存前缀为 20000 token 的单请求，以及两者同时合批。每个请求本轮分别新增 64 或 128 个 Prefill 输入 token；不是 Decode。 | 每请求新增 64 token 时，两条单请求分别耗时 0.8980、1.7725 毫秒；合批耗时 1.7638 毫秒。合批时间约等于较慢的那条请求，而不是两条时间之和。 | 四卡 Attention 张量并行，CUDA Graph 开启，前缀 KV 已在 GPU。单请求耗时相加是派生比较，不是直接测量的两次串行调用。 |
| [简短报告 A06](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/A06.md) · A100 TP4；[19 点 CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_attention_headroom_model_20261016_r1/predictions.csv)／[曲线](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_attention_headroom_model_20261016_r1/headroom_latency_throughput.png) | 固定前缀 KV 已在 GPU，主要测每请求新增 128 个 Prefill 输入 token，另有新增 256 token 的控制点。请求数在 1–16 范围选点；前缀包含均匀 15000 token、混合 10000/20000 token、混合 1000/29000 token，以及每请求 49152 token。 | 每请求已缓存 49152 token、新增 128 token 时，3 请求为 4.3438 毫秒，4 请求为 4.3407 毫秒。增加一条请求基本没有增加单层耗时；这不是完整模型的吞吐或容量结论。 | 四卡 Attention 张量并行、CUDA Graph 开启。7 个点校准模型，12 个点留出验证；测量耗时与预测耗时分列，不把预测当实测。 |

### 7.2 Expert：实际配对行数、活跃专家数与权重工作集

**活跃专家数不是常驻专家数。** 在固定常驻权重下，把相同数量的实际 token–专家配对行分到更多专家，不会增加理想有效计算量，但会改变每专家输入行数、补齐、权重复用和调度成本。一行表示一个 token 在一个专家上的一次计算，不是一个请求；同一个 token 命中多个专家时会形成多行。所有扫描均衡分配，不测路由倾斜。

| ID／模型、入口 | 扫描了哪些维度 | 可以直接引用的结果 | 计时和解释边界 |
|---|---|---|---|
| [简短报告 E01](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/E01.md) · GLM-AWQ / A100 单卡；[CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_ep4_local_expert_profile_20261005_r2/full_metrics.csv)／[二维热图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_ep4_local_expert_profile_20261005_r2/latency_heatmap.png) | 单张 A100 常驻 GLM 的 40 个专家，使用真实 INT4 权重。实际交给本 GPU 专家计算的 token–专家配对行数为 8、32、128、512、2048、8192；活跃专家数为 1、2、4、8、16、32、40，保留 37 个有效组合。 | 实际配对行数不超过 128 且活跃专家不超过 16 时，耗时约 0.25 毫秒。固定配对行数，分散到更多专家通常会增加成本；并非增加了同样倍数的有效计算量。 | 这里一行表示一个 token 在一个专家上的一次计算；同一个 token 命中多个专家会形成多行。不是请求数或唯一 token 数。旧测试从四份物理来源构造输入；CUDA Graph 关闭，不含 Attention 或通信。 |
| [简短报告 E02](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/E02.md) · GLM-AWQ / A100 单卡；[完整 CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_ep8_pp2ep4_expert_compute_20261021_r2/full_metrics.csv)／[EP/PP 曲线](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_ep8_pp2ep4_expert_compute_20261021_r2/expert_compute.png) | 只用一张 A100 模拟两种布局的局部专家计算：EP8 每卡常驻 20 个专家；两阶段流水、每阶段 EP4 时每卡常驻 40 个专家。全局新增逻辑 token 数取 8、32、128、512、2048、4096；流水配置将每批拆成两个等大微批次，两个槽均测量。 | 全局新增逻辑 token 为 32 时，CUDA Graph 下 EP8 局部计算耗时 0.1997 毫秒，流水配置两个槽耗时中位数的均值为 0.2847 毫秒。有效 FLOPs 相同不保证分块与 padding 成本相同。 | 这是旧四份来源复制协议：两种布局每次本卡实际配对行数都为全局逻辑 token 数的四倍，不代表生产服务有四倍唯一 token。没有前缀缓存、Attention、网络或流水调度；不能区分完整请求的 Prefill/Decode 性能。 |
| [简短报告 E03](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/E03.md) · GLM-AWQ / A100 单卡；[CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_expert_weight_footprint_20261022_r1/full_metrics.csv)／[权重工作集曲线](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_expert_weight_footprint_20261022_r1/expert_footprint.png) | 单卡常驻 40 个 GLM INT4 专家。固定本 GPU 的实际 token–专家配对数为 128、512、8192，分别分配到 2、4、8、10、20、40 个活跃专家。其余常驻专家没有输入，不会重新加载整组权重。 | 实际配对数为 128 时，活跃专家从 2 增到 40，CUDA Graph 耗时从 0.0829 增到 0.3564 毫秒；配对数为 8192 时则从 2.2902 增到 2.6501 毫秒。 | 这里只测专家核心，不测请求、前缀、Attention 或网络。本次配对行数不含旧来源复制倍数；重复运行偏向热权重状态，权重文件字节数不是实测显存流量。 |
| [简短报告 E04](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/E04.md) · Hy3 BF16 / A100 单卡；[CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/hy3_expert_surface_20261022_r2/full_metrics.csv)／[二维图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/hy3_expert_surface_20261022_r2/surface.png)／[24/48专家配对](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/hy3_expert_surface_20261022_r2/matched_comparison.csv) | 单卡常驻 48 个 Hy3 BF16 专家。实际本地配对行数为 8、32、128、512、2048、8192；活跃专家数为 2、4、8、12、24、48，共 32 个有效主网格点。另有常驻 24/48 专家的配对控制，以及低延迟接口的数据布局适配测试。 | 48 个专家活跃时，配对行数从 128 增至 512，CUDA Graph 耗时只从 1.4013 增至 1.4730 毫秒。固定 128 行，常驻 24 专家的配对控制耗时为 0.7593 毫秒，常驻 48 专家为 1.4106 毫秒。 | 真实 BF16 权重、当前 A100 Triton 默认配置。主表展示核心网格，完整表保留布局控制；适配测试也不含网络。没有请求前缀，也不是完整 Prefill/Decode。 |
| [简短报告 E05](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/E05.md) · Hy3 vs GLM / 同一 A100 GPU0；[CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/hy3_glm47_expert_doubling_20261022_r1/full_metrics.csv)／[二维图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/hy3_glm47_expert_doubling_20261022_r1/expert_doubling_surface.png)／[曲线](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/hy3_glm47_expert_doubling_20261022_r1/expert_doubling_curves.png) | 同一张 A100 顺序测两个模型的一层专家。实际配对行数从 8 开始逐次翻倍到 8192；活跃专家数为 1、2、4、8、16、32。剔除专家数多于配对行数的组合，每模型 63 个有效形状。 | 固定 128 行，活跃专家从 4 增到 32：Hy3 耗时从 0.1751 增到 0.9236 毫秒，GLM 从 0.0891 增到 0.2857 毫秒。 | Hy3 使用 BF16/Triton，GLM 使用 INT4/Humming；两者不是同精度架构消融。配对行数不是请求数，没有前缀缓存、Attention 或网络，不能把曲线直接称为 Prefill/Decode 请求延迟。 |
| [E06：本轮 GLM／Hy3 单层 Expert](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/single_layer_ep4_ep8_matched_20261024/compute_v2/REPORT.md) · A100 065 GPU0；[全部比较表](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/single_layer_ep4_ep8_matched_20261024/compute_v2/expert_comparison.csv)／[通信与Expert完整联表](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/single_layer_ep4_ep8_matched_20261024/component_comparison/REPORT.md)／[曲线](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/single_layer_ep4_ep8_matched_20261024/compute_v2/expert_curves.png) | 整批唯一 token 从 8 翻倍到 4096；分别对比 EP8 整批／EP4 前后半批，以及相同单次输入。只用真实一层权重、均衡 top-8 路由，不扫专家数。每卡实际配对数与 T01 通信扫描对齐。 | 整批128 token、两边每卡均128配对行：GLM 的 EP8／EP4半批均值为 0.2708／0.3471 毫秒；Hy3 为 0.7639／1.4144 毫秒。 | 预热 eager 主表；GLM AWQ/Humming、Hy3 BF16/Triton。单GPU模拟一个代表性目的rank，不含 Dispatch／Combine、Attention、KV或PP调度；不是完整PP吞吐／TPOT。 |

### 7.3 同时研究 A/E 平台，或把通信也加进来的实验

| ID／入口 | 维度与可以引用的结果 | 这组能回答什么，不能回答什么 |
|---|---|---|
| [简短报告 C01](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/C01.md) · A100 TP4 / EP4 平台；[完整 CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_ae_plateau_tp4_ep4_20261009/analysis/full_metrics.csv)／[Attention 热图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_ae_plateau_tp4_ep4_20261009/analysis/heatmap_12_attention_cuda_graph.png)／[Expert 热图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_ae_plateau_tp4_ep4_20261009/analysis/heatmap_14_expert_core_cuda_graph.png) | Attention 分支在单节点四卡内做张量并行：Decode 每请求处理 1 token，已缓存前缀为 1024/16384/49152 token，请求数从 1 到 128 选点；Prefill 另扫描每请求新增输入量，并检查 127/128/129 token 的分块边界。Expert 分支在另一个四卡节点独立测均衡路由。 4 请求、每请求 GPU 已缓存 16384 token 时，Prefill 新增输入从 128 增至 129 token，Attention 从 1.4684 增至 2.9133 毫秒。固定每卡 40 个活跃专家，全局唯一新增 token 从 20 增到 256 时，Expert 从 0.3564 增到 0.3727 毫秒。 | 专家分支在路由前去除来源复制：全局每个唯一 token 命中 8 个专家，均衡后每卡平均收到唯一 token 总数的两倍配对行。两分支的节点与 L2 协议不同，不可相加求整层占比。 |
| [简短报告 C02](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/C02.md) · H100 TP4 / EP4 平台；[CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_h100_ae_plateau_tp4_ep4_20261009/full_metrics.csv)／[报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_h100_ae_plateau_tp4_ep4_20261009/report.md)／[Expert 图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_h100_ae_plateau_tp4_ep4_20261009/expert_d4_core.png) | Attention 同样区分已在 GPU 缓存的前缀长度、请求数、本轮新增输入量；Decode 每请求处理 1 token，Prefill 单列。Expert 主扫描保留旧来源复制协议，同时提供去除复制后的独立控制。 4 请求、每请求已缓存 16384 token 时，Prefill 新增输入从 128 增至 129 token，Attention 从 1.0561 增至 2.0704 毫秒。 | 专家主扫描的每卡实际配对数是表内全局逻辑 token 数的 8 倍；去除来源复制的控制为 2 倍。表中“来源复制倍数”区分两者，不能直接与 A100 按同名 token 数比较。独立组件不能相加成完整请求延迟。 |
| [简短报告 C03](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/C03.md) · A100 单层 A/E/DeepEP 网格；[16点 CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_single_layer_ae_ep4_20261006_r1/main_cells_wide.csv)／[分项图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_single_layer_ae_ep4_20261006_r1/p_by_q_absolute_latencies.png)／[占比图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_single_layer_ae_ep4_20261006_r1/p_by_q_component_shares.png) | 每次固定 2 个请求。每请求 GPU 已缓存前缀为 0、1024、10240、196608 token；本轮每请求处理 1、16、256、512 个 token，共 16 组。处理 16/256/512 token 时使用 Prefill/Extend；处理 1 token 且已有前缀时使用 Decode；没有前缀且仅 1 token 的点是 Extend 边界控制。 每请求已缓存 10240 token、本轮新增 16 个输入 token 时，Attention、专家、通信的事件跨度中位数分别为 0.9923、0.2877、0.3604 毫秒。表中发给专家与取回结果分别列出。 | 前缀 KV 已位于 GPU；关闭 CUDA Graph，强制 DeepEP 低延迟路径。Attention 与专家输入独立，不是完整 Transformer 层；事件跨度可能含 CPU 提交空档，不等于纯 GPU 忙碌时间。 |

**怎么用这些曲线：** 想找Attention平台先看 **A02、A04、A06、C01**；想看Expert多命中权重的成本先看 **E03–E05**；想把EP8/PP的单卡计算形状对齐看 **E02、E04、E06**；想分开看Dispatch/Combine与Expert子内核看 **C03**。这些组件数据能筛选后续配置，但还需真实容量、PP边界和调度关键路径，才能预测完整请求速度。

扫描范围、原始表头/行数、文件哈希和图入口另存 [component_sweeps.json](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_sweeps.json)；链接、矩形网格及代表数值复核见 [component_validation.json](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_validation.json)。**没有把这些内核计时塞进请求级 metrics.csv，也没有为它们补造 TTFT/TPOT。**

<!-- BEGIN Q01 QWEN EP2 PP2 -->
## 8. 补充：单节点双卡 EP2 与 PP2（不是八卡 EP8）

以下是独立资源规模的相关实验，不混入第 2–4 节的八卡排名或旧 metrics.csv／profiles.csv。每个条目先读简短报告中的术语和计时边界。

| ID／简短报告 | 设备与明确的请求负载 | 已有结果 | profile／原始数据 |
|---|---|---|---|
| [Q01：Qwen3.5-122B，EP2／四槽 PP2／两槽 PP2](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q01.md) | 单节点同一对 A100 NVLink；8 请求。Prefill：每条无旧缓存、新输入4096、输出1。Decode：每条GPU已缓存4096、新输入1、输出128；无CPU换入。 | Prefill：EP2 **8670.75**、四槽 PP **8810.58**、两槽 PP **8185.11** input token/s。Decode：EP2 **72.08**、四槽 PP **49.04**、两槽 PP **97.26** output token/s。TTFT／TPOT／端到端均值、中位数、P99 见报告，不能只按吞吐认定全面领先。 | [全部请求指标](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_ep2_pp2_metrics.csv)；[Prefill逐卡计算／通信](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_ep2_pp2_prefill_profiles.csv)；[来源与哈希](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_ep2_pp2_archive.json)；[Attention／槽位详细诊断](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_prefill_diagnosis/REPORT.md)。含真实模型 profile；GPU毫秒／层不等于请求TPOT。 |
| [Q02：PP2 发送就绪优化](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q02.md) | 与 Q01 同节点同一对 A100；两槽 PP，8 请求，无旧缓存、每条新输入 4096、输出 1；另有真实 GPU 缓存 Decode 回归。 | 同进程 A/B/B/A Prefill：8160.97→9391.74 input token/s（+15.08%）。本轮正式 Prefill token 一致。 Decode 独立判定见报告。 | 同时保存每卡 Attention／投影、Routed／Shared Expert、NCCL 重叠的独立 GPU trace；原失败证据保留，未默认修改共享引擎。 |
| [Q03：H100 双卡 EP2／两槽 PP2](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q03.md) | 单节点 178 四张 H100，两组双卡并行。中等 8 请求；长缓存总计 214016 token，分 8／16 请求；另有 PP 自身 674816 token 容量点。Prefill 和 Decode 分开，旧前缀在 GPU，准备和 profile 不计时。 | 同工作量 Decode：PP 吞吐低约 **20%–23%**；8 请求长缓存 Prefill 高 **19.69%**，16 请求本轮均值低 **5.61%**。KV 逻辑容量 EP **237952**／PP **750848** token。不是 A100 到 H100 的硬件单变量比较。 | 原生 FP8 DeepGEMM、FA3、实际 Decode Graph；已核验缓存命中及独立逐卡 GPU trace；完整 TTFT／TPOT／端到端三种统计见报告。  [Decode 分项补分析](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_h100_20261025/runs/h100_ep2_pp2_r7/analysis/component_followup/REPORT.md)。 [Prefill 分项补采](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_h100_20261025/runs/h100_prefill_components_r1/analysis/REPORT.md)。 |
| [Q04：单 GPU 大小模型 H100 EP2／PP2](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q04.md) | Qwen3.6-35B-A3B BF16；178 节点两对 NV6 双卡并行；8 请求中等负载，16／32 请求固定总量长缓存。 | 同工作量 PP Prefill 吞吐 +14.15%／+11.72%／−4.48%；Decode 低 18%–25%。全部逐请求 TTFT／TPOT／端到端均值、中位数及 P99，见完整报告。 | 真实 BF16 DeepGEMM、FA3／Triton、Decode Graph；分开采全部 Prefill chunk 和实际 Decode Graph，可看 Attention、Expert、Dispatch／Combine 以及 PP 边界驻留。 |
| [Q05：H100 PP2 Prefill 局部优化](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q05.md) | 同Q04的35B BF16；178节点GPU2／3。8请求无旧缓存、每条新增4096；另有32请求、各GPU已缓存89,984、各新增512；输出1。 | 冷Prefill chunk4096→8192：吞吐+10.63%；长缓存发送就绪：+4.08%。短／长缓存预算扫描及有界合批负结果均保留；不外推Decode。 | 46正式cohort／23组全指标，独立两阶段GPU trace、TTFT／端到端mean／median／P99和原始哈希；不与EP8规模混排。 |
| [Q06：35B BF16 A100 补齐及三维对照](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q06.md) | 069 节点同一对 NV12 两卡，EP2／两槽 PP2 顺序运行；8 请求；冷 Prefill4096→1，GPU 已缓存4096的 Decode1→128；各3轮。 | Prefill EP **27644.14**／PP **31841.16** 输入 token/s（PP+15.18%）；Decode EP **90.80**／PP **121.76** 输出 token/s（PP+34.09%）。对齐既有 H100 与122B，区分硬件、模型、工作负载证据及混杂因素。 | 独立真实 GPU A/E/D/C 分项及 PP SendRecv 驻留；96请求计时、8完整rank导出、缓存计数、Guard／cache／数值资格及清理通过；无跨节点阈值结论。 |
| [Q07：H100 跨模型 Expert／通信取舍](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q07.md) | 离线对齐 Q03／Q04；每组两卡 NVLink。8 请求；冷 Prefill 每条无旧缓存、4096 新输入；Decode 每条 GPU 已缓存4096、新输入1、输出128。 | 35B／122B 的 PP Prefill 均值优势为 +14.15%／+6.39%；Decode 吞吐低25.08%／19.92%。冷 Prefill 多出的 Expert 累计66／118 ms/卡，EP D+C约133／129 ms/卡；支持机制线索，不支持模型越大一定越偏EP。 | 26行正式指标、独立 Prefill／Decode 分阶段值、每层归一化、图、来源SHA256和可重算脚本；无新GPU实验，精度与实现混杂及历史原始审查状态均保留。 |
| [Q08：35B A100／H100 稳态 Decode 与决策 PPT](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q08.md) | 每平台同一对 NVLink 双卡顺序 EP2／两槽 PP2；8请求，各 GPU 缓存4096、新输入1、输出128，各3轮。 | A100 PP 的 TPOT **86.258→64.271 ms（−25.5%）**；H100 EP **11.149→7.962 ms（−28.6%）**。Expert／通信取舍、Attention对照与例外纳入12页中文PPT。 | 36份rank trace，每份24个稳态B4 forward；原生Graph与eager诊断分开、PP边界单列；96请求计时、来源哈希、Guard／cache／清理及释放验证。 [PPT](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/deliverables/PP_EP_deployment_decision_20261026.pptx)／[PDF](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/deliverables/PP_EP_deployment_decision_20261026.pdf)。 |
| [Q09：H100 PP Decode 提前发送激活](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q09.md) | 同一卡对，35B BF16／122B FP8；8请求，各GPU缓存4096、新输入1、输出128；固定两个PP槽位各4请求。 | 固定分组后PP吞吐+3.98%／+1.35%；首stage的Expert与NCCL驻留重叠消失。PP仍慢于EP2；绝对TPOT差额2.683／3.040ms，不能推导普遍的模型大小规律。 | 最终两次Guard／缓存／清理通过；6行指标、12份逐卡Graph统计、3072条传输校验及逐请求输出一致。保留早期随机分组队列，未改共享引擎默认。 |
| [Q12：H100 PP Expert 执行碎片化优化](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q12.md) | 同一对178节点H100 GPU0/1；35B BF16；8请求；冷Prefill4096→1，GPU缓存4096后Decode1→128；原生3轮、优化5轮。 | PP Decode **666.79→805.67 tok/s（+20.83%）**，Expert GEMM **94.19→57.62µs（−38.83%）**；EP为863.20tok/s。Prefill无稳定收益，慢轮全部保留。 | 等输入/权重/路由的resident Expert维度控制支持布局成本；完整生成存在数值差异，尚未验证任务质量。7次Guard均清理和释放；完整图、逐请求CSV与来源见报告。 |

<!-- fair_ep_pp_20261026 --> Q12扩展：双方应用相同小批量GEMM优化后的[公平EP2/PP2对照](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/deliverables/REPORT.md)；EP +3.06%，PP +18.61%，独立保留历史cohort。


<!-- END Q01 QWEN EP2 PP2 -->

<!-- BEGIN H100 COMM DOMAIN Q10 -->
## 9. 跨规模通信域对照：H100与A100两卡及八卡

不同规模与模型的历史cohort不混入同workload排名；端到端现象和独立纯通信证据分开呈现。

| ID／简短报告 | 两组部署与workload | 已有观察 | 数据、图与限制 |
|---|---|---|---|
| [Q10：H100通信域与EP／PP选择](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q10.md) | 2卡NVLink：Qwen122B FP8，8请求；冷Prefill4096→1，GPU缓存4096后Decode1→128。8卡两节点：GLM INT4，4／16请求；冷Prefill8192→1，冷输入128→512。 | 两卡Decode EP比PP快24.87%；跨节点八卡EP比PP慢21.50%／17.12%。同库纯通信的跨节点EP8／节点内EP4：小消息约5.7–6.1倍，大消息约9.8–10.1倍。 | [12行指标、归一化图、绝对通信图及来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_comm_domain_comparison_20261026/REPORT.md)。模型、缓存、batch、native及chunk不同；没有严格同workload网络因果估计；纯通信不包含计算或PP调度。 **[PP合并的通信／Expert对比](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_component_shares_20261026/GROUP_COMPARISON.md)**：两卡Prefill／Decode、八卡Prefill单层；八卡Decode分项缺失。 |

<!-- END H100 COMM DOMAIN Q10 -->

<!-- BEGIN A100 COMM DOMAIN Q11 -->
### A100 对应证据

| ID／简短报告 | 部署与准确负载 | 已有观察 | 数据、图与限制 |
|---|---|---|---|
| [Q11：A100两卡／八卡通信域对照](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q11.md) | 两卡Qwen122B FP8／35B BF16，8请求：无旧缓存Prefill新输入4096→1输出；GPU缓存4096后新输入1→128输出。八卡GLM INT4，4请求：无旧缓存8192→1及冷输入128→512。 | 两卡Decode PP快34.92%／34.09%；八卡GLM Prefill PP快46.93%，Decode约1.85%。同库1024-token通信CE策略节点内1.3348／跨节点7.0528ms；RDMA策略8.4070／6.3670ms，方向相反。 | [完整图、12行指标、20行同调用通信表与来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/a100_ep_pp_comm_domain_comparison_20261026/REPORT.md)。没有H100式Decode反转；模型、内核、缓存、Graph不同，不能唯一归因于网络。T01没有计算或完整PP调度，旧四槽结果单列。 |

<!-- END A100 COMM DOMAIN Q11 -->

## 10. 以后怎么追踪

**看请求速度：** 第2/3节与 [metrics.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/metrics.csv)。**看慢在哪里：** 第4节与 [profiles.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/profiles.csv)。**核查来源：** [provenance.json](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/provenance.json)、[validation.json](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/validation.json)。

CSV 共122条性能行（含相关2×EP4、历史记录及重复 cohort，**不是122个独立实验**）、49条 profile 行，来自45份文件；旧网页引用的源文件哈希全部匹配。空白表示未测/未提供，**不表示0**。小样本 P99 仅描述本批请求，不是稳定的服务尾延迟。本索引覆盖已找到并核验的结果；没有找到完整 profile 的地方不会用微基准冒充。

<!-- begin qwen35b_h100_pp_graph_gap_optimization_20261027 -->
## PP2 Decode 的 Graph 外调度优化（2026-10-27）

[简短报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_graph_gap_optimization_20261027/REPORT.md) · [图表](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_graph_gap_optimization_20261027/pp_decode_scheduling.png) · [完整归档](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_graph_gap_optimization_20261027/manifest.json)

同一 H100178 NVLink GPU0/1、Qwen3.6-35B-A3B BF16、B8（固定 B4+B4）、真实 GPU-cached4096+new1→128。五轮 profiler-off：同等 GEMM 的 EP2 为 7.657 ms/token、890.4 tok/s；PP2 基线 9.303 ms/token、804.9 tok/s；PP 下一槽位提前 metadata/真实 GPU forward 为 6.998 ms/token、1048.3 tok/s。GEMM 未再改，完整20,480个PP正式token逐位置一致。Graph外间隔包括对端依赖/CPU调度/其他槽位工作，不能当作纯NVLink传输。静态PP2固定B4原型；profile、启动/验证和正式计时分开。失败候选、逐轮数据、源哈希与清理证明均保留。
<!-- end qwen35b_h100_pp_graph_gap_optimization_20261027 -->

<!-- begin qwen35b_h100_ep2_native_overlap_20261027 -->
## EP2 原生通信/计算 overlap（2026-10-27）

[报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_ep2_native_overlap_20261027/REPORT.md) · [图表](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_ep2_native_overlap_20261027/native_ep_overlap.png) · [归档](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_ep2_native_overlap_20261027/manifest.json)

同一 H100178 NVLink GPU0/1、Qwen3.6-35B-A3B BF16、B8真实GPU-cached4096+new1→128，同等小批次GEMM。五轮 profiler-off：原生shared-expert overlap关闭 8.413ms/token、824.4tok/s；开启 7.664ms/token、889.8tok/s。当前Qwen路径默认开启，因此此前约7.66ms的EP2已包含此优化。两rank实际Graph stream分支及独立GPU并行区间验证，所有结局与清理证据保留。TBO flag outcome=failed; no valid performance point.
<!-- end qwen35b_h100_ep2_native_overlap_20261027 -->

<!-- begin qwen35b_h100_expert_workload_validation_20261027 -->
## H100 Expert 优化的 workload 扩展验证（2026-10-27）

[报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_expert_workload_validation_20261027/report/REPORT.md) · [图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_expert_workload_validation_20261027/report/workload_generalization.png) · [归档](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_expert_workload_validation_20261027/manifest.json)

同一完整Qwen3.6-35B-A3B BF16、H100178 NVLink GPU0/1；EP2、PP2均比较native与相同small-tile内核。全局B8/B16/B32，对应每卡T4/T8/T16；另有B32多主题输入。每配置5轮真实GPU缓存4096+new1→128 Decode，含独立冷Prefill、路由与Graph分项。b8_repeated PP +18.20%; b16_repeated PP +15.37%; b32_repeated PP +11.92%; b32_diverse PP +11.73%。本轮仅扩大小tile启用范围，不启用PP调度优化或TBO；完整检查及清理通过。
<!-- end qwen35b_h100_expert_workload_validation_20261027 -->

<!-- begin qwen35b_h100_pp_lookahead_workloads_20261027 -->
## H100 PP 提前发射的 workload 扩展验证（2026-10-27）

[报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_lookahead_workloads_20261027/report/REPORT.md) · [吞吐/TPOT图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_lookahead_workloads_20261027/report/workload_comparison.png) · [调度分解](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_lookahead_workloads_20261027/report/scheduling_decomposition.png)

同一完整Qwen3.6-35B-A3B BF16、H100178 NVLink GPU0/1；B8/B16/B32及B32多主题，真实GPU缓存4096+new1→128。EP/PP均保留相同Expert内核；原生PP与提前发射PP每组5轮交替对照，实际分组、输出、wire hash及提前发射计数验证通过。b8_repeated: lookahead vs native PP +28.79%, vs EP +18.17%; b16_repeated: lookahead vs native PP +26.65%, vs EP +7.79%; b32_repeated: lookahead vs native PP +21.00%, vs EP +0.32%; b32_diverse: lookahead vs native PP +18.69%, vs EP +0.89%。独立GPU分项及冷Prefill控制已保存，Guard与清理通过。
<!-- end qwen35b_h100_pp_lookahead_workloads_20261027 -->

<!-- begin qwen35b_h100_prefill_optimized_20261027 -->
## H100 冷 Prefill 的 EP2 / PP2 优化对照（2026-10-27）

[报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_prefill_optimized_20261027/report/REPORT.md) · [吞吐图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_prefill_optimized_20261027/report/prefill_throughput.png) · [Expert/通信分解](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_prefill_optimized_20261027/report/prefill_components.png)

完整Qwen3.6-35B-A3B BF16、同一H100178 NVLink GPU0/1。缓存0，输入128/512/4096→输出1，B8/B32及B32长文本多主题；双方相同Expert选择器，PP开启适用于Prefill的提前提交activation。主矩阵每点5轮，独立GPU GEMM/dispatch/combine归因；18项CPU及Guard/清理检查通过。b8_q128_repeated: PP early vs EP +53.84%, vs PP native +0.18%; b8_q512_repeated: PP early vs EP +61.31%, vs PP native -0.35%; b8_q4096_repeated: PP early vs EP +12.06%, vs PP native -10.09%; b32_q128_repeated: PP early vs EP +50.60%, vs PP native -1.38%; b32_q512_repeated: PP early vs EP +22.87%, vs PP native +0.02%; b32_q4096_repeated: PP early vs EP +17.57%, vs PP native +0.07%; b32_q4096_diverse: PP early vs EP +18.96%, vs PP native +0.30%。

B8/Q4096独立10轮复测：提前发送相对原生PP仅 +0.145%，无稳定Prefill收益。该复测wrapper在后续非计时logprob探针OOM而保持failed，仅使用完整审计的已完成计时。短输入原生EP/PP有数值差异，未认定跨布局质量等价；Q4096主实验首token一致。单独降低静态显存预算的PP数值诊断不进入性能比较。详见报告的边界说明与followups.json。

两rank现有trace复查：B32/Q4096每卡每层dispatch324.82、combine259.22、GEMM484.73 µs；通信与routed GEMM/Attention的时间交集均0，与shared expert交集3.98 µs。未重启TBO或新GPU实验；详见[重叠分析](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_prefill_optimized_20261027/report/overlap.md)。
<!-- end qwen35b_h100_prefill_optimized_20261027 -->

<!-- begin qwen35b_h100_ep_prefill_overlap_20261027 -->
### qwen35b_h100_ep_prefill_overlap_20261027

Snapshot: /data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_ep_prefill_overlap_20261027

Full Qwen3.6-35B-A3B BF16, H100178 NVLink GPU0/1; AttentionDP2/TP1 MoEEP2 versus two-stage/two-slot PP2. Cold Prefill context0 + input4096 → output1; B8/B32 repeated and B32 diverse in the main matrix, both layouts using an 8192-token per-Attention Prefill budget. Three cold warmups and five counterbalanced profiler-off cohorts/policy. EP p0 native, p1 serial two-chunk, p2 overlapped two-chunk, p3 overlapped with communication SM reservation. Attention/GDN unchanged, shared-expert overlap ON, full-model TBO OFF; native path retained for mixed/idle tails. Per-forward CPU policy synchronization included in all EP arms. PP native/early controls unchanged. Actual same-GPU communication/GEMM intersections are measured in separate profiles. Per-layer real-weight numerical tests, formal output agreement, exact input/cache/output and handle-drain receipts, 21 CPU tests, frozen source hashes, canonical weight-cache usage, Guard postcheck and cleanup/release audited. An independent 4096-token-budget round and a ten-cohort diverse-input recheck remain separate. Failed qualification attempts retain their status and contribute no performance samples. At the common 8192 budget, B32 repeated EP overlap/reserved-SM throughput improves +1.83% over native EP; PP2 remains faster.

B32 input4096: ep2 p0: 88,448.4 input tok/s, 990.63ms mean E2E; ep2 p1: 82,856.3 input tok/s, 1058.85ms mean E2E; ep2 p2: 86,105.9 input tok/s, 1014.70ms mean E2E; ep2 p3: 90,066.4 input tok/s, 970.08ms mean E2E; pp2 p0: 103,701.9 input tok/s, 715.10ms mean E2E; pp2 p1: 102,825.5 input tok/s, 732.87ms mean E2E.

[Report](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_ep_prefill_overlap_20261027/report/REPORT.md) · [Figure](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_ep_prefill_overlap_20261027/report/prefill_overlap.png)
<!-- end qwen35b_h100_ep_prefill_overlap_20261027 -->

<!-- begin qwen35b_h100_ep_decode_overlap_20261027 -->
### qwen35b_h100_ep_decode_overlap_20261027

Same H100178 NVLink GPU0/1, full Qwen3.6-35B-A3B BF16. B8/B16/B32 repeated and B32 diverse, real GPUcached4096 + new1 -> output128. Static per-process Decode CUDA Graph policies, identical generic Expert kernel family. Native EP2, two-chunk overlapping EP2, B32 serial control, and PP2 with two fixed slots and policy9 lookahead. Five profiler-off cohorts after three warmups; independent two-rank profiles with actual communication/GEMM intersections. Attention/GDN whole, shared-expert overlap ON; no Prefill DeepGEMM SM budget carried into Decode. Every EP variant/workload validates both ranks all40 real-weight MoE layers. EP long-sequence cross-cohort variation is recorded, including native controls; PP fixed-group sequences must agree. 23 CPU tests, exact cache/output/stream counts, actual graph replay receipts, frozen hashes, canonical CPU weight-cache postcheck, owned cleanup and lease release audited. Failed EP sequence-stability attempt and PP schema-warmup attempt excluded. b8_repeated: overlap vs native -19.65% throughput; b16_repeated: overlap vs native -20.12% throughput; b32_repeated: overlap vs native -19.84% throughput; b32_diverse: overlap vs native -20.07% throughput.

[Report](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_ep_decode_overlap_20261027/report/REPORT.md) · [Figure](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_ep_decode_overlap_20261027/report/decode_overlap.png)

Snapshot: /data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_ep_decode_overlap_20261027
<!-- end qwen35b_h100_ep_decode_overlap_20261027 -->

<!-- begin qwen35b_a100_nvlink_regression_20261027 -->
## A100 NVLink 双卡回归补测（2026-10-27）

[报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_a100_nvlink_regression_20261027/report/REPORT.md) · [吞吐图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_a100_nvlink_regression_20261027/report/throughput.png) · [指标](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_a100_nvlink_regression_20261027/analysis/metrics.csv) · [Expert／Dispatch／Combine／Attention](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_a100_nvlink_regression_20261027/analysis/components.csv)

Qwen35B BF16，A100069物理GPU0/1，租约内确认NV12。EP2与PP2同原生Triton Expert；3组Decode、3组Prefill，每配置5轮，另列PP0/PP9调度比较。15个计时点与30个rank profile，完整输出/通信载荷/分组校验，Guard清理和释放通过。decode_b8_repeated p0: PP/EP 1.348x; decode_b8_repeated p9: PP/EP 1.338x; decode_b32_repeated p0: PP/EP 1.374x; decode_b32_repeated p9: PP/EP 1.379x; decode_b32_diverse p0: PP/EP 1.385x; decode_b32_diverse p9: PP/EP 1.395x; prefill_b8_q128_repeated p0: PP/EP 1.624x; prefill_b8_q4096_repeated p0: PP/EP 1.150x; prefill_b32_q4096_diverse p0: PP/EP 1.118x。
<!-- end qwen35b_a100_nvlink_regression_20261027 -->

<!-- begin qwen122b_h100_latest_ep_pp_decode_20261027 -->
### qwen122b_h100_latest_ep_pp_decode_20261027

2×H100178 GPU0/1 NVLink, complete Qwen3.5-122B-A10B native FP8/DeepGEMM. B8/B16/B32 repeated and B32 diverse; identical prior prompts, real GPUcached4096 + new1 → output128. EP2 AttentionDP2/TP1 and 48 layers/rank versus PP2 24+24 layers and two fixed B/2 slots. Both BF16 KV/residuals and FP32 GDN; BF16-only Triton kernel substitution is inapplicable to this model. Common native redundant-prefix-lock skip and 1.875GiB KV/rank. EP native/shared-expert overlap and optional two-chunk MoE overlap versus PP policy9 early send/lookahead. Five profiler-off cohorts/point, independent rank profiles, 768 real-weight layer comparisons, 23 CPU ownership/admission tests plus 12 native state-cache tests. Exact input/cache/output/Graph checks, canonical CPU weight-loader receipts and owned cleanup audited. Exploratory failures and prepared-only configurations remain excluded and preserved. b8_repeated: PP vs overlapping EP +25.09%, EP overlap vs native -20.13% throughput; b16_repeated: PP vs overlapping EP +12.91%, EP overlap vs native -20.38% throughput; b32_repeated: PP vs overlapping EP +17.32%, EP overlap vs native -22.40% throughput; b32_diverse: PP vs overlapping EP +18.31%, EP overlap vs native -22.11% throughput.

[Report](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen122b_h100_latest_ep_pp_decode_20261027/report/REPORT.md) · [Figure](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen122b_h100_latest_ep_pp_decode_20261027/report/decode_comparison.png)

Snapshot: /data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen122b_h100_latest_ep_pp_decode_20261027
<!-- end qwen122b_h100_latest_ep_pp_decode_20261027 -->

<!-- begin qwen122b_h100_pp_refinement_20261027 -->
### qwen122b_h100_pp_refinement_20261027

Qwen3.5-122B-A10B native FP8, node178 two H100s over NVLink, B32 repeated GPUcached4096+new1→output128. PP9 versus PP10 tail-local retirement: five counterbalanced profiler-off pairs, unchanged Graph compute, complete native output agreement. Measured-time 25+23 PP partition trial and fresh native optimized EP2 control; matching-position independent profiles. PP2 24+24 + policy9: 1593.46 tok/s, TPOT 18.888 ms; PP2 24+24 + tail-local feedback: 1600.53 tok/s, TPOT 18.793 ms; PP2 25+23 + policy9: 1594.40 tok/s, TPOT 18.865 ms; PP2 25+23 + tail-local feedback: 1600.88 tok/s, TPOT 18.771 ms; EP2 native optimized: 1752.15 tok/s, TPOT 16.145 ms.

[Report](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen122b_h100_pp_refinement_20261027/report/REPORT.md) · [Figure](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen122b_h100_pp_refinement_20261027/report/decode_refinement.png)

Snapshot: /data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen122b_h100_pp_refinement_20261027
<!-- end qwen122b_h100_pp_refinement_20261027 -->

<!-- begin qwen35b_h100_ep4_vs_four_ep1_capacity_20261028 -->

## 35B 四卡容量补充：EP4 与四个 EP1（2026-10-28）

[简短报告](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep4_vs_four_ep1_capacity_20261028/REPORT.md) · [逐轮数据](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep4_vs_four_ep1_capacity_20261028/full_metrics.csv)。同一 H100 四卡、Attention TP1、BF16，B64；KV token 池 EP4 13.57×。相同 P8256：EP4 热 Prefill 0.752×、Decode 吞吐 1.051×。EP4 稳定测试点 P130304；四个 EP1 同负载另记排队/重算，不能混入热缓存比较。此 B64 长 context 容量实验作为额外对照，未并入132格标准 Benchmark 计数。

<!-- end qwen35b_h100_ep4_vs_four_ep1_capacity_20261028 -->

<!-- begin qwen35b_h100_five_layouts_hicache_20261028 -->
### qwen35b_h100_five_layouts_hicache_20261028

Five four-GPU H100 layouts, Qwen3.6-35B-A3B BF16, native GPU/CPU HiCache enabled throughout. B64, P8256/P130304; new Prefill Q512->1 and Decode Q1->128; natural replay and controlled CPU replay, three formal cohorts, phase-specific PP pilot selection. Exact native tier hits, request timings, capacities, frozen inputs, canonical CPU weight-cache usage, owned cleanup and released Guard leases audited. All five formal layouts are on node178; node171 exploratory runs are excluded. Additional B64 capacity cohorts do not change the standard132-cell benchmark count.

[Simple report](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_five_layouts_hicache_20261028/SUMMARY.md) · [Full report](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_five_layouts_hicache_20261028/REPORT.md) · [Round data](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_five_layouts_hicache_20261028/full_metrics.csv)

Snapshot: /data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_five_layouts_hicache_20261028
<!-- end qwen35b_h100_five_layouts_hicache_20261028 -->

<!-- begin h100_four_model_expert_surface_20261029 -->
### H100 四模型：逐 GPU 的绝对原始耗时

GLM-4.7 AWQ、HY3 BF16、Qwen3.6-35B BF16、Qwen3.5-122B FP8 的真实单层 Expert 二维图。T=8..32768 个本地 token–expert 配对行，E 独立控制。两个 H100 节点，每模型两份独立单 GPU 测量。

默认展示各 GPU 的第 1/30 次原始 CUDA event，单位 ms；不归一化，不计算均值或中位数，不合并 GPU。Graph 与 eager 分开，所有 47520 次原始计时均已保存。GLM 使用 use_tma_bzp=False 兼容配置。原实验的数值和清理检查已通过，当前更新只重画已有记录。

[绝对耗时报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_four_model_expert_surface_20261029/REPORT.md) · [每格原始计时](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_four_model_expert_surface_20261029/absolute_single_event.csv) · [全部 30 次原始值](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_four_model_expert_surface_20261029/absolute_all_events.csv)。Source run: /data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_expert_surface_4models_20261029/v4/runs/r6。
<!-- end h100_four_model_expert_surface_20261029 -->

<!-- begin h100_ep_token_comm_surface_20261029 -->
### H100：EP2/EP4/EP8 × 总 token 通信二维表（2026-10-29）

四种激活尺寸 GLM/HY3/35B/122B，统一 BF16 / top8，固定全 EP 组原始 token 总数8–32768。
EP2/EP4 单节点 NVLink，EP8 两节点 NVLink+RDMA；统一全组2048-token 分块，大档为完整2/4/8/16次交换实测。
624点、87360条 rank原始事件；每格固定事件1，分轮显示，Dispatch/Combine/合计绝对ms，无平均或relative；Guard校验与清理通过。
[报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_ep_token_comm_surface_20261029/REPORT.md) · [交互表格](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_ep_token_comm_surface_20261029/report.html) · [二维图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_ep_token_comm_surface_20261029/total_ms_graph_round1.png) · [原始CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_ep_token_comm_surface_20261029/absolute_event01.csv)
<!-- end h100_ep_token_comm_surface_20261029 -->

<!-- begin h100_ep_pp_component_tradeoff_20261029 -->
### H100：Expert 计算与通信的布局权衡组件模型（2026-10-29）

按现有单层 Expert 与 DeepEP LL 通信原始数据做同 GPU 预算比较；没有新跑 GPU。
GLM/HY3 最小 EP4，仅 EP8 对 PP2×EP4；35B/122B 另含纯 PP2/4/8。
130 个条件，每种模式 120 个有完整组件；Graph/Eager、两份计算与两轮通信分开，30次事件无平均。
差额为省通信减多计算的绝对 ms；以全部观测样本 min/max 的保守包络标 PP/EP/边界。
理想饱和流水估算，未测 PP 交接、气泡、Attention 与 overlap；大 N 同时改变分块，缺点不外推。
[报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_ep_pp_component_tradeoff_20261029/REPORT.md) · [区域图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_ep_pp_component_tradeoff_20261029/tradeoff_cuda_graph.png) · [组件绝对值 CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_ep_pp_component_tradeoff_20261029/component_estimates_event01.csv) · [来源](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_ep_pp_component_tradeoff_20261029/sources.json)
<!-- end h100_ep_pp_component_tradeoff_20261029 -->
