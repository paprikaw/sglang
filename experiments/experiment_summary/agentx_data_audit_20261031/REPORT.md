# AgentX：已有数据盘点与 benchmark 审计

2026-10-31，墨尔本。[实验总索引](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/README.md) · [EP／PP 速览](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/EP_PP_EXPERIMENTS_COMPACT.md)。本次只核对、整理已有文件，没有启动 GPU 实验或修改已测协议。

**现有数据可以保留，但应称为「AgentX 派生负载的系统实验」。18 轮混部是同协议下的有效观测；3 轮独立 P/D 是修复后的跑通与压力证据。它们还不足以证明真实 AgentX 中 P/D 分离、EP／PP 动态配比有收益。除了父子分支重叠，最需要闭环的是分支延迟基准和计时输出差异。**

## 1. 实际测过什么

「点」指一个部署×负载组合；「轮」指一次完整有限回放。一次完整回放不等于整组计划通过，也不等于官方协议或数值正确性全部通过。

| 数据组 | 已完成内容 | 如何使用 |
|---|---|---|
| 早期缓存片段 | 两组各 4 GPU：2×EP2、2×PP2；C1／C8，4 点×3 轮，共 480 调用 | 单独保留为兼容性初筛。只有连续 5 轮片段、无子代理 DAG；两个布局在不同节点，不能直接推出布局胜负。随后 PD 启动失败，整组 Guard 未通过；合并部分另有独立核查 |
| E5＋V3 混部基线 | 两节点共 8 GPU；2×EP4、4×PP2、8×单卡；C8／C128，6 点×3 轮＝18 轮，26,163 调用 | 通过整组完成、后检查、清理和本次请求级复算。用于当前派生回放的固定部署对照，保留容量及协议限制 |
| E7＋V3 独立 PD | 4P4D，EP4 P→EP4 D，C128；1 点×3 轮＝3 轮，8,208 调用 | 通过整组验收；证明真实 KV／GDN 交接及修复后压力流程可跑通。与 E5 混部不是同源码控制，不能据此归因 PD 得失 |
| 早期／失败组中的完整 episode | 7 个历史点、21 轮不重复测量，其中包括已整组完成的 EP4 混部／PD C2／C8 对照 | 源码或预热协议不同，见历史表，不并入主表；失败前完成的 C8 也不代表 C128 或整组通过 |
| E7＋V4 准备 | 原计划 24 轮，GPU 完整结果 0 | 原自动队列已取消，准备文件保留；不能当作已测数据 |

主实验目录共 74 个，含 46 份 `formal` 文件，按内容 SHA256 去重为 42 份。其中 4 份是 EP4 前一目录导入的同一测量，已核对来源哈希，**不能重复算独立轮次**。主表只取上述 21 轮；其余 21 份不重复的历史文件在清单中保留，不混算。目录名、预建空日志或 `queued` 状态本身不证明已经运行。

入口：[早期片段及独立核查](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_pd_feasibility_20261030/REPORT.md) · [逐运行清单](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_data_audit_20261031/run_inventory.csv) · [逐 formal 文件及去重依据](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_data_audit_20261031/formal_inventory.csv)。

### 主结果：绝对时间、三轮算术平均

Qwen3.6-35B-A3B，BF16 权重／KV、FP32 GDN；两台各 4 张 H100 80GB。以下延迟越低越好；E5、E7 分开解释。TPOT 为每请求平均再对请求取均值，不是每个 token 的最坏间隔。

| 源码／回放 | 根会话 C | 8 GPU 部署 | TTFT ms | TPOT ms/token | 请求 E2E ms | 压缩整轮 s |
|---|---:|---|---:|---:|---:|---:|
| E5／V3 | 8 | 2×EP4 合并 | 431.71 | 9.640 | 3542.97 | 443.42 |
| E5／V3 | 8 | 4×PP2 合并 | 277.15 | 7.137 | 3266.10 | 441.65 |
| E5／V3 | 8 | 8×单卡合并 | 348.46 | 4.990 | 2558.71 | 397.42 |
| E5／V3 | 128 | 2×EP4 合并 | 2271.32 | 73.435 | 22624.02 | 1168.21 |
| E5／V3 | 128 | 4×PP2 合并 | 1188.73 | 39.419 | 13128.90 | 835.80 |
| E5／V3 | 128 | 8×单卡合并 | 27305.92 | 18.662 | 33312.39 | 1449.09 |
| E7／V3 | 128 | EP4 P→EP4 D，4P4D | 60765.86 | 11.170 | 65526.39 | 2409.73 |

本次从原始请求重新核对：21 轮共 **34,371 调用**，所选家族、每家族副本数、调用身份、输入／输出长度、完整标记及 TTFT／TPOT／E2E／根会话时间／整轮均值均对得上。这个通过范围是完成性和统计复算，**不包含“所有动态 batch 下数值输出完全一致”**。早期 480 调用沿用已有独立核查，不混入本次 21 轮复算。

[逐轮复算](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_data_audit_20261031/accepted_episodes.csv) · [复算均值](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_data_audit_20261031/accepted_means.csv) · [原 E5 比较和全部 SLO](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/matrix_analysis/REPORT.md) · [E7 等待分解](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/eight_gpu_pd_ep4_e7_t300_b7200_r1/analysis/PD_WAITS.md)。

### 历史表：另外 7 点，保留但分组解释

以下也各有三轮完整输出，本次复算了请求数及 TTFT／TPOT／E2E 均值。它们没有进入上面的规范主矩阵；其中 `ep_r6` 确实整组完成，不能笼统说「之前只有一个 PD 点」。表中延迟单位与主表相同，均为绝对值。

| 历史运行／源码 | 配置 | C | TTFT ms | 请求 TPOT ms/token | E2E ms | 组别说明 |
|---|---|---:|---:|---:|---:|---|
| ep_r6／f5f5997 | 2×EP4 合并 | 2 | 435.48 | 6.163 | 3151.18 | 与下方同源码 PD 配对；缺后来的完整预热 |
| ep_r6／f5f5997 | EP4 P→EP4 D | 2 | 438.22 | 6.203 | 3173.45 | 同一旧组，共 12/12 轮完成 |
| ep_r6／f5f5997 | 2×EP4 合并 | 8 | 408.68 | 9.737 | 3530.11 | 有首用效应，保留全部三轮 |
| ep_r6／f5f5997 | EP4 P→EP4 D | 8 | 576.88 | 6.946 | 3598.59 | 平均 TPOT 改善没有变成整请求 E2E 改善 |
| col_single_warm_r1／f823795 | 8×单卡合并 | 8 | 347.93 | 4.980 | 2553.70 | 旧组只完成 3/6；后续有主基线替代 |
| pd_ep4_durable_slots8_r4／E5 | EP4 P→EP4 D | 8 | 572.04 | 6.923 | 3590.81 | C8 完整，整组高负载阶段失败 |
| pd_ep4_checkpoint_e6_r2／E6 | EP4 P→EP4 D | 8 | 566.96 | 6.938 | 3585.29 | C8 完整，整组高负载阶段失败 |

这里还有一个容易误读的点：`ep_r6` C8 的请求平均 TPOT 为 **9.737→6.946 ms**，但按输出 token 间隔加权后是 **7.180→6.951 ms**，首 token 后的请求均值为 **3121.43→3021.71 ms**。短输出与长输出等权会放大平均 TPOT 的变化；评估 PD 收益应同时看请求 E2E、加权 TPOT 和同一 SLO。该组的首用效应也尚未排除。

[历史均值与完整运行名](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_data_audit_20261031/historical_means.csv) · [逐轮及导入去重标记](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_data_audit_20261031/historical_episodes.csv) · [早期首用效应核查](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/eight_gpu_ep_r6/analysis/EXPOSURE.md)。

## 2. 这个 benchmark 实际如何进行

1. **取数据并筛选完整家族。** 使用已下载、固定 revision／SHA256 的 AgentX Weka 256k 哈希 trace。全量有 393 个根会话、68,266 调用。按完整家族 4–80 调用、总输出≤16,000、单请求输入＋输出≤256,000 筛选，有 34 个合格家族；导出 24 个候选，再固定其中 8 个进入八卡比较。保留官方 loader 的 orphan 处理，不截断长请求。
2. **重建输入与请求图。** 官方 harness 固定为 `e4bb2f835ffedbbe6b76252ea5c93caec68c5faf`，用于 loader／规则核查；实际 GPU 测量运行自写 replayer，直接请求 `/generate`。原 trace 只有 64-token 块哈希：按家族及副本隔离命名空间，用标记 token＋重复代码正文构造相同长度及块级前缀关系；父子家族内部可共享前缀。它不是原始用户文本。
3. **组成有限负载。** 8 个家族共 171 调用、45 个嵌套调用。C8 使用一份；C128 使用 16 份缓存隔离副本，共 2736 调用。根会话在并发许可下从本轮开头准入，后续调用受依赖、完成事件与工具等待驱动；没有稳态开放到达流量扫描。主结果使用 V3 分支触发。
4. **固定模型及部署，保留缓存。** 每个 rank 开 HiCache、CPU 32 decimal GB／rank，GPU KV 按静态比例 0.95 原生分配。每轮前做非计时串行参考及覆盖各实例／Attention-DP rank 的冷／热前缀预热，再清空原生缓存开始正式轮次；**正式会话内不清缓存**。同一子会话粘在首次选择的实例／DP owner，主矩阵没有动态路由或实例重配置。EP4 为 Attention TP1／DP4、MoE EP4；PP2 为层间 PP2、EP1。
5. **实际执行 P、D。** 混部在同一实例完成 P 和 D；PD 同时发出配对 P／D 请求，实际交接 KV＋GDN 状态，不是把两类完整请求分别隔离。输出采用温度 0、忽略 EOS，生成来源指定的输出长度。下一轮输入按 trace 独立构造，不拼接实时生成文本。
6. **计时、压缩和验收。** HTTP 执行开始→首个输出为 TTFT；首→末输出除以输出间隔数为请求 TPOT；HTTP 执行开始→末输出为请求 E2E。另记请求就绪→HTTP 执行开始的客户端等待。只有全局无在途请求时，超过 10 秒的空闲被压到 10 秒。每点三轮，完整保存请求及完成标记，再做后检查和资源清理；部分输出、失败和未完成轮次不并入主表。

**不拼接实时输出不是这里独有的 bug：** 固定官方 Weka 默认 `DELTAS_WITH_RESPONSES` 也使用 trace 预置的 assistant 内容，不把实时输出反馈进下一轮。区别在于本次还使用自造 token、完整有限回放、自定义准入／时序及空闲压缩，未运行官方 chat 渲染、轨迹中段抽样与计分流程。因此本结果不称为官方 AgentX 分数，也不测 agent 任务完成质量。[输入与官方模式核查](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/PROTOCOL_AUDIT.md)

版本名也应分开：**V3／V4 是回放器版本，E5／E7 是引擎补丁版本**。运行环境的包元数据为 SGLang 0.5.18；实际执行以冻结的源码为准。E5=`37e0828c642ce4f3795c4c4394f4ce913bcd1a35`；E7=`797f6488cf5fea2f88e38ccaeb1703443deec537`。E7 增加状态压力处理和 HiCache 有界恢复重试，不能只按相同包版本号视为相同控制。

## 3. 除父子重叠之外，还有什么问题

| 核对项 | 已确认事实 | 对结论的影响与处理 |
|---|---|---|
| **分支延迟基准有误** | 不止重叠 SPAWN 被延后；V3 的部分分支以父请求源结束为基准，官方用 branch-start 相对偏移。例如一个分支 V3 加约 144,993 秒，官方对应返回路径偏移为 0 | 改变 batch、空闲和缓存历史。V3 原值保留为独立协议；V4 只完成规则 CPU 检查，还没有 GPU 结果，不能声称已修复重测 |
| **计时输出差异未定位** | 同 E5、C128 第 1 轮，PP2 对 EP4 的整段输出相同 865/2736，首 token 相同 2462/2736；双方均无缓存的 434 调用中，56 次首 token 不同。本次也另列 E7 配对 | 不同 batch／浮点路径可能导致贪心分歧，但尚未用受控 logits 定位；既不能全归因缓存错误，也不能宣布正确性全部闭环。输出差异还可能改变实际 Decode 专家路由 |
| **负载覆盖窄且重复放大** | 只用 8/393 家族、171/68,266 调用；平均输出 436 对全量 860 token，嵌套调用占比 26% 对全量 58%。C128 是相同 8 家族的 16 份副本 | 能制造容量和并发压力，不能替代 128 个多样真实家族。不能由此推广到长输出、强 fan-out、原始语义下的 MoE 路由或所有 AgentX；调用数也不是独立样本数 |
| **空闲压缩及有限准入** | 三轮均值跳过约 14.3–14.5 万秒，且各布局跳过量不同；根会话在有限一轮内准入，没有稳定到达率扫描 | 数百秒整轮时间是自定义压缩回放墙钟；请求/s 不是生产到达吞吐或饱和容量。它仍可作为该有限实验目标，但不单凭它论证动态扩缩容 |
| **容量、缓存及源码不同** | 单卡 GDN 等效预算 40，EP／PP 为 80；GPU KV 总容量单卡 271 万、PP2 1508 万、EP4 1781 万 token。PD 使用 E7，混部 E5；PD 接收预留8、并受状态池压力影响 | 固定预算部署比较可保留，但不能把差值全部解释为通信／Expert 计算。HiCache 开启不代表缓存容量或命中相同；E7 需同源混部控制。PD 的高 TTFT 也不能直接等同传输开销 |
| **根均衡不等于实际工作／缓存均衡** | C128 的根家族初始分配已做副本轮转；子会话仍按首次就绪顺序取 owner，后续保持亲和。就绪顺序随运行完成时间变化 | 相同路由代码不保证相同 rank 工作量和缓存历史；P 侧已观察到新增 token 不均衡。它可以是系统真实行为，但尚不能单独归因 EP／PP 计算能力 |
| **延迟、SLO 与并发口径需明确** | 上表 TTFT／E2E 未加客户端就绪后等待；原分析另提供 ready 指标和已加回等待的 SLO。TPOT 是请求平均，未提供逐 token p99 ITL；C128 是根数，不是 Decode batch | 标明计时起点，不能把 HTTP 均值和 ready SLO 混读；不能声称满足每个 token 的间隔上限。C128 本地纯 Decode batch 均值仅约 EP4 5.62、PP2 5.04、单卡 2.21 |
| **尚无动态策略收益证据** | 正式主矩阵仅固定粘性分配；最新修复仅一个完整高负载 PD 点，早期另有 C2／C8；尚缺同源码最强混部、各 P:D 比例及单卡 D 对照 | 不能说“PD 无效”，也不能说“动态配比有效”。探索性 `cache_queue` 的成本常数、剩余输出及缓存估计尚未校准，不计作正式验证 |

对应证据：[分支／空闲](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/TIMING_AUDIT.md) · [输出逐轮配对](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_data_audit_20261031/output_agreement.csv) · [负载覆盖](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/WORKLOAD_COVERAGE.md) · [缓存与 owner 诊断](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/CACHE_ROUTING_DIAGNOSIS.md) · [动态策略边界](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/POLICY_AUDIT.md)。

客户端就绪后等待的三轮均值如下，均为绝对 ms。它远小于单卡和 PD 的高负载 TTFT，不能解释主要差距；但严格 SLO 应使用 ready 起点。原 E5 分析已同时保存两种 SLO，本次再次核对原始计数并并列导出：[HTTP／ready 两种 SLO](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_data_audit_20261031/latency_scope_slo.csv)。

| 根会话 C | EP4 合并 | PP2 合并 | 单卡合并 | E7 EP4 P→D |
|---|---:|---:|---:|---:|
| 8 | 6.99 | 7.13 | 7.73 | 未测完整对照 |
| 128 | 27.83 | 33.25 | 135.28 | 222.72 |

## 4. 目前可以保留的结论与最小补齐顺序

- **可保留：** 当前 V3 有限派生回放，按请求平均 E2E，C8 单卡最好，C128 PP2 最好；EP4 在部分严格 SLO 格仍有优势。缓存容量、排队及恢复共同参与，不能套用单层组件图直接解释。
- **尚不能下结论：** 真实 AgentX 的最优部署、P/D 分离的净收益、同一目标下 EP／PP 动态配比收益，以及全部高并发缓存路径的数值正确性。
- **先闭环协议和输出。** 固定一个回放版本与引擎；对出现首 token 分歧的来源请求做小规模冷输入／GPU 命中／CPU 恢复对照，以首步 logits 和状态恢复证据区分数值路径与功能错误。无需先重跑整张大表。
- **再补关键受控比较。** 同 E7、同修正时序、同 8 GPU 预算，先固定混部单卡／PP2／EP4；再与实际 PD 最好配比比较。保持会话内 HiCache，明确 GDN、接收槽和 GPU KV 容量，记录实际 batch、ready 延迟、缓存与排队。
- **最后再问动态是否必要。** 补入较长输出和更强 fan-out 的完整家族，在同一 SLO 下先看到固定配置互补，再测动态分配能否超过最强固定基线；不能把不同目标下的最好格子拼成动态收益。

本次复算脚本、来源哈希、导入去重关系和统计 CSV 都保存在本目录：[审计清单](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_data_audit_20261031/audit.json) · [复算脚本](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_data_audit_20261031/recompute_audit.py)。大日志与逐请求输出留在原运行目录；本专题不复制权重、KV 内容或认证文件。
