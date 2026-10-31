# AgentX：P/D 分离与 EP／PP 动态分配验证

2026-10-31 · 阶段报告，P/D控制与动态收益仍待验证。[数据盘点与 benchmark 审计](../agentx_data_audit_20261031/REPORT.md) · [Compact 总览](../EP_PP_EXPERIMENTS_COMPACT.md) · [统一索引](../README.md)

**在已验收的V3派生回放下，合并部署低负载单卡最好，高负载 PP2 最好；尚未证明 EP 与 PP 动态配比有必要。** EP4 在部分严格延迟目标下仍有优势。E7的EP4 4P4D高负载已完成三轮并验收；同源码合并控制尚缺，失败样本不计作性能差。后续关键比较统一使用修正后的V4分支时序，独立保存结果。

早期还完成过同一旧源码的EP4混部／PD C2、C8共12轮；另有旧单卡和E5／E6 PD C8完整轮次。它们已在[数据盘点历史表](../agentx_data_audit_20261031/REPORT.md)单列，含首用效应与不同源码边界，不与下方18轮规范基线混算。

## 已完成、已验收的同预算基线

Qwen3.6-35B-A3B，BF16，两个既有节点共8张H100；同一E5源码、来源请求、预热和副本分配协议。下列全部是**三轮算术平均**，18轮完整，Guard后检查与GPU清理均通过。

| 8 GPU配置 | 根会话并发 | TTFT ms | 请求平均TPOT ms/token | 请求E2E ms | 根会话完成 s | 压缩整轮 s |
|---|---:|---:|---:|---:|---:|---:|
| 2×EP4 合并 | 8 | 431.71 | 9.640 | 3542.97 | 277.341 | 443.421 |
| 4×PP2 合并 | 8 | 277.15 | 7.137 | 3266.10 | 275.731 | 441.653 |
| 8×单卡合并 | 8 | 348.46 | 4.990 | 2558.71 | 244.118 | 397.420 |
| 2×EP4 合并 | 128 | 2271.32 | 73.435 | 22624.02 | 790.365 | 1168.206 |
| 4×PP2 合并 | 128 | 1188.73 | 39.419 | 13128.90 | 554.648 | 835.796 |
| 8×单卡合并 | 128 | 27305.92 | 18.662 | 33312.39 | 1012.758 | 1449.092 |

C8／C128是根会话并发上限，不是每一步GPU Decode batch。单卡GDN预算40，EP4／PP2为每GPU完整模型等效80；PP2每阶段实际160槽位、微批上限16、原生policy0。这些是显式容量配置，不是只改变并行方式的理想化对照。

已有18轮计数表明，C128下每次本地纯Decode forward的batch均值为EP4 **5.622**、PP2 **5.036**、单卡 **2.212**；C8均约1。PP只计stage0，它不是整个实例全部在途请求；这些值也不是全EP组token或Expert收到的行数。因此不能把根会话并发128直接代入单层token图。[实际batch与口径](../../agentx_pd_dynamic_20261030/matrix_analysis/FORWARD_BATCHES.md)

实际ready记录确认：EP4与单卡保留原生overlap；PP2使用原生流水调度。EP4使用DeepEP auto，全部配置保留Decode CUDA Graph；不能将配置值`auto`当作各通信后端实际使用比例。

**为什么不能只看Decode：** C128单卡的TPOT低于PP2，但平均TTFT为27.31秒，完整请求反而最慢。三轮平均有1257次host命中、8次回退。首token前的延迟占了主要部分；缓存、计算和服务端排队的独立因果贡献尚未完全分离。

实际GPU KV总容量：单卡 **2,712,832** token、PP2 **15,083,904** token、EP4 **17,806,080** token。PP2阶段重复已去除。容量较大不自动意味着命中更多；C128每请求报告的缓存token均值分别约44,699、75,197、71,318。不能把全部性能差距归因于Expert计算或通信。

**EP4仍有目标相关优势：** C128在TTFT≤1秒、请求平均TPOT≤10ms/token时，达标请求/s为EP4 **0.3445**、单卡 **0.2175**、PP2 **0.1369**。TPOT放宽至15ms时，PP2 **0.5946**、EP4 **0.4214**、单卡 **0.2278**。这组SLO使用ready起点，已把另记的客户端就绪到发出等待加回TTFT；上表TTFT／E2E均值从HTTP执行开始计时，未加这段等待。分母是压缩回放时间；完整预设网格都保留。这尚不等于同一个目标下动态池有收益。

按8个原始家族拆分，C8和C128的请求E2E均为PP2低于EP4，且各家族三轮方向一致。它是混合负载中的事后分解，不能把每个请求的最好结果拼成动态路由收益。

[完整E5表、加权TPOT和缓存容量](../../agentx_pd_dynamic_20261030/matrix_analysis/REPORT.md) · [直接差值及每轮波动](../../agentx_pd_dynamic_20261030/matrix_analysis/COMPARISONS.md) · [家族与全部SLO](../../agentx_pd_dynamic_20261030/matrix_analysis/FAMILY_TRADEOFFS.md)

## 修复与正在验证的P/D

- **PP HiCache恢复已修复：** 全局层号与局部层号映射错误导致后半段跳过H2D复制。E5通过27项CPU检查和48/48实际恢复输出检查。[证据](../../agentx_pd_dynamic_20261030/PP_RESTORE_DIAGNOSIS.md)
- **P/D启动容量已核对：** 默认32个接收预留曾把GDN80扩为144，而KV预算仍按80计算，导致OOM。8槽配置已能启动；另安排显式GDN160、保留默认32槽的同源控制，避免低估P/D。[预算与控制](../../agentx_pd_dynamic_20261030/GDN_QUEUE_CONTROL.md)
- **P/D运行期压力：** E6通过C8三轮后，在高负载HiCache恢复环节失败。E7区分前缀丢失与临时分配不足，并增加有限重试；22项CPU检查通过，真实GPU上10次临时不足均已成功重新排队。该轮随后因继承的120秒交接等待上限中断，未得到完整高负载样本。此等待包含Prefill排队与计算，不是单纯网络时间；现在恢复引擎默认300秒重新验证，完整计入等待成本。[检查点证据](../../agentx_pd_dynamic_20261030/DECODE_CHECKPOINT_PRESSURE.md) · [恢复与超时证据](../../agentx_pd_dynamic_20261030/DECODE_HICACHE_RESTORE_PRESSURE.md)

E7缓存修复的V3 C128稳定性验证现已完成三轮。尚未启动的三组V3控制已取消，没有产生GPU结果。此前准备的同一E7源码、同一V4时序四组控制如下；其自动队列也已按用户最新顺序要求撤销，准备文件保留，尚未启动：

| 对照 | 固定容量与角色 | C128重复 |
|---|---|---:|
| 合并EP4 vs EP4 P→EP4 D | 8卡；分离为4P4D；GDN160／默认receive32 | 3+3 |
| 合并PP2 vs PP2 P→PP2 D vs PP2 P→EP4 D | 8卡；分离为4P4D；GDN80／receive8，PP微批16 | 3+3+3 |
| 合并EP2 vs EP2 6P2D | 8卡；GDN80／receive8 | 3+3 |
| 8个单卡合并副本 | 8卡；GDN40，保留较大KV容量 | 3 |

原24轮准备均保留，**尚无V4 GPU结果，原队列不再自动启动**。全部采用相同回放、分支helper、请求图、预热、整轮上限和失败捕获；16项CPU检查通过。GDN160与receive32是联合容量配置；6P2D有一对P/D同节点，不能忽略这些差异。后续先补全合并部署，再比较P:D配比和配比内布局，并加入单卡D对照；新矩阵仍需独立冻结与校验。[原准备校验](../../agentx_pd_dynamic_20261030/e7_v4_cohort_preflight.json) · [原队列撤销记录](../../agentx_pd_dynamic_20261030/wait_for_e7_v4_controls_r1.json) · [V4独立结果入口](../../agentx_pd_dynamic_20261030/e7_v4_b7200_analysis/REPORT.md)

恢复300秒交接等待后，首个尝试遇到共享资源租约中断，同一时刻另三个后台租约也过期；后续重跑又触发回放器40分钟整轮上限，尾部请求尚未结束。两次均无完整episode且已清理。当前新组统一使用2小时整轮上限，保持300秒交接等待和600秒HTTP等待；失败时单独保存已收集请求，不计均值。8项CPU回归检查通过。[整轮上限审计](../../agentx_pd_dynamic_20261030/EPISODE_TIMEOUT_AUDIT.md)[中断证据](../../agentx_pd_dynamic_20261030/runs/eight_gpu_pd_ep4_hicache_retry_e7_t300_r1/interruption.json)

**11:11进展：V3压力验证三轮完整完成，共8208次调用，整组验收、Guard后检查与GPU清理均通过。** 三轮请求平均TTFT **60765.86 ms**、TPOT **11.170 ms/token**、E2E **65526.39 ms**，压缩整轮均值 **2409.734 s**；P可运行队列等待均值 **42.661 s**。这是E7源码的独立4P4D点，与E5基线和V4正式对照分开，尚无同源码对照支持P/D收益归因。下表保留前两轮，全部三轮见链接。

|轮次|请求TTFT s|请求TPOT ms/token|请求E2E s|P可运行队列等待 s|压缩整轮 s|
|---|---:|---:|---:|---:|---:|
|r1|57.813|11.816|62.793|40.188|2485.712|
|r2|65.827|10.506|70.297|46.493|2451.284|

两轮均无请求回退，且都超过原2400秒整轮上限。P侧长等待重复出现；第二轮Decode更快，却有更长的首token等待和请求E2E。因此不能仅凭Decode加速认定P/D提高了整项任务性能。P和D等待互有重叠，不相加，也不当作纯网络时间。

四个P侧rank在两轮中均各接收32个根会话、50,038,272个根输入token，但新算量仍有差异。首轮根新算量10,044,544至41,188,992 token（约4.10倍），第二轮25,240,832至39,082,560 token（约1.55倍）：**P排队问题重复出现，但首轮卡间差异的幅度没有稳定复现。** 后续在线策略须给同质池相同的缓存／队列策略，才能区分路由本身与EP／PP混合配置的额外收益。

[三轮完整值与均值](../../agentx_pd_dynamic_20261030/e7_t300_b7200_analysis/REPORT.md) · [P/D等待分解](../../agentx_pd_dynamic_20261030/runs/eight_gpu_pd_ep4_e7_t300_b7200_r1/analysis/PD_WAITS.md) · [逐rank绝对值及既有基线核查](../../agentx_pd_dynamic_20261030/CACHE_ROUTING_DIAGNOSIS.md)

**所有角色HiCache开启，每rank 32 decimal GB CPU预算，GPU KV按0.95静态显存比例原生自动分配。** 同一episode内不清缓存，保留会话／DP亲和；仅在计时外预热及独立重复之间flush。

[E6独立记录](../../agentx_pd_dynamic_20261030/e6_analysis/REPORT.md) · [E7旧超时记录](../../agentx_pd_dynamic_20261030/e7_analysis/REPORT.md) · [当前队列计划](../../agentx_pd_dynamic_20261030/E7_V4_RUN_PLAN.json) · [实时状态](../../agentx_pd_dynamic_20261030/STATUS.json)

## 最终要回答什么

1. 更好的固定P/D布局和P:D配比，能否胜过强合并基线的完整任务时间或同一延迟目标下的达标吞吐。
2. 固定异构P/D是否有额外收益；EP与EP=1／PP是否在同一评价目标下稳定互有胜负。
3. 若互补仍存在，再测在线分配，并计入状态传输、缓存损失、排队和batch变化；同质池也使用相同缓存／队列策略。固定池分流与原地权重重分片分开报告。

单卡作为D的布局尚未实测，结论仅限所列配置。目前动态路由、动态实例配比及操作成本尚未实测；已有`cache_queue`只是未校准的探索实现。[策略核查](../../agentx_pd_dynamic_20261030/POLICY_AUDIT.md)

## 结果适用范围

已下载并校验官方相关Weka完整数据和256k版本。当前预选8个完整家族、每轮171次调用；C128使用16份缓存隔离副本，共2736次调用，未增加原始家族多样性。输入由token块hash重建前缀关系，输入／输出长度保留，不能还原真实语义或保证各布局实际专家路由一致。

**这是AgentX派生实验，不是官方分数。** 原始数据包含约40小时的间隔，V3仅在全局空闲时压缩长等待，且分支时序与官方运行时有差别。因此表中root、wall和goodput均是压缩回放指标，不代表真实生产会话时间。V4修正了分支发出／返回的触发与相对偏移，仍保留自定义根会话准入、hash-token输入和全局空闲压缩；不实现官方轨迹抽样、chat渲染或评分。V4结果将独立报告，不拼进上表。

受控串行／host恢复检查的通过，也不代表动态batch回放中的每个输出token都相同。计时输出存在差异；无缓存请求也有首token差异，尚未通过logits控制归因数值或缓存问题。

[下载来源](../agentx_pd_feasibility_20261030/REPORT.md) · [协议](../../agentx_pd_dynamic_20261030/PROTOCOL_AUDIT.md) · [覆盖范围](../../agentx_pd_dynamic_20261030/WORKLOAD_COVERAGE.md) · [时间核查](../../agentx_pd_dynamic_20261030/TIMING_AUDIT.md) · [输出一致程度](../../agentx_pd_dynamic_20261030/output_agreement_e5/OUTPUT_AGREEMENT.md)

[既有固定P/D阶段数据](../ep4_fixed_phase_comparison_20261030/REPORT.md) · [早期先导实验，独立协议](HISTORICAL_PILOT.md) · [此前详细进度记录](PROGRESS_HISTORY_BEFORE_E5_BASELINES_COMPLETE.md) · [全部原始运行](../../agentx_pd_dynamic_20261030/runs)
