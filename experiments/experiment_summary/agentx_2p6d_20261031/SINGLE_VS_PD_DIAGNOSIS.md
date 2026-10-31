# C8 单卡混部为什么比当前 PD 快

当前最有支持的解释：低 Decode batch 下，单卡副本的执行路径更快；固定 P 池又增加了首 token 前等待。P 排队只解释部分总差距。当前 PD 的 D 使用 EP，尚不能把这组差距归因为 P/D 分离本身。

这里的单卡是 8 个独立单卡实例，总预算仍为 8 张 H100。所有结果均为当前 AgentX 派生 C8 的完整 episode，HiCache 开启。旧单卡、EP4混部、4P4D均为 E5/V3 三轮，新2P6D为 E8/V4各一轮。

## 先定位差距

| 部署 | 首 token 前 ms | 首 token 后至结束 ms | 请求 E2E ms |
|---|---:|---:|---:|
| 混部：8×单卡 | 348.46 | 2210.25 | 2558.71 |
| 混部：2×EP4 | 431.71 | 3111.26 | 3542.97 |
| 4P4D：EP4 P→EP4 D（历史完整 C8） | 572.04 | 3018.77 | 3590.81 |
| 2P6D：PP2 P→EP4＋EP2 D | 616.39 | 2920.23 | 3536.62 |
| 2P6D：PP2 P→3×EP2 D | 591.09 | 2994.27 | 3585.36 |

对新的 EP4＋EP2 D：比单卡多 977.92ms，首 token 前增加267.93ms，首 token 后增加709.99ms，后者约占73%。旧同版本4P4D的额外1032.10ms中，首 token后约占78%。首 token后包含通信、GPU执行、调度和流式返回，不能直接称为纯Expert计算。

同一E5/V3的配置路径为：单卡混部2558.71ms → EP4混部3542.97ms（增加984.27ms）→ EP4的4P4D 3590.81ms（再增加47.84ms）。这是配置均值差分；不能将其中任何百分比解释成纯通信占比。

## 已有证据支持什么

- 实际执行粒度小：旧单卡本地Decode batch步均值约1.04，旧EP4混部约1.06，新PD约1.12；这些计数以发生forward的步为权重，不能当GPU利用率。EP带来逐层专家交换和rank协调，其权重分片优势未必转化为低batch延迟优势。- 启动代码开启DP Attention，EP4的tp=dp=ep=4。Attention在每个DP rank独立处理本地请求；EP4不会把单个请求的长context Attention均摊到四卡。每卡只持有四分之一专家，也不等于每步只执行单卡的四分之一工作。
- 一个均衡路由的简化例子：四卡各有1个Decode token、top-8。四个单卡副本各处理8行token-expert工作；EP4全组32行平均分到四卡，每卡仍是8行，却需要交换。此例不表示8个唯一专家，实际专家复用仍取决于路由。较大batch下权重复用与分片容量才可能抵消这些额外成本。
- 独立纯Decode实验也有相同方向：四卡节点、热缓存P4096/N8时，单卡副本4.752ms、EP4 6.122ms；P131072/N8时为6.550/7.970ms。它们是不同协议的旁证，不与AgentX数值合并。
- 缓存命中几乎一致：P侧约91.46%，本组host命中均为0；新增Prefill约7245与7248token/请求。没有证据把C8差距主要归为缓存未命中。权重已在GPU中，CPU权重加载不在本轮请求计时内。

## P 排队与交接

| 部署 | 原生P排队 ms | P首次forward至Prefill结束 ms |
|---|---:|---:|
| 混部：8×单卡 | 3.60 | 326.59 |
| 4P4D：EP4 P→EP4 D（历史完整 C8） | 72.25 | 413.27 |
| 2P6D：PP2 P→EP4＋EP2 D | 286.80 | 236.41 |
| 2P6D：PP2 P→3×EP2 D | 270.52 | 238.77 |

P阶段的跨度包括chunk之间的等待，不是纯GPU kernel时间。PP2 P的这一跨度反而较短，但它集中承接全部请求，排队显著增多。原生queue_time是进入wait queue到首次forward，不包含所有bootstrap等待。TTFT还包含握手、接收预分配、缓存状态交接、D开始服务与API开销；不能用TTFT减P计算就得到纯传输耗时。
客户端E2E与联合／D服务端原生E2E的差约为11–15ms（EP4混部约22ms）。这没有显示HTTP/客户端线程开销能解释约1秒的总差距；服务端CPU调度仍包含在原生计时中。

8个单卡混部实例可随本地会话阶段执行P或D；固定2P6D把P工作汇聚到一个PP2实例，D也仅有六卡。这种资源分工可能在有限并发与工具等待时留下闲置容量，目前缺少时间加权利用率，不能定量归因。

若保持当前PD首token后的执行轨迹，即使假设TTFT为0，平均E2E仍约2920ms，高于单卡完整请求的2559ms。该代数假设说明单独压缩首token等待不够，不是对改变合批或增加D资源后的性能预测。

## 输出一致性与结论边界

另外筛选同版本单卡与4P4D的逐token输出完全相同请求：185/513对，其间隔加权TPOT仍为5.312/7.114ms。新EP4＋EP2 D与单卡第一轮的40/171个同输出对子集为5.468/6.902ms。这个有选择的子集不是全量性能估计，但说明生成序列变化不能单独解释全部慢速。其余请求并未达到完整输出一致，不能称为全并发数值等价验证。

尚缺的最关键对照是在同引擎、同回放下让PD的D端也使用单卡副本，并保持相同输入和可驻留的缓存工作集。单卡必须使用容量可行的GDN/KV预算；不能直接照搬EP的256个GDN槽。该对照有助于区分D布局成本与P/D分离成本；目前还没有这组结果。

[延迟分解CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_2p6d_20261031/single_pd_decomposition.csv) · [输出分组及同输出子集](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_2p6d_20261031/single_pd_diagnosis.json) · [复算脚本](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_2p6d_20261031/analyze_single_pd.py) · [C8总表与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_2p6d_20261031/C8_COLOCATED_COMPARISON.md)

[独立纯Decode对照](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/REPORT.md) · [本轮DP Attention启动代码](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_ep4ep2_owner_balanced_e8_v4_gdn256_20261031_control/code/runner.py) · [原生计时定义](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/agentx-mha-pp-kv-layer-map-r8-20261031/python/sglang/srt/observability/req_time_stats.py)
