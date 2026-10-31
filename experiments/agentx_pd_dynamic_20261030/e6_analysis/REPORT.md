# AgentX：E6 修复版与同版本控制

两节点共8张H100，HiCache开启、GPU KV原生自动容量；V3派生回放，完整SLO网格。本表只纳入E6，交接等待上限120秒；其他源码或超时协议保留在原表。均值为三轮算术平均；少于三轮或整组未验收的行仅作进度。已保存但缺完成标记的原始回放列在audit中，不计入均值。

| 配置 | 负载 | 轮数 | 整组验收 | TTFT ms | 请求TPOT ms/token | 加权TPOT ms/token | 请求E2E ms | 含客户端排队E2E ms | 根会话 s | 压缩整轮 s |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| pd_ep4 (GDN80) / receive8 | agentx_c8 | 3/3 | False | 566.96 | 6.938 | 6.943 | 3585.29 | 3592.53 | 282.341 | 451.413 |

| 运行 | 阶段 | 完成/计划 | 整组验收 | 有压力延期日志的进程日志数 |
|---|---|---:|---|---:|
| eight_gpu_pd_ep4_checkpoint_e6_r2 | failed | 3/6 | False | 1 |
| eight_gpu_pp2_pd_controls_checkpoint_e6_r1 | queued | 0/9 | False | 0 |
| eight_gpu_ep2_split26_checkpoint_e6_r1 | queued | 0/6 | False | 0 |
| eight_gpu_ep4_native_queue32_checkpoint_e6_r1 | queued | 0/6 | False | 0 |

| 运行 / 配置 | 角色 | GPU | 独立缓存位置 | 完整层KV容量 token | HiCache配置预算 GB |
|---|---|---:|---:|---:|---:|
| eight_gpu_pd_ep4_checkpoint_e6_r2 / pd_ep4 | prefill | 4 | 4 | 8,901,888 | 128 |
| eight_gpu_pd_ep4_checkpoint_e6_r2 / pd_ep4 | decode | 4 | 4 | 8,780,032 | 128 |

KV容量按每个实例报告的可用token上限乘Attention DP数；PP各阶段只计一次。P和D分别列出，不能相加当成Decode可用容量。HiCache一栏是CPU配置预算（十进制GB），不是实际驻留字节。所有已有正式样本的配置均核对了8个rank的原生HiCache挂载日志。

GDN表示每GPU的完整模型等效状态预算；PP2每阶段的实际槽位数翻倍。receive表示每Attention-DP worker的Decode接收预留。160/receive32是独立容量配置控制，不与80/receive8合并平均。

延期日志每个组件只记录首次发生，不是累计频率或延迟。P/D客户端缓存字段可能合并P侧计数，不等于D侧独立复用量或传输字节。压缩回放的root、wall及达标请求/s不能当作原始生产时间或官方AgentX分数。

[每轮值](trials.csv) · [全部均值](means.csv) · [完整SLO网格](slo_means.csv) · [逐实例缓存容量](cache_capacities.csv)

[验收、压力日志和来源](audit.json) · [E5原始对照](../matrix_analysis/REPORT.md)
