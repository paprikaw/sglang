# AgentX：E7 修复版与同版本控制

两节点共8张H100，HiCache开启、GPU KV原生自动容量；V3派生回放，完整SLO网格。本表只纳入E7，交接等待上限120秒；其他源码或超时协议保留在原表。均值为三轮算术平均；少于三轮或整组未验收的行仅作进度。已保存但缺完成标记的原始回放列在audit中，不计入均值。

| 配置 | 负载 | 轮数 | 整组验收 | TTFT ms | 请求TPOT ms/token | 加权TPOT ms/token | 请求E2E ms | 含客户端排队E2E ms | 根会话 s | 压缩整轮 s |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|

| 运行 | 阶段 | 完成/计划 | 整组验收 | 有压力延期日志的进程日志数 |
|---|---|---:|---|---:|
| eight_gpu_pd_ep4_hicache_retry_e7_r1 | failed | 0/3 | False | 1 |
| eight_gpu_ep4_native_queue32_e7_r1 | queued | 0/6 | False | 0 |
| eight_gpu_pp2_pd_controls_e7_r1 | queued | 0/9 | False | 0 |
| eight_gpu_ep2_split26_e7_r1 | queued | 0/6 | False | 0 |

GDN表示每GPU的完整模型等效状态预算；PP2每阶段的实际槽位数翻倍。receive表示每Attention-DP worker的Decode接收预留。160/receive32是独立容量配置控制，不与80/receive8合并平均。

延期日志每个组件只记录首次发生，不是累计频率或延迟。P/D客户端缓存字段可能合并P侧计数，不等于D侧独立复用量或传输字节。压缩回放的root、wall及达标请求/s不能当作原始生产时间或官方AgentX分数。

尚无完整E7 GPU样本，CSV在第一份正式样本完成后生成。

[验收、压力日志和来源](audit.json) · [E5原始对照](../matrix_analysis/REPORT.md)
