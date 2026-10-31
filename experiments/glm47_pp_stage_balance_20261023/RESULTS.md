# PP2×EP4 MTP Decode 优化

SemanticB8, logicalP4096 / nativeCPU4032, Q1 ->64, MTP topk1/steps2/VerifyQ3; CPU HiCache H2D timed

| 配置 | 输出 tok/s ↑ | TTFT mean ms ↓ | TPOT mean ms ↓ | E2E mean ms ↓ |
|---|---:|---:|---:|---:|
| 原 PP 46/46 | 128.603 | 502.200 | 52.818 | 3829.735 |
| PP 46/46，本地反馈退休 | 132.586 | 503.078 | 50.976 | 3714.559 |
| PP 50/42，本地反馈退休 | 140.726 | 515.796 | 47.398 | 3501.888 |

EP8 同 Bank 历史参照：131.585 tok/s，TPOT 46.739 ms；不是本次重新运行的控制组，跨布局严格输出等价仍未通过。

候选与前一轮固定 Verify / 本地退休基线逐 token 和文本完全一致；各臂两个生产 cohort 自重复通过。MTP-off 的已有数值差异没有在本轮修复。

两次优化的实际分段、无逐次同步的调度诊断、每 cohort 指标和原始来源哈希保存在 summary.json。组件 Graph 时长不等于 TPOT，不能将 stage 时长相加解释为完整流水线周期。

第一步只删除已由尾 stage 持有的结果从首 stage 回传的冗余环路。第二步把4个目标层移到首 stage，补偿尾 stage 的草稿工作；总层数、Expert 数、请求、MTP 参数及输出预算均未减少。

## 控制条件与测量边界

- 同一对已预约节点 065／167、8×A100 80GB、同型号 ConnectX-6 Dx；没有新增预约。
- GLM-4.7-AWQ / Humming INT4，BF16 KV；Attention TP4，Expert EP4，PP2。固定两个 B4 slot，不减少全局 B8。
- 真实语义前缀 P4096，原生 CPU HiCache 实际持有4032 token，GPU 起始为空；计时内重算64 token 尾部并新增 Q1，生成64 token／请求。
- CPU 权重预载及源 Bank 导入不计时，原生 HiCache→GPU 换入计时。两种分层方式复用同一个不可变语义 Bank，按逻辑层映射，不重建前缀。
- MTP top-k1、steps2、Verify Q3，完整 CUDA Graph；每个配置两个无 profiler 的生产 cohort，之后独立做分段与调度诊断。样本数小，不给出显著性或泛化结论。
- DeepEP 原生库、Expert 内核、通信容量不变；节点内 NVLink pair，pair 外均走 RDMA，PCIe threshold=0。本轮没有启用或修改 CE 路径。

## 归因

第一步移除冗余反馈：尾 stage 已持有结果，但原来仍需等待结果经首 stage 回传。现在保留该 slot 的独占结果，等生产事件就绪后在尾 stage 本地消费；尾→首反馈和 stage 激活传输均保留。slot 生命周期、请求归属和退出排空均有检查。

无逐 replay 同步的诊断中，尾 stage 活跃反馈 API 平均耗时由3.193降至1.048 ms。该值包含主机调度、序列化与等待，不能当作纯 NIC 延迟。目标 Graph 计算时长基本不变，生产 TPOT 降低3.49%。

第二步按实际工作而非目标层数平衡：模型是92个目标层加1个物理 MTP 层，steps2并不代表额外的三层权重。原来46／46分层没有扣除尾 stage 的草稿工作。移动4个目标层到首 stage 后，目标计算总量没有减少。

| 诊断项：每卡／每次相应 Graph，ms | 本地反馈，46／46 | 本地反馈，50／42 |
|---|---:|---:|
| 首 stage 目标 Graph | 45.934 | 50.083 |
| 尾 stage 目标 Graph | 48.597 | 44.454 |
| 尾 stage 草稿 + draft extend | 4.705 | 4.654 |
| 尾 stage 目标 + 草稿工作 | 53.302 | 49.108 |
| 两段目标 Expert 合计 | 26.068 | 26.051 |

这张表只用于比较分段工作量，不把其和解释为生产 TPOT。无逐次同步的实际同 slot 启动周期，rank0／rank4由121.39／121.57降至112.28／112.49 ms，和独立生产测试的改善方向一致。间隙包含其他计算及依赖，不等于 GPU 空闲时间。

## 验证与限制

- 两步候选均与相应固定 Verify PP 基线的8个请求逐 token ID、文本完全相同，自重复也完全一致；平均推进2.4653 token／验证周期、接受率0.73034和每请求验证次数26均未改变。
- CPU 协议与 Bank 分层映射测试共18项通过，真实 GPU 性能、profile、Guard 后验检查、退出清理均通过；两次 GPU 租约均已验证释放。
- 同 Bank 旧 EP8 参照的 Expert 目标计算约25.06 ms，而 PP 两段约26.05 ms；DeepEP dispatch+combine约26.71对17.93 ms。这里的 Expert 小幅增加不足以抵消通信差额，但这些 Graph 组件不能直接相加推导生产速度。
- 相对旧 EP8，优化 PP 的吞吐高6.95%、TPOT仍高1.41%；EP8不是本次新跑的控制组，不能据此宣称严格胜出。
- 固定 Verify 的 MTP 与 MTP-off 原有数值不一致仍未修复，EP／PP之间也未通过严格输出等价。本轮证明的是保持 PP 基线输出不变的调度优化，不是无损 MTP 的完整正确性证明。
- 50／42只在本次 B8、P4096、MTP 参数下验证，不是所有 context／batch 的通用最优分层。代码目前是冻结 runtime 的显式启用适配，不是默认全局开启。
