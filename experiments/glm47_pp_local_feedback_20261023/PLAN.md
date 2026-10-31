# PP2×EP4 / MTP：反馈本地退休

目标是减少 PP 的闭环开销；不改变模型、Attention/Expert 内核、路由、两组 B4 slot、采样参数或输出预算。

## 对照

- `pp2ep4_baseline`：固定 Verify 修复 + 原有完整反馈环。
- `pp2ep4_local_tail`：最后一个 stage 保留自己已拥有的输出副本，在对应 slot 退休时本地处理。仍向首 stage 发送相同反馈，但首 stage 不再把整份反馈传回。
- 两臂顺序使用 065/167，8×A100、同型号高速 NIC，不申请节点。
- GLM-4.7-AWQ，Attention TP4 / Expert EP4 / stage；MTP topk1、2 draft steps、Verify Q3；完整模型有 92 个目标层和 1 个物理 MTP 层。
- B8，逻辑 P4096，原生 CPU HiCache 已有 4032 token；重算尾部64 + Q1，生成64 token。权重及语义 Bank 导入在计时前，HiCache→GPU 在计时内。
- DeepEP：NVLink pair 内直连，pair 外 RDMA；PCIe CE 未改变。

## 验证

1. CPU 队列测试执行固定引擎中的真实交换函数，覆盖双 slot、单 slot 提前结束、单 slot 为空、排空与请求祖先检查。不能代替 NCCL 或模型验证。
2. 每臂两个不带 profiler 的生产 cohort，输出逐 token 自重复。
3. 候选与本轮基线输出逐 token、文本完全相同，才通过优化正确性门槛。
4. 独立 `flow` cohort：每次真实 Graph replay 使用独立 CUDA event，回放间不做同步、不读时间戳；最后空闲时统一读取。记录 PP 主机 API 时长和同 slot 周期。当前 stream 的空档不解释为整张 GPU 的空闲时间，主机 API 也不是纯链路耗时。
5. 独立有 fence 的 A/E/通信分段诊断沿用原协议，不能把该诊断的节奏当成生产气泡。
6. 保留退出、GPU 清理、权重 bank 使用与 Guard postcheck 证据。

## 数值边界

此前固定 Verify 与 MTP-off 并未全部逐 token 等价。这里检查的是“调度优化保持固定 Verify 基线不变”，不宣称修复了已有的 MTP 数值等价问题，也不据此宣称 EP/PP 是严格等价输出的最终性能排名。

所有运行输入通过 Guard 冻结。保留原始失败，不修改已经冻结的运行代码；需要修正时创建新版本。
