# EP8 与 PP2×EP4：通信优势为何未表现为更低的平均 TPOT

完成状态：有效 PP 时间线来自原配对 run；旧 EP 诊断失败记录原样保留。修正路由后的 EP-only 补测已完成、Guard postcheck 与清理通过；两组有效结果和完整归因见 [REPORT.md](REPORT.md)。不能把原配对 run 的整体 failed 改称 passed。

本目录是单轮配对诊断，不改变原推理调度或模型数学。已有组件 profile 只用于说明层内通信优势，不能与本轮未插桩 TPOT 相减并把剩余称为 GPU bubble。

## 固定实验条件

- 065／167，同型号 ConnectX-6 Dx，8×A100 80GB；不申请新节点。
- GLM-4.7-AWQ／Humming INT4、BF16 KV；完整 CPU 权重 Bank、原始语义目标＋草稿 KV Bank 复用。
- B8、逻辑 P4096，计时开始 native CPU prefix4032／GPU0；第一次 forward 每请求仍需计算64个前缀尾 token＋Q1，输出64 tokens。
- EP8 为 Attention TP4×DP2／Expert EP8；PP2×EP4 为两个真实 B4 slot、各 stage Attention TP4／Expert EP4，目标层50／42，MTP 只在尾部。
- 两组都使用已经修复的 Verify Attention、EAGLE top1／steps2／Verify Q3、local Graph B4。
- 同一原生 DeepEP 9dc34887…；NVLink pair 内 NVLink，pair 外及跨节点 RDMA，无 CE。

## 验证步骤

1. CPU 测试覆盖初始 Prefill／Decode／Verify 阶段分类、八卡完整性、双 slot 请求归属、拒绝逐轮 fence、既有层映射和反馈协议。
2. 一次 Guard lease，EP8／PP 顺序运行，每组一次预热、两次未插桩生产 cohort，再独立跑一次无逐轮同步的时间线。
3. 所有 ModelRunner.forward（包括初始 eager Prefill）和 Scheduler.run_batch 分配独立 CUDA Event。Graph 外层事件独立记录。仅最终 idle 才同步和读取时间戳。
4. 时间线与生产输出逐请求一致；两次生产重复一致；与既有同布局参照逐请求一致。跨布局输出严格等价尚未完成资格验证，不作无损 MTP 或跨布局质量结论。
5. 记录 PP 接收、发送、反馈、结果处理及通信 Work 等待的本地 CPU 区间。不同节点的 CPU 时钟不对齐；不同 GPU 的事件时钟不相加。

## 分析定义

对每条固定64输出的请求：

`TPOT = (最后输出时间－首输出时间) / 63`

`TPOT = 首个输出后到第一次 Verify 输出的间隔 / 63 + 其余时间 / 63`

第一段是启动过渡，不能把它与稳态每次 Verify 的周期混为一谈。中段 SSE 间隔还受每次验证推进 token 数影响，不等于单 token GPU 计算时间。

`analyze.py` 从原始事件重算上述恒等式，并读取本轮实际时间线。输出 `causal_evidence.json`，所有来源绑定 SHA256。Graph、forward、batch 是嵌套区间，不能相加。Graph 未覆盖的 current-stream 区间可能包含边界、采样、元数据、同步与空档；本诊断不把它们全部当作设备空闲。

完整结果与清理证据必须以本轮 `runs/paired_r1/task/results.json`、`task/cleanup.json` 和 Guard postcheck 为准；在这些证据齐备前，本文件不表示实验完成。
