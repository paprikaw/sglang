# Q01：两张 A100 上的 Qwen3.5-122B，EP2 与 PP2

归档日期：2026-10-25。读取已经完成的测量，不重新运行 GPU。四槽与两槽是不同的 PP 调度设置，分别保留。

## 1. 术语、数量和计时边界

本报告不使用单个字母代替请求数、上下文长度或输入长度。后文缩写与术语的确切含义如下。

| 名称 | 本报告的含义与范围 |
|---|---|
| token | 模型分词后的单位；输入、已缓存上下文和生成输出分开计数，不是字符数。 |
| Prefill | 本轮有 8 个请求；每条没有已缓存上下文，处理 4096 个新输入 token，只生成 1 token。整批新输入共 32768 token。 |
| Decode | 本轮有 8 个请求；每条已有 4096 token 的真实模型上下文缓存在 GPU，新增输入 1 token，生成 128 token。前缀构建不计时。 |
| EP／专家并行 | 同一层的路由专家分布在几张 GPU 上。EP2 为两卡共享该层的 256 个路由专家，每卡 128 个；不是两个独立模型副本。 |
| PP／流水并行、stage | 模型按层拆为前后阶段。PP2 每卡负责连续 24 层，每个已分配层含全部 256 个路由专家；一条请求仍经过两阶段。 |
| TP／张量并行、DP／数据并行 | TP 拆分同层张量计算；DP 分配不同请求。EP2 的 Attention 为 TP1×DP2，每卡处理自己的请求；PP2 每阶段 Attention 为 TP1。 |
| 流水槽位、microbatch、chunk | 槽位是调度器轮转维护的批次位置，不是 GPU 或 stage 数。microbatch 是一次流水处理的请求子集，本轮最多 4 请求；Prefill 实际每次每卡处理 1 请求的 4096 个新输入 token。chunk 是单次输入分块。 |
| 四槽／两槽 | 原生循环槽数＝PP stage 数＋额外异步深度。额外深度 2 对应四槽；额外深度 0 对应两槽。槽数不证明持续满流水。 |
| Attention | 注意力及本轮注解内的投影；模型有 36 层 GatedDeltaNet 线性 Attention、12 层 Full Attention。下表按实际层调用加权。 |
| Routed／Shared Expert | Routed 是路由选择的专家计算，每 token 选择 8 个专家；Shared 是共享专家计算。两者分别统计。 |
| Dispatch／Combine | 前者将 token 发到专家所在卡，后者取回并合并结果。PP 没有跨卡专家交换，但仍有阶段间激活传输、反馈与同步。 |
| TTFT | 客户端放行该请求至收到第一个输出 token 的时间，包含排队与初始输入处理，不是纯 GPU 计算时间。 |
| TPOT | 首输出 token 至末输出 token 的时间除以输出数量减一；单位毫秒。只输出 1 token 的 Prefill 点不适用，留空而非写 0。 |
| 端到端延迟／吞吐 | 延迟从请求放行至最后输出。每轮吞吐为全批输入或输出 token 数除以整批完成耗时，再对三轮吞吐取平均；不是平均请求延迟的倒数。 |
| mean／median／P99 | 分别是均值、中位数、99 百分位数。延迟汇总 3 轮×8 请求共 24 个观测；最近秩法 P99 等于样本最大值，不代表稳定服务尾延迟。 |
| FP8／W8A16／BF16 | 序列化权重为 8 位浮点；Marlin 以 8 位权重、16 位激活运行；激活和 KV 均为 BF16。A100 不使用原生 FP8 Tensor Core。 |
| GPU 活动时间 | 通过真实 launch correlation 将 kernel 归属到组件，计算活动区间并集。不同组件可重叠；CPU 范围、NCCL kernel 驻留与网卡传输耗时不是同一个量。 |

## 2. 设备、模型与公平性

完整 Qwen3.5-122B-A10B-FP8 文本模型，48 层、每层 256 个路由专家，序列化张量约 118.42 GiB；不截层。固定节点 spartan-gpgpu069 的 GPU0／1，两张 A100 80GB，经 NV12 NVLink 直连。两个布局顺序运行。

两组均使用 Marlin、Triton Attention，关闭 CUDA Graph 和 overlap schedule。EP2 为 DeepEP normal 的原生 NVLink 路径；不是之前跨节点的 CE、PCIe 或 RDMA low-latency 对照。PP2 无远端专家交换。完整权重从 canonical CPU bank 复用；加载、首次编译和预热不计时。

每项三轮正式测量，计时不开 profiler；profile 单独采集。Prefill 每轮清空原生 GPU 前缀缓存，Decode 每轮复用同样的真实前缀。本轮容量充足，没有合成 KV、CPU HiCache 换入或容量不足换出。两槽 PP 是后续补测，复用原 EP2 基线，不是一次同时重跑全部配置。

## 3. Prefill：包含约 8811 token/s 的四槽结果

延迟单元格均为 **均值 / 中位数 / P99**；单位秒。每条只输出 1 token，所以 TTFT 等于端到端延迟。

| 配置 | 输入吞吐 token/s ↑ | TTFT 秒 ↓ | 端到端秒 ↓ | TPOT |
|---|---:|---|---|---|
| EP2 | 8670.75 | 2.3224 / 2.4141 / 3.9419 | 2.3224 / 2.4141 / 3.9419 | 不适用 |
| PP2，四槽 | 8810.58 | 2.5028 / 2.5374 / 3.7275 | 2.5028 / 2.5374 / 3.7275 | 不适用 |
| PP2，两槽 | 8185.11 | 2.3892 / 2.3888 / 4.0201 | 2.3892 / 2.3888 / 4.0201 | 不适用 |

三轮输入吞吐分别如下；轮号只表示各配置自己的重复测量，不是交错配对。

| 配置 | 第 1 轮 | 第 2 轮 | 第 3 轮 |
|---|---:|---:|---:|
| EP2 | 8319.33 | 9381.56 | 8311.37 |
| PP2，四槽 | 8831.24 | 8811.24 | 8789.27 |
| PP2，两槽 | 8213.45 | 8192.40 | 8149.48 |

四槽 PP 输入吞吐比 EP2 高 1.61%，平均 TTFT 则高 7.77%；其最后一条请求较早完成，不代表每条请求都更早。EP2 三轮有波动，不能据 1.61% 判定 PP 稳定胜出。两槽 PP 的 Prefill 吞吐低于 EP2 5.60%，不能把两槽数据替换四槽后继续使用同一结论。

## 4. Prefill：对应每卡每层的计算与通信

下表对两张卡的匹配本地形状调用取均值，单位 **毫秒／每卡／一次实际层调用**。每条调用处理 1 个请求的 4096 个新输入 token；EP 还可能接收另一张卡发来的专家 token，因此不是严格相同全局输入的隔离 Expert 微基准。Attention 按 12 层 Full、36 层线性类型加权。

| 配置 | Attention | Routed Expert | Shared Expert | Dispatch | Combine |
|---|---:|---:|---:|---:|---:|
| EP2 | 7.6347 | 6.7216 | 0.5912 | 0.7606 | 0.9489 |
| PP2，四槽 | 8.2377 | 7.2201 | 0.6566 | 0.0000 | 0.0000 |
| PP2，两槽 | 8.4920 | 7.6374 | 0.6870 | 0.0000 | 0.0000 |

四槽 PP 的 Routed Expert 增加约 0.498 ms，Attention 增加约 0.603 ms，同时避免约 1.710 ms 的逐层专家通信。但 PP 阶段传输在层范围外，GPU 分项可重叠，以上数字不能直接相加得到端到端关键路径。

每层本地专家从 128 增到 256、每个专家收到的 token 行数改变，可能增加权重读取和小矩阵成本；这是与观察相符的解释，不是已经测得的 HBM 带宽瓶颈。真实路由、不同层和通信／计算重叠尚未完全隔离。只改槽数就改变 Prefill 排名，也说明不能把差距全部归因于专家权重。

## 5. Decode：保留相同实验的槽位对照

每条 GPU 已缓存 4096 token，新增 1 token，输出 128 token。延迟仍为 **均值 / 中位数 / P99**。这是另一种 workload，不能与上一节 Prefill 的毫秒／层直接相减。

| 配置 | 输出吞吐 token/s ↑ | TTFT 秒 ↓ | TPOT 毫秒 ↓ | 端到端秒 ↓ |
|---|---:|---|---|---|
| EP2 | 72.08 | 0.2780 / 0.2746 / 0.3778 | 109.6631 / 109.6738 / 110.8126 | 14.2052 / 14.2069 / 14.3029 |
| PP2，四槽 | 49.04 | 0.4368 / 0.2973 / 0.8634 | 160.9375 / 160.7455 / 163.4925 | 20.8758 / 20.7452 / 21.2235 |
| PP2，两槽 | 97.26 | 0.1739 / 0.1622 / 0.2153 | 81.3724 / 81.3033 / 82.4480 | 10.5082 / 10.5082 / 10.5783 |

两槽 PP 的 Decode 吞吐比 EP2 高 34.92%、平均 TPOT 低 25.80%；四槽 PP 的 Decode 则明显更差。Prefill 与 Decode 的合适槽位可能不同；本轮没有做完整槽位调参或容量边界扫描。

## 6. 来源、状态与可复查入口

原始 r7 完成测量、cache 审计及 GPU 清理，但退出包装器误捕获正常 SystemExit(0)，原始 Guard 状态仍保留为失败。不能把它写成 Guard 全通过；本归档独立复算六份测量的 24 请求统计与三轮吞吐，并核对 profile 原始 trace 哈希。最终两槽 r8 的 Guard、cache 审计和清理均通过。没有修改原始运行记录。

- [请求指标 CSV，六条独立配置×workload 记录](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_ep2_pp2_metrics.csv)；[Prefill 逐卡分项 CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_ep2_pp2_prefill_profiles.csv)。
- [来源路径与 SHA-256 清单](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_ep2_pp2_archive.json)；[本次归档校验](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_ep2_pp2_validation.json)。
- [原完整实验报告](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/results/summary.md)；[原逐层表](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/results/stage_metrics.csv)；[原测量完整性检查](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/results/data_validation.json)。
- EP2：[Prefill 原始测量](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/runs/qwen122_ep2_pp2_r7/ep2/metrics.prefill.json)；[Decode 原始测量](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/runs/qwen122_ep2_pp2_r7/ep2/metrics.decode.json)；[冻结配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/runs/qwen122_ep2_pp2_r7/config.json)。
- PP2，四槽：[Prefill 原始测量](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/runs/qwen122_ep2_pp2_r7/pp2/metrics.prefill.json)；[Decode 原始测量](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/runs/qwen122_ep2_pp2_r7/pp2/metrics.decode.json)；[冻结配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/runs/qwen122_ep2_pp2_r7/config.json)。
- PP2，两槽：[Prefill 原始测量](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/runs/qwen122_pp2_two_slots_r8/pp2/metrics.prefill.json)；[Decode 原始测量](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/runs/qwen122_pp2_two_slots_r8/pp2/metrics.decode.json)；[冻结配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/runs/qwen122_pp2_two_slots_r8/config.json)。

### 进一步解释 Prefill 的吞吐与 Attention 慢点

[详细诊断：Attention TP、槽位、请求完成分布与 NCCL／投影重叠](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_prefill_diagnosis/REPORT.md)。这是对已有 trace 的 CPU 分析，不是新 GPU 测量；保留原报告和原始 Guard 状态。
