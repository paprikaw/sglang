# Qwen3.5-122B：同一对 A100 上的 EP2 与 PP2

完整 48 层文本模型；序列化 FP8 权重约 118.42 GiB。固定 spartan-gpgpu069 的 GPU0／1，NV12；只使用已有预约。

| 配置 | 每卡模型层数 | 每层每卡 routed experts | Attention | GPU 间通信 |
|---|---:|---:|---|---|
| EP2 | 48 | 128 | TP1 × DP2 | 每层 DeepEP normal NVLink Dispatch／Combine |
| PP2，两槽 | 24 | 256 | 每 stage TP1 | stage 边界激活、反馈；无跨卡专家交换 |

两组同用 FP8 W8A16 Marlin、BF16 激活与 KV、Triton Attention；CUDA Graph 和 overlap schedule 均关闭。A100 不使用原生 FP8 Tensor Core。

| 场景 | 请求数 | 每条已缓存上下文 | 每条本轮新输入 | 每条输出 |
|---|---:|---:|---:|---:|
| Prefill | 8 | 0 | 4096 | 1 |
| Decode | 8 | 4096，真实模型生成、GPU 常驻 | 1 | 128 |

完整权重从 canonical CPU bank 加载，加载、首次编译、上下文构建和预热不计时。不使用合成 KV 或 CPU HiCache 换入；本轮容量充足。Decode 的首 token 包含新增输入的初始 Extend，TPOT 只统计首 token 之后。

## 未开启 profiler 的性能

每项 3 轮、每轮 8 请求。延迟表均给出 mean / median / P99；24 个观测的 nearest-rank P99 等于样本最大值，不代表稳定服务尾延迟。吞吐按每轮 cohort makespan 计算后取均值。

| 场景 | 配置 | 吞吐 token/s ↑ | TTFT 秒 ↓ | TPOT 毫秒 ↓ | 端到端秒 ↓ |
|---|---|---:|---|---|---|
| prefill | EP2 | 8670.75 | 2.3224 / 2.4141 / 3.9419 | — | 2.3224 / 2.4141 / 3.9419 |
| decode | EP2 | 72.08 | 0.2780 / 0.2746 / 0.3778 | 109.6631 / 109.6738 / 110.8126 | 14.2052 / 14.2069 / 14.3029 |
| prefill | PP2 two slots | 8185.11 | 2.3892 / 2.3888 / 4.0201 | — | 2.3892 / 2.3888 / 4.0201 |
| decode | PP2 two slots | 97.26 | 0.1739 / 0.1622 / 0.2153 | 81.3724 / 81.3033 / 82.4480 | 10.5082 / 10.5082 / 10.5783 |

## 本轮结论

- 固定最终两槽配置：Decode 的 PP2 吞吐比 EP2 高 34.9%，平均 TPOT 低 25.8%。
- 同样的两槽配置下，Prefill 的 EP2 吞吐比 PP2 高 5.9%。这不是所有场景下 PP 都更快的结论。
- PP 的 Decode 每层 routed Expert GPU 成本略增，Attention 接近，避免了 EP2 的逐层 Dispatch／Combine。客户端改善还包含流水和 CPU 控制路径的影响，不能把全部 TPOT 差额都归给链路时间。

### 槽位设置的独立影响

原生循环槽数为 PP stage 数＋额外异步深度。旧 `pp_async_batch_depth=2` 实际为四槽，最终设为 0 才是两槽。模型、GPU、请求及计算内核不变；下表保留旧测量，不将其混进最终两槽均值。

| 场景 | 四槽 PP2 吞吐 token/s | 两槽 PP2 吞吐 token/s | 四槽 TPOT ms | 两槽 TPOT ms |
|---|---:|---:|---:|---:|
| prefill | 8810.58 | 8185.11 | — | — |
| decode | 49.04 | 97.26 | 160.94 | 81.37 |

这也说明 Prefill 与 Decode 的槽位选择可能不同；本轮只补一个两槽对照，没有进行完整调参扫描。

## 逐层 GPU profile

独立于上述计时请求采集。原始八份 trace 的每层数据见 `stage_metrics.csv`；Attention 包括投影，routed Expert 与 shared Expert 分开。CPU 注解与 GPU 镜像注解去重，GPU kernel 由实际 launch correlation 归属。不同 forward mode、microbatch 形状不混合。

**以下为每卡、一次实际层调用的 GPU 活动均值，单位毫秒；不是每输出 token 的端到端时间。** Full Attention 只平均 Full Attention 层，linear 只平均线性 Attention 层；Expert、Dispatch、Combine 平均对应形状的全部层。

| 场景与实际本地形状 | 卡／stage | Full Attention | Linear Attention | Routed Expert | Shared Expert | Dispatch | Combine |
|---|---|---:|---:|---:|---:|---:|---:|
| Decode，每卡 4 请求 | ep2-TP-0-DP-0-EP-0 | 0.1740 | 0.1178 | 0.0745 | 0.0305 | 0.0573 | 0.0334 |
| Decode，每卡 4 请求 | ep2-TP-1-DP-1-EP-1 | 0.1749 | 0.1178 | 0.0735 | 0.0309 | 0.0786 | 0.0418 |
| Prefill，每卡 1 请求 × 4096 新 token | ep2-TP-0-DP-0-EP-0 | 9.7588 | 6.8681 | 6.5082 | 0.5870 | 0.8180 | 1.1504 |
| Prefill，每卡 1 请求 × 4096 新 token | ep2-TP-1-DP-1-EP-1 | 9.8751 | 6.9466 | 6.9351 | 0.5954 | 0.7031 | 0.7474 |
| Decode，每卡 4 请求 | pp2-TP-0-PP-0 | 0.1701 | 0.1173 | 0.0793 | 0.0305 | 0.0000 | 0.0000 |
| Decode，每卡 4 请求 | pp2-TP-0-PP-1 | 0.1731 | 0.1174 | 0.0795 | 0.0305 | 0.0000 | 0.0000 |
| Prefill，每卡 1 请求 × 4096 新 token | pp2-TP-0-PP-0 | 11.1792 | 8.4683 | 8.3176 | 0.7665 | 0.0000 | 0.0000 |
| Prefill，每卡 1 请求 × 4096 新 token | pp2-TP-0-PP-1 | 10.2378 | 7.0379 | 6.9572 | 0.6076 | 0.0000 | 0.0000 |

**PP 的表内 Dispatch／Combine 为零，只表示没有跨卡专家交换，不能理解为没有通信。** PP 的 NCCL stage 传输在层范围外；NCCL kernel 驻留时间可包含等待，不能直接当作链路传输延迟。GPU 分项也可能重叠，不能用相加值冒充客户端关键路径。

## 完整性与限制

- EP2 与 PP2 都检查完整层数；EP2 新通信接口使用真实一层、不同输入与稀疏／空目标 rank 进行数值对照。PP2 也经过真实 EP1 层 forward 检查。
- EP2 来源 r7：四个性能点及八份 trace 已完整写出，canonical cache 审计和 GPU 清理通过；退出包装器误捕获 `SystemExit(0)`，因此原始 Guard 状态仍为失败，未补写成通过。数据单独校验见 `data_validation.json`。
- 最终 PP2 来源 r8：仅补测 PP2，将额外异步槽数从 2 改成 0，使原生循环槽数从 4 改为 2；保留原 EP2 基线，不混用旧 PP 四槽结果。
- 原始 r7 的 PP 曾有 2／4／2 动态 microbatch。最终两槽的实际形状可在原始 `step[...]` 注解与逐层表中检查；不把启动参数当作持续满流水的证据。
- 此轮没有多组 GPU 型号、通信域或容量边界扫描；不能据此断言所有 EP 或 PP 部署的优劣。

## 来源

- EP2：[冻结配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/runs/qwen122_ep2_pp2_r7/config.json)，[原始测量与 traces](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/runs/qwen122_ep2_pp2_r7/ep2)。
- PP2 two slots：[冻结配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/runs/qwen122_pp2_two_slots_r8/config.json)，[原始测量与 traces](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/runs/qwen122_pp2_two_slots_r8/pp2)。
