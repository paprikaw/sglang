# H100：四个模型的 Expert token × 活跃专家数二维测量

生成时间：2026-10-29T12:22:48.315643+11:00。

## 口径

- T 是单卡实际接收的 token–expert 配对行数，取 8 到 32768 的 2 次幂。它不是请求 batch、全局 Prefill token 总数或缓存上下文长度。同一个原始 token 路由到八个专家会形成八行，此处直接控制配对行。
- E 是有非零输入的真实 routed experts 数。路由均衡，每个活跃专家处理 floor(T/E) 或 ceil(T/E) 行；E 大于 T 的格子不测。每个模型始终加载相同的一组 resident 专家。
- 测真实一层的 W13 GEMM、SiLU/multiply、W2 GEMM；FP8 路径包含两次 GEMM 之间的激活量化。初始输入量化、输入排列/对齐、权重读取/转换、JIT、router、Attention、shared expert、通信和模型加载都不计时。该结果没有 TTFT、TPOT 或完整模型吞吐的含义。
- 每个模型在两台已预约 H100 节点上各运行一个独立单 GPU 进程，八张卡并行。每点每种模式 5 次准备预热，3 个旋转/逆序计时块、每块 3 次即时预热及 10 次 CUDA event，共 30 个事件样本。事件样本不是 30 次独立进程。
- eager 与 CUDA Graph 数据分开保存。图表采用两份独立进程的 Graph 中位数的算术平均，不把 60 个事件视为 60 次独立实验。
- GLM-4.7 使用 AWQ/Humming grouped masked；T≤8192 时保持原 A100 实验的 8192 行固定槽位，T 更大时槽位等于 T，同一 T 的所有 E 共用相同槽位大小。
- GLM 使用 Humming 已有的 `use_tma_bzp=False` 配置，绕过本环境 grouped AWQ 零点参数的 TMA 描述符错误；W13/W2 的零点参数使用备用加载路径。AWQ 权重和数值计算精度保持不变。这是明确记录的兼容配置，不代表默认 TMA 路径性能。
- HY3 和 35B 使用 H100 原生 DeepGEMM BF16 grouped contiguous；122B 使用 FP8 W8A8 grouped contiguous。预先按专家分组并做 128 行对齐；对齐产生的额外 GEMM 行已保存。它们不包含完整 EP/PP 的本地适配、masked/contiguous 切换或调度差异。
- 这是单层组件实验，未修改完整模型层数，未运行完整模型，未创建、替换或驱逐 CPU 权重/KV 缓存。

## 直接回答：小模型是否也会随 token 增大而缩小专家数差距？

以下数值均从这次通过验证的 H100 数据重算；T 是单卡接收的 token–expert 配对行数。固定 E2→E32 的共同范围：

| 模型 | T128 | T8192 | T32768 |
|---|---:|---:|---:|
| GLM-4.7 AWQ | 3.36× | 1.02× | 0.99× |
| HY3 BF16 | 12.39× | 1.34× | 1.07× |
| Qwen3.6-35B BF16 | 6.52× | 1.36× | 1.08× |
| Qwen3.5-122B FP8 | 8.59× | 1.34× | 1.08× |

数值是 E32/E2 耗时比，接近 1 表示这个专家数区间的耗时接近。应逐模型读取该表，不把曲线靠拢预设为所有后端的共同定律。

对 Qwen 的完整 E256 范围，需要保留另一个结论：

- Qwen3.6-35B BF16：T32768 时 E256/E2 仍为 **2.37×**；E256/E128 为 **1.62×**。
- Qwen3.5-122B FP8：T32768 时 E256/E2 仍为 **2.16×**；E256/E128 为 **1.56×**。

因此，适合写入 benchmark 的问题是“差距在什么 T/E 与后端配置下缩小到多少”，不能只给出“Prefill 大了就一样快”的无条件判断。

### 补齐行数与有效计算量

固定 T 时，非补齐的有效 GEMM FLOPs 相同；DeepGEMM 实际执行的补齐行数则可能随 E 改变。例如 T128：

- HY3 BF16：E2 为 256 行，E32 为 4096 行，实际补齐行数相差 16 倍。
- Qwen3.6-35B BF16：E2 为 256 行，E32 为 4096 行，实际补齐行数相差 16 倍。
- Qwen3.5-122B FP8：E2 为 256 行，E32 为 4096 行，实际补齐行数相差 16 倍。

到 T32768，E2、E32、E128、E256 均无需这种额外的 128 行补齐（在各模型已测 E 范围内）。此时残留的性能差异不能归因于额外补齐行数；它与更多专家权重、更小的单专家 GEMM 和 kernel 调度粒度等因素一致，但本实验没有把这些原因逐项隔离。

GLM 使用另一条 grouped masked AWQ 路径，不适用上面的 DeepGEMM 补齐行数解释。

### 从组件实验到 EP/PP 的边界

这些数据可以检验“Expert 服务时间不能只按有效 FLOPs 估算”，并给出单层后端的 T/E 依赖关系。将它们用于 EP/PP 选型时，还需测实际路由偏斜、各微批重复激活的专家、dispatch/combine、PP 调度与通信暴露；当前图不能独立证明某种整模型并行布局更快。

## 二维图

![Expert latency](expert_surface_ms.png)

![Relative to E1](expert_surface_relative.png)

![Expert curves](expert_curves.png)

## 所有模型的共同专家数对照

下面固定 T，报告 **E32 耗时 / E2 耗时**。越接近 1，表示这个专家数范围内耗时越接近；小于 1 时应保留其方向，不能解释为普遍等价。

| 模型 | T=128 | T=512 | T=2048 | T=8192 | T=16384 | T=32768 |
|---|---|---|---|---|---|---|
| GLM-4.7 AWQ W4A16 | 3.36× | 1.63× | 1.28× | 1.02× | 1.22× | 0.99× |
| HY3 BF16 | 12.39× | 11.21× | 4.21× | 1.34× | 1.10× | 1.07× |
| Qwen3.6-35B-A3B BF16 | 6.52× | 6.18× | 3.25× | 1.36× | 1.16× | 1.08× |
| Qwen3.5-122B-A10B FP8 | 8.59× | 7.51× | 3.24× | 1.34× | 1.14× | 1.08× |

## 与部署专家数有关的补充对照

下面只是均衡路由下的局部活跃专家数对照，并不代表已经运行 EP/PP。Qwen 的 128/256 是对应 resident 数量的一种满覆盖情形，实际 Decode 通常不会全部激活。

| 模型 | 活跃专家数 | T=128 | T=512 | T=2048 | T=8192 | T=16384 | T=32768 |
|---|---|---|---|---|---|---|---|
| GLM-4.7 AWQ W4A16 | 20 → 40 | 1.80× | 1.11× | 1.04× | 1.01× | 1.17× | 1.00× |
| HY3 BF16 | 24 → 48 | 2.03× | 2.03× | 2.03× | 1.43× | 1.14× | 1.16× |
| Qwen3.6-35B-A3B BF16 | 128 → 256 | — | 2.04× | 2.03× | 2.03× | 2.03× | 1.62× |
| Qwen3.5-122B-A10B FP8 | 128 → 256 | — | 2.01× | 2.01× | 2.01× | 2.01× | 1.56× |

## 验证与波动

两份独立进程加载的全部所选权重逐张量 SHA256 汇总一致。每个 T/E 在计时外，对第一个和最后一个活跃专家的最多两行输出进行独立 dense reference 检查。BF16/GLM 使用相对 L2≤2%，FP8 使用≤12%，并同时检查最大绝对误差与有限性；下表给出实际最大误差。

| 模型 | resident experts | 所选权重 GiB | 最大参考相对 L2 | 跨进程差距中位数 | 跨进程最大差距 |
|---|---|---|---|---|---|
| GLM-4.7 AWQ W4A16 | 40 | 0.457 | 0.68% | 0.5% | 5.1% |
| HY3 BF16 | 48 | 1.688 | 0.13% | 0.9% | 12.8% |
| Qwen3.6-35B-A3B BF16 | 256 | 1.500 | 0.21% | 0.6% | 2.3% |
| Qwen3.5-122B-A10B FP8 | 256 | 2.250 | 7.17% | 0.6% | 4.3% |

只有两份独立进程；不提供稳定置信区间或服务尾延迟结论。保留每点各进程的中位数、三块中位数和全部事件样本，以识别编译后的时钟、缓存和运行波动。

## 如何解释

固定 T 时，有效两次 GEMM 的计算量为 6×hidden_size×intermediate_size×T，不随 E 额外相乘。E 增加会改变活跃权重范围、每个专家矩阵的行数、对齐/填充与 kernel 执行粒度。因此需要逐模型、逐后端检查曲线是否靠拢，不能预设一定收敛。

原 A100 HY3 使用 Triton，这次 HY3 使用 DeepGEMM；大模型之间和不同精度之间的绝对耗时不可视为纯模型规模对照。四个模型的二维图也不能直接代替 EP/PP 端到端比较。

## 可追溯来源

- [冻结配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_expert_surface_4models_20261029/v4/runs/r6/config.json)、[计时计划](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_expert_surface_4models_20261029/v4/runs/r6/measurement_plan.json)、[逐张量权重选择](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_expert_surface_4models_20261029/v4/runs/r6/selection.json)、[节点/GPU 映射](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_expert_surface_4models_20261029/v4/runs/r6/workers.json)。
- [完整结果](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_expert_surface_4models_20261029/v4/runs/r6/profile/result.json)、[Guard 规格与源版本](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_expert_surface_4models_20261029/v4/runs/r6/dashboard_spec.json)。
- [逐进程计时 CSV](full_metrics.csv)、[两份独立进程汇总 CSV](aggregate_metrics.csv)、[来源校验](sources.json)。
- [绝对延迟 PDF](expert_surface_ms.pdf)、[相对延迟 PDF](expert_surface_relative.pdf)、[曲线 PDF](expert_curves.pdf)。
