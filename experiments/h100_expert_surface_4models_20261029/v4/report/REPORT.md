# H100 四模型：逐 GPU 的绝对原始耗时

更新时间：2026-10-29T13:00:20.502683+11:00。所有图表单位均为 **毫秒（ms）**。

每个格子直接使用该 GPU 已保存的第 **1 / 30** 次 CUDA event 计时；对应第 1 个计时块中的第 1 次。不做归一化，不计算均值或中位数，不合并两张 GPU。展示保留四位小数，CSV 保留原始精度。

T 是单卡实际接收的 token–expert 配对行数，E 是有输入的活跃专家数；采用均衡路由。测真实单层 Expert 的两次 GEMM、SiLU/multiply 及 FP8 内部激活量化；输入准备、初始量化、Attention、router、shared expert、通信及模型加载不计时。

## CUDA Graph：两次独立测量分别展示

### 独立测量 1：spartan-gpgpu011

![绝对原始耗时](absolute_raw_cuda_graph_rep1.png)

| 模型 | GPU | T128 E2 | T128 E32 | T8192 E2 | T8192 E32 | T32768 E2 | T32768 E32 |
|---|---|---|---|---|---|---|---|
| GLM-4.7 AWQ W4A16 (BZP TMA off) | 0 | 0.0625 | 0.2070 | 0.6591 | 0.6764 | 2.5466 | 2.4800 |
| HY3 BF16 | 1 | 0.0397 | 0.4971 | 0.4281 | 0.5818 | 1.7506 | 1.7725 |
| Qwen3.6-35B-A3B BF16 | 2 | 0.0162 | 0.1007 | 0.0920 | 0.1245 | 0.3207 | 0.3481 |
| Qwen3.5-122B-A10B FP8 | 3 | 0.0194 | 0.1634 | 0.1580 | 0.2130 | 0.5524 | 0.5939 |

### 独立测量 2：spartan-gpgpu171

![绝对原始耗时](absolute_raw_cuda_graph_rep2.png)

| 模型 | GPU | T128 E2 | T128 E32 | T8192 E2 | T8192 E32 | T32768 E2 | T32768 E32 |
|---|---|---|---|---|---|---|---|
| GLM-4.7 AWQ W4A16 (BZP TMA off) | 2 | 0.0628 | 0.2096 | 0.6640 | 0.6826 | 2.3953 | 2.4163 |
| HY3 BF16 | 3 | 0.0404 | 0.5012 | 0.4458 | 0.5882 | 1.7834 | 1.9759 |
| Qwen3.6-35B-A3B BF16 | 0 | 0.0169 | 0.1006 | 0.0916 | 0.1247 | 0.3238 | 0.3494 |
| Qwen3.5-122B-A10B FP8 | 1 | 0.0188 | 0.1638 | 0.1585 | 0.2125 | 0.5574 | 0.5991 |

## 完整数值与其他执行模式

- [当前展示的每格单次计时 CSV](absolute_single_event.csv)。
- [每格全部 30 次原始计时 CSV](absolute_all_events.csv)。
- [eager：独立测量 1](absolute_raw_eager_rep1.png) · [独立测量 2](absolute_raw_eager_rep2.png)。
- [CUDA Graph PDF：测量 1](absolute_raw_cuda_graph_rep1.pdf) · [测量 2](absolute_raw_cuda_graph_rep2.pdf)。
- [展示口径与节点/GPU 映射](absolute_raw_view.json) · [来源校验](sources.json)。

## 配置与验证

GLM-4.7 为 AWQ/Humming grouped masked，驻留 40 个真实专家；使用 `use_tma_bzp=False` 兼容配置绕过 grouped 零点参数的 TMA 描述符错误。HY3 为 BF16 DeepGEMM，驻留 48 个专家；35B 为 BF16 DeepGEMM，122B 为 FP8 DeepGEMM，均驻留 256 个专家。DeepGEMM 每专家按 128 行对齐，补齐开销包含在测量中。不同模型使用不同精度和内核。

沿用已经通过数值检查和 Guard 清理检查的原实验；这里只改变已有数据的展示方式。原始结果、冻结配置、计时顺序、模型权重摘要和源版本均保持原样。

[原始结果](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_expert_surface_4models_20261029/v4/runs/r6/profile/result.json) · [冻结配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_expert_surface_4models_20261029/v4/runs/r6/dashboard_spec.json)。
