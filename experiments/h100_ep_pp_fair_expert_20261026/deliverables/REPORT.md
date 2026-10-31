# H100：双方应用小批量 Expert 优化后的 EP2 / PP2 对照

同一节点 spartan-gpgpu178 的 NVLink GPU0/1，完整 Qwen3.6-35B-A3B BF16，固定全局8请求。EP2采用专家并行2路、Attention数据并行2路且每路Attention TP1；PP2采用两个20层stage、两个流水槽位，每个微批次最多4请求。整节点4GPU租约，实际模型始终使用同一对2GPU；四个配置顺序执行。

## 结果

双方都尝试现有 Triton 小批量 Expert GEMM 家族后，Decode吞吐变化为：EP2 **+3.06%**，PP2 **+18.61%**。本轮测试候选中，EP2较快实现为 `triton_small`（886.9 tok/s），PP2为 `triton_small`（790.2 tok/s）；前者相对后者的吞吐变化为 **+12.23%**。这是本轮候选比较，不是穷尽调优后的性能上限。

每个配置、每个阶段5轮正式cohort。吞吐列为cohort均值±样本标准差；延迟为请求均值。Prefill：每请求无缓存、新输入4096 token、输出1 token。Decode：每请求真实GPU缓存4096 token、新输入1 token、输出128 token。吞吐以全部8请求完成的cohort墙钟时间计算。TTFT为请求开始到首token，TPOT为首token之后每个输出token的平均间隔，E2E为请求开始到完成。

| 配置 | Prefill input tok/s ↑ | Decode output tok/s ↑ | Decode TPOT ms ↓ | Decode E2E ms ↓ | Decode TTFT ms ↓ |
|---|---:|---:|---:|---:|---:|
| EP2 native | 65,069.4 ± 3,329.0 | 860.5 ± 4.4 | 7.956 | 1189.3 | 178.9 |
| EP2 small-batch Triton | 63,181.6 ± 3,979.5 | 886.9 ± 4.1 | 7.663 | 1153.9 | 180.7 |
| PP2 native | 78,687.5 ± 109.6 | 666.3 ± 1.5 | 11.184 | 1533.6 | 113.2 |
| PP2 small-batch Triton | 78,619.6 ± 145.3 | 790.2 ± 8.9 | 9.277 | 1293.0 | 114.9 |

Prefill的大形状继续使用DeepGEMM。本轮没有观察到Prefill提速；EP均值波动较大，两个实现的Prefill中位数接近，不能把该差异归因为小批量GEMM优化。

[逐轮CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/analysis/rounds.csv)；[逐请求CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/analysis/per_request.csv)；[完整指标CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/analysis/metrics.csv)。所有轮次均保留；未因快慢剔除cohort。

## Expert、Attention和通信

下面为真实Decode CUDA Graph内的GPU kernel时间，单位为µs/GPU/实际层调用。每rank选择稳定本地B4的16个forward，并排除最初4个。Attention仅core，Expert仅路由GEMM；索引适配kernel单列。Dispatch/Combine保留原有DeepEP路径，GPU事件可包含等待和重叠；这些列不能直接相加解释E2E。Prefill全模块累计数据另存components.csv，计时范围不同。

| 配置 | Attention core | Expert GEMMs | Dispatch | Combine | EP新增索引kernel |
|---|---:|---:|---:|---:|---:|
| EP2 native | 13.04 | 59.60 | 15.26 | 17.89 | 0.00 |
| EP2 small-batch Triton | 12.92 | 47.74 | 15.26 | 18.81 | 3.22 |
| PP2 native | 12.47 | 94.39 | 0.00 | 0.00 | 0.00 |
| PP2 small-batch Triton | 12.37 | 57.31 | 0.00 | 0.00 | 0.00 |

![Matched throughput and component comparison](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/deliverables/fair_ep_pp_comparison.png)

## 什么优化应用到了双方

PP沿用之前的策略：实际本地token数≤4时使用已有BF16 Triton Expert执行，大形状继续DeepGEMM。EP也在DeepEP low-latency、本地输入token数≤4时调用**同一个 `invoke_fused_moe_kernel` GEMM实现**，按自身专家数和逻辑token数选择现有小批量tile配置。

EP新增GPU索引适配器，将DeepEP的专家分组buffer及每专家有效行数转换为共享GEMM所需的索引；输出保持原有专家分组布局，交回原生DeepEP加权combine。没有改变路由、专家选择、BF16权重/激活/KV、FP32 GDN状态、Attention、通信二进制、流水槽位或chunk。索引kernel和全部适配执行成本都包含在EP端到端计时中。EP保留原生masked SwiGLU激活；未增加通信或转回其他通信后端。

因此，这轮小批量内核选择属于EP和PP均可尝试的通用优化。此前PP自身提速的证据继续成立；研究中的PP专有贡献需要进一步利用stage局部性、流水调度或微批次组织，并与两边经过同等优化机会的基线比较。不能只拿PP的目标估算与未优化EP作最终胜负判断。

## 正确性与真实路由

启动时每rank使用真实第一层权重检查：0/1/3/15/16/17/64行边界、全部无活跃专家、全部128个本地专家活跃。索引验证覆盖每一个有效assignment，检查没有遗漏、重复或错误专家映射。真实运行另检查两rank各40层；本轮选定运行的最大相对L2误差为 0.000000（相同1.5%层误差门限）。以下assignment数是实际到达本GPU的token-expert配对数，active experts是本GPU该次调用实际触及的不同专家数。

| 运行 | 本地B4层样本数 | 平均assignments/GPU/层 | 平均active experts/GPU/层 | 最大相对L2 |
|---|---:|---:|---:|---:|
| fair_native_20261026_v3 | 80 | 32.00 | 10.84 | 0.000000 |
| fair_triton_20261026_v1 | 80 | 32.00 | 10.88 | 0.000000 |

PP在相同真实输入与权重上的原生/Triton层输出对照也通过相同1.5%相对L2门限，本轮最大相对L2为 0.003270。[PP数值核验](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/analysis/pp_numerical_validation.json)。单层诊断microbenchmark不用于上面的正式吞吐或CUDA Graph分项值。

独立的8-token概率验证使用原子请求批次：EP每个Attention DP副本4请求，PP输入8请求并由原生流水切为B4。每配置3轮，比较相同提示及完全相同生成历史处的top-10自然对数概率；下表同历史位置之外不进行概率差值比较。

| 比较 | 相同历史位置/候选位置 | 相同argmax/相同历史位置 | top-10平均交集 | 平均绝对logprob差 |
|---|---:|---:|---:|---:|
| fair_native_20261026_v3.ep2.r0 | 64/64 | 64/64 | 10.00 | 0.00000 |
| fair_native_20261026_v3.ep2.r1 | 64/64 | 64/64 | 10.00 | 0.00000 |
| fair_native_20261026_v3.ep2.r2 | 64/64 | 64/64 | 9.73 | 0.10634 |
| fair_triton_20261026_v1.ep2.r0 | 64/64 | 64/64 | 9.73 | 0.10634 |
| fair_triton_20261026_v1.ep2.r1 | 64/64 | 64/64 | 9.73 | 0.10634 |
| fair_triton_20261026_v1.ep2.r2 | 64/64 | 64/64 | 9.73 | 0.10634 |
| fair_native_20261026_v3.pp2.r0 | 64/64 | 64/64 | 10.00 | 0.00000 |
| fair_native_20261026_v3.pp2.r1 | 64/64 | 64/64 | 10.00 | 0.00000 |
| fair_native_20261026_v3.pp2.r2 | 64/64 | 64/64 | 10.00 | 0.00000 |
| fair_triton_20261026_v1.pp2.r0 | 55/64 | 53/55 | 9.65 | 0.08916 |
| fair_triton_20261026_v1.pp2.r1 | 55/64 | 53/55 | 9.65 | 0.08916 |
| fair_triton_20261026_v1.pp2.r2 | 55/64 | 53/55 | 9.65 | 0.08916 |

EP原生自对照的最后一轮已有logprob变化；优化版三轮与原生最后一轮的64个验证位置是否全部token选择及top-10 logprob完全一致：**True**。因此不能将相对原生第一轮的该项概率变化单独归因于Expert优化。[原生自对照补充核验](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/analysis/probability_native_last_reference.json)。BF16执行路径不自动等于完整生成逐位一致；本轮没有任务级质量评价。PP之前的精度边界仍按本轮概率结果呈现。

## 测量与来源

模型初始化、JIT、图捕获、缓存准备、路由诊断、数值测试和profiler均在正式计时之外。CPU KV注册表没有该Qwen混合GDN模型的兼容项，使用原生模型生成的GPU radix前缀与GDN状态，没有CPU KV恢复或伪造缓存。权重来自Guard认证的完整模型CPU bank。

引擎 `870269702285d94bdd926a50cc29ad1a20bcf60c`，不可变源tag `experiment/sglang/glm47-ep16-canonical-profile-20261004-cpu-validated`，引擎工作树当前clean。模型revision `995ad96eacd98c81ed38be0c5b274b04031597b0`。实际DeepEP native `/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/runtime/deepep_h100_custom_ep8_20261007_r2/native12/python/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so`，SHA256 `d9b26ac86748517800ce9f9734d865be20d089d473d9d4f6d66dde98c673d76b`；wrapper `/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.third_party/deepep-pcie-coalesced-20261006`。每个run的私有代码和输入均独立冻结。native与optimized之间另有一处仅在untimed microbenchmark保留输入storage的修正，正式推理路径不受该修正影响。

主对照运行为 `fair_native_20261026_v3` 和 `fair_triton_20261026_v1`。最初`v1`仅准备、未启动；`native v2`因适配函数关键字参数名不匹配在启动阶段失败，没有正式性能数据，已修复并在新目录重跑。所有已启动尝试均保留Guard终态与清理证据；[审计记录](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/analysis/run_audit.json)；[概率验证](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/analysis/probability_validation.json)；[完整EP数值资格检查](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/analysis/ep_qualifications.json)；[真实EP路由与内核对照](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/analysis/ep_routing.json)。

复现：先用 `prepare.py` 创建全新run，再通过原有Guard监督的 `h100_pp_expert_fragmentation_20261026/launch.py` 启动。重新确认booking与缓存；保留新run目录与冻结输入。离线分析运行 `analyze.py`，再运行 `finalize.py --native <native-run> --optimized <optimized-run>`。不修改已完成run。
