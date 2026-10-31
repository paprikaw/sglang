## 01 什么时候用 EP，什么时候用 PP？

本报告综合历史端到端测量与本轮35B Decode补测。目标是形成有证据的部署判断，而非宣布通用最优布局。平台含原生后端、Graph和overlap差异。

来源：

- [native_metrics.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/a100_r2/analysis/native_metrics.csv)
- [components.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/a100_r2/analysis/components.csv)
- [native_metrics.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/h100_r2/analysis/native_metrics.csv)
- [components.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/h100_r2/analysis/components.csv)
- [moderate_comparison.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen35b_a100_20261025/runs/a100_qwen35b_ep2_pp2_r2/analysis/moderate_comparison.csv)
- [prefill_components.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen35b_h100_20261025/runs/h100_qwen35b_ep2_pp2_r3/analysis/prefill_components.csv)
- [decode_components.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen35b_h100_20261025/runs/h100_qwen35b_ep2_pp2_r3/analysis/decode_components.csv)
- [full_metrics.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen35b_h100_20261025/runs/h100_qwen35b_ep2_pp2_r3/analysis/full_metrics.csv)
- [comparison.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_h100_20261025/runs/h100_prefill_components_r1/analysis/comparison.csv)
- [qwen_ep2_pp2_prefill_profiles.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_ep2_pp2_prefill_profiles.csv)
- [REPORT.md](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_prefill_diagnosis/REPORT.md)
- [Q02.md](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q02.md)

## 02 固定两卡：总专家权重相近，每层工作集不同

均衡层、专家同尺寸的示意。EP每卡40层×128专家=5120专家块；PP每卡20层×256专家=5120专家块。Attention复制、KV和通信/Graph缓冲不同，不能据此声称总显存完全相等。EP的Attention为TP1×DP2，PP为分层TP1，两槽每槽最多4请求。

来源：

- [launch_manifest.json](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/a100_r2/launch_manifest.json)
- [launch_manifest.json](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/h100_r2/launch_manifest.json)

## 03 历史端到端结果：同样两卡，没有固定赢家

三轮吞吐均值。PP均为两槽。35B为BF16、122B为FP8检查点；A100与H100的Expert/Attention后端及Graph/overlap不同。不能将平台对照解读为纯硬件因果。35B H100 Prefill EP方波动较大，122B A100的Prefill小差距也只作线索。

来源：

- [moderate_comparison.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen35b_a100_20261025/runs/a100_qwen35b_ep2_pp2_r2/analysis/moderate_comparison.csv)

## 04 本轮补测：稳态 Decode 的选择发生反转

各轮8请求，真实模型GPU前缀4096，新输入1，输出128。A100保留Triton/eager/无overlap；H100保留DeepGEMM/FA3/Decode Graph及原生EP overlap。H100 r1首轮TTFT异常及未生效诊断保留，r2补齐本地1/2/3/4形状预热后重新测全部三轮，不从同轮挑选样本。

来源：

- [native_metrics.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/a100_r2/analysis/native_metrics.csv)
- [native_metrics.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/h100_r2/analysis/native_metrics.csv)

## 05 A100：Attention 相近，专家通信代价明显

每配置3个独立profile、每profile2 ranks；每rank丢弃最初4个Decode forward后选择24个完整B4 forward，EP每rank960层调用、PP每rank480。误差条是3份profile卡均值的样本标准差，非同一trace内核重复的独立样本。PP边界收发单列，不记为零通信。

来源：

- [components.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/a100_r2/analysis/components.csv)
- [validation.json](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/a100_r2/analysis/validation.json)

## 06 H100：保留原生 Graph，检查专家与通信的变化

只使用实际cudaGraphLaunch correlation匹配的Decode GPU kernel；每次核对Full/Linear Attention层数。PP以每层归一化，避免将20层stage与40层EP直接比总时间。其他项保留投影、共享专家、路由等，不能把它们视作不重要。PP边界在Graph外，另行统计驻留。

来源：

- [components.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/h100_r2/analysis/components.csv)
- [validation.json](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/h100_r2/analysis/validation.json)

## 07 补充完整模块范围：H100 的 eager 诊断

三次独立eager诊断，每rank选24个稳态B4 forward；必须没有整模型cudaGraphLaunch，并有40/20层完整CPU组件注解。正式TPOT只在原生Graph路径测量。该诊断用于查看含投影的Attention与完整Routed Expert范围，不将eager绝对耗时套到Graph端到端。

来源：

- [components.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/h100_r2/analysis/components.csv)
- [components_per_rank.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/h100_r2/analysis/components_per_rank.csv)

## 08 模型维度：更大的 Expert 成本会吃掉通信收益

Expert包含Routed+Shared，Attention包含投影。35B与122B分别为BF16与FP8、不同架构，不能隔离参数规模因果。图中的Expert增量和D/C活动来自重叠GPU组件，不能相减后称作净端到端收益。35B EP Prefill吞吐三轮波动大，14%与6%只作方向性线索。

来源：

- [prefill_components.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen35b_h100_20261025/runs/h100_qwen35b_ep2_pp2_r3/analysis/prefill_components.csv)
- [comparison.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_h100_20261025/runs/h100_prefill_components_r1/analysis/comparison.csv)

## 09 workload 维度：Attention 仍然必须保留

两条长缓存点总旧缓存均为2879488 tokens，但16请求与32请求的单条上下文、总新token数不同，不是固定上下文的纯batch sweep。右图为16请求、每条缓存179968的EP2，实际Graph本地batch8，包含40层；仅Attention核心、Routed GEMM、D/C累计内核活动，不能相加为TPOT。

来源：

- [full_metrics.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen35b_h100_20261025/runs/h100_qwen35b_ep2_pp2_r3/analysis/full_metrics.csv)
- [decode_components.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen35b_h100_20261025/runs/h100_qwen35b_ep2_pp2_r3/analysis/decode_components.csv)

## 10 一个必要的修正：通信也会间接拖慢 Attention

数据来自第35层同一kernel形状、同一调用顺序的观测。投影无NCCL重叠0.7718ms、高重叠1.2679ms；FullAttention核心5.0535与5.0619ms。相关性支持通信驻留与SM资源干扰解释，未测精确SM占用和频率，也不是移除通信因果A/B。已有就绪后发送Prefill优化15.08%另有正确性验收；不外推Decode。

来源：

- [REPORT.md](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_prefill_diagnosis/REPORT.md)
- [Q02.md](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q02.md)

## 11 把 insight 落成一个简洁的部署决策流程

这是由测量支持的候选筛选流程，并非已训练或验证的自动预测器。每个候选都固定Attention TP/DP、微批次、Graph、后端及精度。通信曲线要按实际域、实际payload采样；当前补测仅两卡NVLink连接，不给出跨节点EP阈值。最终目标可选Prefill TTFT/吞吐，或Decode TPOT/吞吐与SLO。

来源：

- [native_metrics.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/a100_r2/analysis/native_metrics.csv)
- [native_metrics.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/h100_r2/analysis/native_metrics.csv)

## 12 附录：端到端复测、证据范围与数据入口

A100 a100_r2与H100 h100_r2均要求Guard postcheck和清理通过。H100 r1正式记录仍保留，首轮TTFT异常与错误标记的eager诊断未纳入新结果。另有准备阶段a100_r1仅CPU失败，无GPU测量。PPT与PDF、全部图表源数据、逐卡统计、验证报告及生成脚本一同交付。不同模型、阶段、缓存和Graph范围不合并。

来源：

- [native_metrics.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/a100_r2/analysis/native_metrics.csv)
- [components.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/a100_r2/analysis/components.csv)
- [native_metrics.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/h100_r2/analysis/native_metrics.csv)
- [components.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs/h100_r2/analysis/components.csv)
- [moderate_comparison.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen35b_a100_20261025/runs/a100_qwen35b_ep2_pp2_r2/analysis/moderate_comparison.csv)
- [prefill_components.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen35b_h100_20261025/runs/h100_qwen35b_ep2_pp2_r3/analysis/prefill_components.csv)
- [decode_components.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen35b_h100_20261025/runs/h100_qwen35b_ep2_pp2_r3/analysis/decode_components.csv)
- [full_metrics.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen35b_h100_20261025/runs/h100_qwen35b_ep2_pp2_r3/analysis/full_metrics.csv)
- [comparison.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_h100_20261025/runs/h100_prefill_components_r1/analysis/comparison.csv)
- [qwen_ep2_pp2_prefill_profiles.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_ep2_pp2_prefill_profiles.csv)
- [REPORT.md](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_prefill_diagnosis/REPORT.md)
- [Q02.md](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q02.md)
