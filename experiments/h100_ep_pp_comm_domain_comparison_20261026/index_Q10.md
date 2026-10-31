# Q10：H100 两卡与八卡的通信域、EP／PP部署取舍

这是历史数据的离线归纳，没有新GPU实验；两卡和八卡使用不同模型、不同请求条件，**不是严格相同workload的网络消融**。

- 两卡：Qwen3.5-122B-A10B FP8，H100 178内NVLink，EP2对两槽PP2。8请求；Prefill无旧缓存、每条新输入4096、输出1；Decode每条GPU已缓存4096、新输入1、生成128。
- 八卡：GLM-4.7-AWQ INT4，H100 174/175各4卡，跨节点EP8对PP2×节点内EP4。4或16请求；Prefill无旧缓存、每条新输入8192、输出1；Decode冷输入128、生成512。
- 主图采用各自cohort的EP吞吐÷PP吞吐；不是跨模型绝对吞吐排名。Prefill计新输入token/s，Decode计输出token/s，均为3轮保存结果。

两卡Decode EP2比PP2快24.87%；八卡Decode EP8比PP慢21.50%／17.12%。Prefill两卡EP比PP慢6.01%，八卡慢约69%–70%。早期八卡native库、Expert布局及本地chunk不同，不能将完整差距全归因于网络。

独立同库纯通信使用BF16、隐藏维度5120、160专家、每token命中8专家，固定全局逻辑token数，无模型计算。节点内EP4分别保留174/175；跨节点EP8的Dispatch＋Combine，小消息约5.7–6.1倍、1024-token消息约9.8–10.1倍。三次预热、三次对齐rank最大跨度采样后取中位数；EP宽度／每rank工作划分也不同，不是固定rank只更换链路。

[完整术语、12行吞吐／TTFT／TPOT／端到端指标与两张图](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_comm_domain_comparison_20261026/REPORT.md) · [原始路径与哈希](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_comm_domain_comparison_20261026/source_manifest.json) · [离线校验](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_comm_domain_comparison_20261026/validation.json)。
