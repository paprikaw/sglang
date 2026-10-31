# Q11：A100 两卡与八卡的 EP／PP 及通信域对照

历史数据的离线复核及作图，没有新 GPU 实验。**两卡／八卡不是相同模型、相同缓存或相同请求协议，不能由端到端差值唯一归因于网络。**

## 负载和术语

EP 是同层专家跨卡；PP 是模型按层分两个阶段。两卡 EP2 为 Attention TP1×DP2；PP2 每阶段 EP1，固定两槽。八卡 EP8 为 Attention TP4×DP2；PP2×EP4 每节点一个阶段。

- 两卡 A100 NV12：Qwen122B FP8 或35B BF16，8请求。Prefill 每条无旧缓存、新输入4096、输出1；Decode 每条GPU已有真实4096-token缓存、新输入1、输出128，无CPU恢复。每点三轮。
- 八卡 A100：GLM4.7 AWQ INT4，168/169，4请求。Prefill每条无旧缓存、新输入8192、输出1；Decode冷输入128、输出512。节点内Prefill chunk均1024，EP全局2048／PP全局1024；每点单cohort。pair内NVLink，pair外及跨节点走旧RDMA。
- 通信毫秒是一次 Dispatch＋Combine 的事件跨度，没有Attention、Expert计算或完整PP调度；不是请求TPOT。

| 部署与阶段 | EP 吞吐 token/s | PP 吞吐 token/s | PP 相对 EP |
|---|---:|---:|---:|
| 两卡122B Prefill | 8670.75 | 8185.11 | −5.60% |
| 两卡122B Decode | 72.08 | 97.26 | +34.92% |
| 两卡35B Prefill | 27644.14 | 31841.16 | +15.18% |
| 两卡35B Decode | 90.80 | 121.76 | +34.09% |
| 八卡GLM Prefill | 1322.77 | 1943.53 | +46.93% |
| 八卡GLM Decode | 100.06 | 101.92 | +1.85% |

本组A100没有H100那种Decode偏好翻转；八卡Decode差很小，不声称稳定优胜。原122B EP r7的Guard包装器失败状态保留，测量独立复算；后续两槽PP通过。旧四槽PP不是两槽PP，不能混用来制造反转。Q02后续Prefill优化没有新EP基线，单列在完整报告。

同库T01纯通信固定单次输入、均衡路由。1024-token时，CE策略节点内EP4为1.3348ms、跨节点EP8为7.0528ms（5.28倍）；RDMA策略则为8.4070／6.3670ms，方向相反。8-token时RDMA为0.1852／0.2846ms。EP4每卡配对行数是EP8两倍，Graph关闭，三次rank-max中位数；CE只替换节点内pair外搬运、跨节点仍RDMA。另一个节点结果和完整8–4096扫描均保存。

[完整报告：两张图、TTFT／TPOT／端到端与所有来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/a100_ep_pp_comm_domain_comparison_20261026/REPORT.md) · [12行指标](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/a100_ep_pp_comm_domain_comparison_20261026/full_metrics.csv) · [来源与哈希](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/a100_ep_pp_comm_domain_comparison_20261026/source_manifest.json) · [校验](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/a100_ep_pp_comm_domain_comparison_20261026/validation.json)。
