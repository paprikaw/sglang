### 补图：通信和 Expert 的绝对耗时与占比

[PP合并后的主图与对照表](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_component_shares_20261026/GROUP_COMPARISON.md) · [原逐阶段明细](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_component_shares_20261026/REPORT.md)。

主图已将PP合成一条，EP也按整组累计，避免拿PP两卡之和去比EP每卡均值。两卡Qwen：Prefill按整批8请求的全部GPU内核累计；Decode从每个4请求Graph换算到8请求，正确计入PP两槽、每槽经过两个阶段。百分比**不是端到端独占占比**。Decode的Attention／Expert只识别核心／GEMM，其余保留在其他，PP Graph外边界传输没有被当成零。八卡GLM只找到关联的4请求Prefill层20／50的独立rank分区：EP共16份、PP共8份，分别全部合并。本地新token为EP512／PP1024，含等待，不是完整模型或同步全局轮次。八卡4／16请求Decode和16请求Prefill的匹配模型分项仍缺失，未用纯通信微基准填补。

**八卡路由Expert统计说明：** 每卡每层的平均本地跨度是EP8约0.256 ms、PP约0.494 ms（PP约1.93倍）。图中累计4.093／3.949 ms来自16／8份记录，不能据此说PP Expert更快；本地token、实际路由负载和采样轮次不匹配，约3.52%的合计差也不是受控速度差。

归档保留 6 行两卡分项、24 个八卡原始 rank-layer 样本及其 4 行分层均值、原图 12 行正式请求指标、原始路径和哈希。没有新 GPU 实验，也没有改写历史结果或失败状态。
