# 35B 小 batch：同预算四张 H100 整模型验证

完成 6/6 个点。每点 5 轮，以下全部为算术平均值。

固定同一 H100 节点、完整 BF16 35B、GPU 热缓存 P4096、新输入 Q1、输出 128 token；N 是全组同时请求数。不启用 HiCache，不把 CPU 权重启动时间或冷 Prefill 放进 Decode 计时。

2×PP2 保留已验证的 policy9 lookahead，PP4 是原生调度；所有布局使用相同的小 token Expert GEMM 家族。PP2 的请求分组以每副本的本地序号确定，等待合批计入请求延迟。

| 全组 N | 布局 | 平均 TPOT ms/token | 平均请求 E2E ms | 平均 TTFT ms | 整批完成 ms | 输出 tok/s |
|---:|---|---:|---:|---:|---:|---:|
| 8 | ep4 | 6.106 | 962.21 | 186.70 | 963.05 | 1063.3 |
| 8 | two_pp2 | 5.484 | 789.67 | 93.21 | 795.02 | 1288.0 |
| 16 | ep4 | 6.757 | 1040.92 | 182.80 | 1042.58 | 1964.4 |
| 16 | two_pp2 | 6.711 | 944.59 | 92.23 | 952.04 | 2151.4 |
| 32 | ep4 | 7.785 | 1172.41 | 183.69 | 1175.58 | 3484.3 |
| 32 | two_pp2 | 8.629 | 1192.84 | 96.91 | 1204.03 | 3401.9 |

TPOT 先逐请求计算 `(最后 token 到达 − 首 token 到达) / 127`，再对请求和 5 轮取平均；平均请求 E2E 包含 TTFT，整批完成时间是不同的指标。各指标分别判断反转，不能混用。

四卡布局按顺序测量，每点重复 5 轮；尚未做布局交替复测。自然生成的路由允许随数值与 batch 变化，因此这些结果验证部署效果，不能单独归因于通信或专家数量。

实际 rank／stage batch 与 Graph 资格计数保存在 [forward_histograms.csv](forward_histograms.csv)。这些是每步参与计算的请求数；它们与微基准的 token–expert assignment 行数不同。

原始样本：[formal_rounds.csv](formal_rounds.csv)；汇总：[mean_metrics.csv](mean_metrics.csv)。
原始运行目录：/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_small_batch_layout_verify_20261029/runs/h100011_r9
