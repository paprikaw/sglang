# EP8 与节点内 EP4／PP2×EP4：通信优势历史证据

只读取已完成实验；没有申请资源或运行新 GPU 测试。各组保留独立 workload、native、拓扑和计时协议，不跨组相加。

## 1. 当前 GLM MTP Decode 的目标 Verify 通信

065／167，8×A10080GB、同型号 ConnectX-6 Dx；真实语义 B8／P4096，实际 CPU 前缀4032，Q1→64；MTP top1/steps2/Verify Q3，local Graph B4。EP8 为 Attention TP4×DP2；PP 为两个 B4 slot、Attention TP4／Expert EP4，50／42目标层。

单位 ms／每次目标 Verify 的每卡 Graph。EP8取8卡均值；PP先取每 stage 4卡均值，再沿完整目标模型路径合计。这个合计用于核对相同逻辑层路径的组件工作量，不是一个全局同步通信窗口，不是 TPOT，也不包含 PP stage 边界激活和反馈。

| 目标模型内部通信 | EP8 | PP stage0 | PP stage1 | PP路径合计 | 少多少 |
|---|---:|---:|---:|---:|---:|
| DeepEP Dispatch | 13.3792 | 4.7844 | 4.4189 | 9.2033 | 4.1758（31.2%） |
| DeepEP Combine | 13.3351 | 4.5810 | 4.1455 | 8.7265 | 4.6086（34.6%） |
| DeepEP D+C | 26.7143 | 9.3655 | 8.5644 | 17.9298 | 8.7845（32.9%） |
| NCCL TP | 7.0431 | 3.4891 | 2.9666 | 6.4557 | 0.5875（8.3%） |
| 内部通信合计 | 33.7574 | 12.8546 | 11.5309 | 24.3855 | 9.3719（27.8%） |

原生 DeepEP SHA9dc34887…；NVLink pair内走NVLink，pair外及跨节点走RDMA，PCIe threshold0、无CE。Graph分段有诊断同步，不能用这些数冒充未插桩 TPOT。EP8为此前同Bank参照，并非与50/42 PP重新配对运行；跨布局严格输出等价仍未通过。

## 2. 真实 GLM Prefill：同本地 P／Q 的单个 MoE 层

168／169，8×A100，C4／8192→1；实际本地 chunk1024，DeepEP source256/cap256，BF16 H5120、E160/top8。详细采样 P=1024／5120／7168、Q1024，按真实 rank-layer 调用数加权。native同为9dc34887…；pair外为RDMA，没有CE。

| 每 MoE 层 ms | EP8 | PP内EP4 | 差值 |
|---|---:|---:|---:|
| Dispatch | 6.5218 | 4.0397 | 2.4820 |
| Combine | 6.7401 | 3.6241 | 3.1160 |
| 整层区间 | 16.8171 | 11.3695 | 5.4476 |
| D+C | 13.2619 | 7.6639 | 5.5981（42.2%） |

这些是带完整性校验的 CUDA-event 互斥流区间，包含暴露等待，不是纯wire延迟。独立GPU kernel关联得到PP／EP8整轮规范化D+C7.62895／13.21203ms。增量主要出现在接收完成阶段；不能继续把它全归给NIC或CPU proxy。实际两组expert assignments相同，节点内chunk已经对齐。

## 3. 纯通信：固定全局 token 和路由，节点内 EP4 对跨节点 EP8

168／169；BF16 H5120/E160/top8，native c3405cad…；NVLink pair + 本地CE PCIe + 远端RDMA。无Attention、Expert GEMM、模型或KV；eager，无CUDA Graph。指标是每次对齐rank-max D+C之后的三样本中位数。EP4的source/cap256与40QP，EP8的source/cap128与20QP并不相同，不能把差额只归因于一根链路。

| 全局 tokens | 节点内 EP4 ms | 跨节点 EP8 ms | EP4少多少 | EP8／EP4 |
|---:|---:|---:|---:|---:|
| 8 | 0.3190 | 0.4886 | 34.7% | 1.53× |
| 1024 | 1.4453 | 4.9979 | 71.1% | 3.46× |

D+C独立测量，不能把Dispatch与Combine各自的rank-max中位数再相加。初始化和合成expert输出准备排除，普通打包、动态计数、CE提交、完成保护和weighted reduction计时。

## 4. 纯通信：同一节点的 RDMA／SM PCIe／CE PCIe 实现

node168、4×A100，H5120/E160/top8，cap256、40QP，3次warmup与3次正式样本，eager。以下全部来自同一个r7对照，不把前轮最佳值拼在一起。

| source tokens／卡 | 全局 tokens | RDMA ms | coalesced SM PCIe ms | CE PCIe ms | RDMA／CE |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0.1764 | 0.1403 | 0.2347 | 0.75× |
| 16 | 64 | 0.6529 | 1.0532 | 0.3412 | 1.91× |
| 64 | 256 | 2.1965 | 4.0755 | 0.5407 | 4.06× |
| 256 | 1024 | 8.5668 | 16.7004 | 1.3322 | 6.43× |

**CE大幅加速确实测到过，但不是当前MTP Decode Graph已经使用的收益。** source1下CE反而慢；动态CPU计数→CE提交的已验证接口不能直接捕获进现有Decode Graph。本组较大消息收益不能直接叠加到第一组的TPOT。

## 如何读这些证据

- 当前MTP模型内D+C确实少约8.78ms／完整目标Verify路径，约32.9%；包含TP后，内部通信约少9.37ms。
- Prefill同本地Q的真实单层D+C少约42.2%，并得到独立GPU kernel关联支持。
- 固定全局数据量的CE纯通信，EP4比EP8在两个测试点都快，但它是eager测试，不是当前Graph路径。
- 以上不能直接推导PP的每输出token一定快同样毫秒：还要计PP边界／反馈／调度依赖，以及一次MTP验证推进的token数量。当前数字支持优化动机，不是通信已在端到端关键路径上全部兑现的证明。

## 原始来源及SHA256

- `/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_mtp_verify_fixed_ep8_pp2ep4_20261023/runs/paired_r1/ep8_mtp_fixed/stage_activity_summary.json` — `8566ed2b244db8270f084b264f8e49001af65c738999a7adbd44bfa2f99dc5e7`
- `/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_pp_stage_balance_20261023/runs/balance_r1/pp2ep4_balanced/stage_activity_summary.json` — `1a1384e8f70947abd67c98a7a6cbb1d857ea629156c6fd08d23a5e14f90cb473`
- `/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_a100_ep8_pp_attribution_analysis_20261010/root_layer_comparison.csv` — `03eb3b703d372e9eb240cb77e3c7c5aa7e4f93bc1c93e8010d7573835c000312`
- `/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/deepep_ce_ep8_admission_20261010_r1/summary.csv` — `a07736c06e27c71b70fc5d3160b9a7219c4bf8205b3fa86629d04104365c92d3`
- `/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/deepep_prefill_pcie_optimization_20261010/full_metrics.csv` — `b9ab9c757a5b86aa8edce77246201387f725006adbb62ee6323948511bf40ff8`
