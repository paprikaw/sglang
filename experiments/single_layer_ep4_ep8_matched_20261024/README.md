# 单层 EP4 / EP8 隔离对照

## 本轮 CE／RDMA 对照

**状态：完成，280 个通信点，独立复算及退出清理通过。** [完整表格与曲线报告](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/single_layer_ep4_ep8_matched_20261024/ce_rdma_r1/summary.md)。新增运行目录为 [ce_rdma_r1](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/single_layer_ep4_ep8_matched_20261024/ce_rdma_r1)，入口代码在 `transport_compare_v1`。状态和总结同步到 [总实验索引第 8 节](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/README.md)。

只补通信曲线：GLM／Hy3 两种真实专家层的通信形状；CE／RDMA 两条路径；整批唯一 token 从 8 倍增到 4096。包含“EP8 整批、EP4 半批”与“单次 token 相同”两种边界。两种模型均不加载权重，不执行 Attention、Expert GEMM 或 KV Cache。新运行和下述旧基线独立保存。

## 历史 RDMA 基线：comm_r2

只加载单层 Expert；通信部分完全不加载模型。EP8 处理整批唯一 token；EP4 分别处理前半批和后半批，每卡源 token 数相同，每卡实际 token–expert 配对数相同。

通信固定：同一 c340 native，NVLink pair 内 NVLink，pair 间及跨节点 RDMA，eager 模式，每档 3 次。不是 Copy Engine 或 CUDA Graph 测试；不把混合协议相加。

不会声称单层结果等于完整 PP 的 TPOT：没有测 Attention、stage 边界及调度。
