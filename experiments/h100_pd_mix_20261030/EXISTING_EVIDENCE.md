# 复用已有结果与本轮增量

本轮不重复历史 Decode 网格、Expert 二维图、通信二维图或 CPU KV 缓存重放。

| 已有证据 | 能回答的问题 | 不能直接推论的内容 |
|---|---|---|
| [最新固定批次 Decode 验证](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/REPORT.md) | 35B、四卡、GPU 热缓存的16类 workload 中，优化 PP2 未取得最低平均 TPOT；小批单卡副本、大批 EP4 仍强 | 新 Prefill 到达时的干扰、排队与实例分工 |
| [已有四卡五布局、三轮平均值](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/four_gpu_layout_retrieval_20261029/REPORT.md) | GPU热缓存短前缀 Prefill 中单卡副本有竞争力；受控 CPU 恢复的 Prefill 中 PP2 有优势 | 不能将缓存恢复收益直接当成冷 Prefill 计算收益，也不能将该 PP 实现标成固定批次 policy9 |
| [组件专题与预测边界](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/ep_pp_tradeoff/REPORT.md) | Expert 工作集、通信拓扑、层级映射提供候选方向 | 组件预算没有包含完整合批、流水等待与动态队列，不能直接预测在线最优部署 |

本轮新增的是同一到达轨迹上的完整 P+D 请求，以及 EP2＋PP2 常驻混合池。每条请求本身都包含 Prefill 与 Decode；“P类”指长输入短输出，“D类”指短输入长输出，不是已经做了 P/D 分离。

新测三类较小冷 Prefill 校准点（全组16K或128K输入 token）用于分清计算与缓存恢复。历史64请求、Q8256/130304的冷准备记录不在这一批次规模，且各自仅一轮，故不拿来代替本轮三轮校准，也不重新测那10条历史冷准备。

同一节点内比较同预算候选。两节点反向顺序用于独立复核，均值不跨节点拼接。资格失败、到达偏差和原生容量限制须保留；不会以最慢轮、最快轮或某一缓存协议替换全部正式样本。
