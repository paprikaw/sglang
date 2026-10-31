# 35B：EP4 与四个 EP1 的容量、Prefill、Decode 对照

Qwen3.6-35B-A3B；同一个 H100 四卡节点 spartan-gpgpu178，四卡两两 NVLink。使用完整 BF16 权重、激活和 KV，GDN 状态 FP32。结果来自 2026-10-28 最终 r3，全部正式样本关闭 profiler。r3 末尾正常退出被包装脚本误判为失败；Guard 原始总状态保留 failed，缓存/清理后检通过，正式数据另经独立核验。

每个配置均为四卡总资源。EP4 的 Expert 跨四卡，Attention 为 TP1×DP4；四个 EP1 为四个独立、各持完整模型的单卡副本。PP 均为 1。两边每卡最多 16 个活动请求、84 个 GDN 状态槽，页长64、静态显存比例0.94、单卡 Prefill chunk4096、Decode Graph 开。引擎实际并发上限16已从启动日志验证。

B 是全局请求数，本轮固定64；P 是每请求计时前已驻留的 GPU context tokens；Q 是每请求本次新增输入。TTFT 从发送请求到首 token，TPOT 为首末输出 token 间时间除以127；吞吐按整个64请求 cohort 的完成跨度计算。高吞吐、低延迟较好。

## 主要结果

- EP4 的 KV token 池为 8,692,992，四个 EP1 合计 640,768，容量比 **13.57×**。这是本轮固定状态槽/Graph/显存设置下的原生分配池，不是最大 QPS 或最大并发扫描。
- 相同 B64、每请求缓存 8,256 tokens：EP4 新增512-token Prefill 吞吐为四个 EP1 的 0.752×，Decode 输出吞吐为 1.051×。
- EP4 通过驻留检查的最大测试点为每请求 130,304 tokens、总计 8,339,456 tokens，约占 KV token 池 95.93%。全部64请求在正式热缓存样本中通过命中检查。

## 同时放得下的对照

缓存 context 每请求 P=8,256，全局 B64。冷 Prefill 为 P=0、Q=8,256、生成1 token；热 Prefill Q512→1；Decode Q1→128。冷 Prefill 各1个 cohort，热 Prefill/Decode 各3个正式 cohort，另有未计时预热。

| 阶段 | 四个 EP1 总吞吐 tokens/s | EP4 总吞吐 tokens/s | 四个 EP1 平均 TTFT s | EP4 平均 TTFT s | 四个 EP1 TPOT ms | EP4 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|
| 冷 Prefill：全 context 新输入 | 182,192.144 | 141,037.371 | 1.662 | 2.149 | — | — |
| 热 Prefill：新增512 | 125,542.780 | 94,355.857 | 0.241 | 0.339 | — | — |
| Decode：输出128 | 5,327.057 | 5,599.577 | 0.143 | 0.215 | 10.790 | 9.780 |

## EP4 接近容量上限，交给两种布局相同输入

两边同样 B64、每请求 context 130,304 tokens。EP4 的热 Prefill/Decode 都复用真实 GPU 前缀。四个 EP1 的单卡 KV token 池只有 160,192，每卡最多完整驻留1条这样的请求；其相同负载包含排队与前缀重算。因此后两行是容量压力下的服务表现，不能当作纯热缓存 Expert/通信比值。

| 配置 / 阶段 | 吞吐 tokens/s | TTFT s | TPOT ms | 正式 cohort 数 |
|---|---:|---:|---:|---:|
| EP4 冷 Prefill | 94,025.296 | 47.334 | — | 1 |
| 四个 EP1 冷 Prefill | 110,745.874 | 39.785 | — | 1 |
| EP4 热 Prefill Q512 | 32,242.439 | 0.911 | — | 3 |
| EP4 热 Decode | 2,237.163 | 0.617 | 23.886 | 3 |
| 四个 EP1 同负载 Decode，含重算/排队 | 92.641 | 47.269 | 5.271 | 1 |

四个 EP1 的这64条 Decode 请求均未命中原前缀（cached_tokens=0）。其平均 TPOT 为5.27 ms，表示单卡仅有1条活动请求时连续输出较快；但平均首 token 等待47.27 s，导致整体吞吐仅92.64 tokens/s。

## Expert 收益和通信开销

下表为独立 profiler 中每卡每次真实 B16 Decode Graph replay 的平均 GPU kernel 时间，单位 ms。GEMM 只含 routed Expert 两次矩阵乘；DeepEP kernel 含通信等待。不同流可重叠，kernel 合计不能相加当作独占占比，也不能直接解释正式 TPOT。

| 负载 / 配置 | Graph GPU 跨度 | Routed Expert GEMM 合计 | DeepEP kernel 合计 | rank traces × replay 范围 |
|---|---:|---:|---:|---|
| four_ep1_common | 7.310 | 2.293 | 0.000 | 4 × 10–10 |
| ep4_common | 8.193 | 1.337 | 1.512 | 4 × 10–10 |
| ep4_maximum_f97 | 21.191 | 1.489 | 1.511 | 4 × 6–6 |

同负载下，EP4 每步 Expert GEMM 合计由 2.293 ms 降至 1.337 ms，减少 41.7%；另有约 1.512 ms 的 DeepEP kernel。Graph GPU 跨度由 7.310 ms 变为 8.193 ms，表明本点 Expert GEMM 的节省没有转化为更短的完整 GPU Graph。正式 TPOT 还包含 Graph 间调度、服务等开销，不能用上述 kernel 时间相减替代。
EP4 在长 context 下，完整 Attention kernel 合计从约 1.034 ms 增至 13.908 ms；长 context 的 Decode 成本明显由读取 KV 的 Attention 推高。

## 测量边界与证据

- 前缀由模型真实计算并保留 GPU KV/GDN；没有匹配的 Qwen hybrid CPU checkpoint，本轮没有伪造或恢复 CPU KV。正式热缓存测量前，必须连续两轮64请求全部命中；准备和 profiler 不进入正式指标。所有初始请求使用不同首 token，避免跨请求共享同一个前缀。正文重复构造，属于合成负载，未测生产 trace 或任务质量。
- 两边共享已有的小 batch Triton routed Expert GEMM 优化，局部输入≤16时启用；大 Prefill 仍为 DeepGEMM。EP4 使用原生 DeepEP normal Prefill / low-latency Decode，BF16 dispatch，原生 shared Expert overlap 开；TBO 关。H100 全 NVLink，DEEPEP_LL_NVLINK_PAIR_ONLY=0；本次未设置 CE 实验参数；所用 native12 二进制中检出的 DEEPEP 环境开关为 pair/PCIe threshold。原生路径和哈希见配置、冻结 manifest。
- 早期 r1 因状态槽不足被引擎将每卡并发从16降到10，r2 的接近容量点首轮预热只有63/64命中，均保留为失败 wrapper 轮次。最终表格只用完成11项测量、缓存与清理后检通过的 r3。容量候选与驻留过程留存，不能将97%的配置目标称为已证明的硬件绝对上限。
- 冷 Prefill 和四个 EP1 的容量压力负载只有1个 cohort；热缓存各3个，不能把请求级小样本 p99 当作稳定服务尾延迟。正式端到端结果与独立 profile 的时间边界不同。
- 引擎 `870269702285d94bdd926a50cc29ad1a20bcf60c`，固定源码标签 `experiment/sglang/glm47-ep16-canonical-profile-20261004-cpu-validated`；模型 revision `995ad96eacd98c81ed38be0c5b274b04031597b0`。
- [原始运行目录](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep4_vs_four_ep1_capacity_20261028/runs/h100178_r3) · [配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep4_vs_four_ep1_capacity_20261028/runs/h100178_r3/config.json) · [冻结输入](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep4_vs_four_ep1_capacity_20261028/runs/h100178_r3/dashboard_spec.json) · [逐轮 CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep4_vs_four_ep1_capacity_20261028/full_metrics.csv) · [协议核验](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep4_vs_four_ep1_capacity_20261028/runs/h100178_r3/analysis/protocol_validation.json) · [GPU Graph 分析](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep4_vs_four_ep1_capacity_20261028/runs/h100178_r3/analysis/decode_graph.json)。
- [实际拓扑](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep4_vs_four_ep1_capacity_20261028/runs/h100178_r3/topology.txt) · [容量候选](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep4_vs_four_ep1_capacity_20261028/runs/h100178_r3/ep4_capacity_candidates.json) · [进程与 GPU 清理](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep4_vs_four_ep1_capacity_20261028/runs/h100178_r3/task/cleanup.json)。Guard 的模型缓存与清理检查通过；总状态因退出处理错误为 failed。三组自有模型进程均已退出，租约已释放；没有新建 Slurm 预订。

## 收尾状态说明

r3 的 worker 已完成全部11项测量并成功清理三组服务，但主函数将自己的 sys.exit(0) 放在 except BaseException 内，捕获了正常 SystemExit:0，最终返回1。Guard 因此保留 scientific_success=false、passed=false，不能把它标成成功轮次。后检独立确认模型缓存9份启动日志/12个rank证明与清理全部通过；本报告重新核验了冻结哈希、全部请求时间与输出数、前缀命中和实际Graph批量。
用于以后复跑的修正版在 [reusable_code](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep4_vs_four_ep1_capacity_20261028/reusable_code)，原始冻结运行没有改动。[补丁](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep4_vs_four_ep1_capacity_20261028/entrypoint_fix.patch) 与 [实际主函数退出路径回归验证](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep4_vs_four_ep1_capacity_20261028/runs/h100178_r3/analysis/entrypoint_fix_validation.json) 保留。没有为修复退出标记重新占用GPU。
