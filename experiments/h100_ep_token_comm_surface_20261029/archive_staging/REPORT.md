# H100 EP × token 通信二维表：绝对毫秒值

横轴是**整个 EP 组的原始输入 token 总数**，取 8、16、32、64、128、256、512、1024、2048、4096、8192、16384、32768；纵轴是 EP2、EP4、EP8。
每 token 路由到 8 个 Expert。每卡实际收到的 token–expert 行数 = `总 token × 8 / EP`；前一张 Expert 单卡图使用的是后一个口径，不能直接按相同横轴列相加。

**每格固定显示第 1 次原始 CUDA event 的绝对毫秒值，没有对 30 次采样、两轮或 GPU 求平均，也没有中位数或 relative 指标。** 一次交换要等各 rank 完成，因此每次取该次 Dispatch+Combine 用时最大的 rank；Dispatch 和 Combine 均来自这个相同 rank。原始单次值会波动，完整 30 次数据已保留。

## 测量范围

- 现有预约中的 H100 80GB：spartan-gpgpu011（33758541）、spartan-gpgpu171（33646189），共 8 张。EP2/EP4 在单节点 NVLink 内；EP8 使用两节点，节点内 NVLink、节点间 RDMA。
- 四个模型仅借用 hidden size / Expert 数 / top-k；全部交换 BF16 激活，不加载模型，不做 Expert GEMM、Attention、路由计算或 KV cache。122B 本次也是 BF16 通信，不能据此代表 FP8 通信。
- 同一已验证的 H100 SM90 DeepEP low-latency native；CE、PCIe payload、FP8、LogFMT 关闭。记录实际 native hash、CPU host-memory NIC handler 激活证据、peer 映射和拓扑。
- 每个节点的标准 Slurm 作业步使用该预约已有的 64 个 CPU，避免 4 个持续运行的 CPU NIC 代理与主线程共挤在 4 个 CPU 上；每个 rank 都记录并验证实际 CPU affinity。
- 全局路由和 token ID 在 EP2/EP4/EP8 保持一致。两轮分开展示，第二轮交换 EP2/EP4 所在节点、反转模型顺序。
- 每点初始化、分配、输入构造、正确性验证、Gloo barrier 与 Graph capture 不计时。Eager 先预热 10 次；Graph 先做 10 次 Eager 预热，捕获和验证后再预热 10 次 Graph replay；各正式记录 30 次。计时包含收发 hook、接收等待和 Eager 的 host feed 空隙。
- EP2 / EP4 / EP8 的源 token 容量固定为每卡 1024 / 512 / 256，所有大小沿用该容量；`num_qps_per_rank = Experts / EP`，NVSHMEM QP depth 固定 4096。
- **全组分块上限固定 2048 token，三个 EP 相同。8–2048 为一次 D+C，4096 / 8192 / 16384 / 32768 分别为 2 / 4 / 8 / 16 次。表中是完成全部这些交换的实测总时延，分项也是同一次测量中各块时长之和；没有把一次耗时乘调用数。**
- 这个分块协议是在 DeepEP LL 缓冲容量与 QP 深度约束下采用的受控通信协议；32768 列不是一个不分块的 DeepEP LL 调用，也不是完整模型 Prefill/Decode 延迟。它不包含计算重叠或 PP 调度。

## 主图：Graph，第一轮

![绝对 Dispatch+Combine 毫秒值](total_ms_graph_round1.png)

## 从绝对值可以看到什么

- 小消息不一定因 EP 增加而更快。例如 GLM、8 token，第一轮 EP2 / EP4 分别为 0.035328 / 0.044384 ms；2048 token 时则为 0.920448 / 0.334048 ms。第二轮同一大档分别为 0.918944 / 0.324064 ms。这里观察到固定开销与分摊负载的不同影响。
- 跨节点代价需要单独看：GLM、2048 token，EP8 两轮分别为 3.643520 / 3.785600 ms。EP4 与 EP8 同时改变了 GPU 数和通信域，不能将差异全部归因于 EP 数。
- 原始值中存在长等待：35B、2048 token、EP8，两轮事件 1 分别为 94.979683 / 2.094272 ms；GLM、32768 token、EP8 则分别为 134.626816 / 71.568222 ms。未删除或平滑这些样本，也尚未定位长等待原因。单个格子以及相邻 token 档的下降不能直接当作稳定的扩展规律；需要同时查看第二轮和完整 30 次原始事件。
- 单卡 Expert 计算图中大 token 时不同 Expert 数的差距是否收敛，需要按对应的 token–expert 行数查图。本通信图本身不能证明 EP/PP 的端到端排名，也不能把通信随 token 增长的代价当成已经被计算重叠隐藏。

## 简表：同一张图中的原始合计值

| 模型尺寸 | EP | 8 token (ms) | 128 token (ms) | 1024 token (ms) | 8192 token (ms) | 32768 token (ms) |
|:--|--:|--:|--:|--:|--:|--:|
| GLM-4.7 | 2 | 0.035328 | 0.084192 | 0.474368 | 3.669216 | 14.782784 |
| GLM-4.7 | 4 | 0.044384 | 0.063904 | 0.187840 | 1.301088 | 5.139904 |
| GLM-4.7 | 8 | 0.182368 | 0.378208 | 1.944352 | 17.549503 | 134.626816 |
| HY3 | 2 | 0.036256 | 0.074496 | 0.385280 | 2.971936 | 11.961824 |
| HY3 | 4 | 0.036448 | 0.066048 | 0.147776 | 1.064416 | 4.214528 |
| HY3 | 8 | 0.192192 | 0.355424 | 1.588352 | 14.800128 | 64.289726 |
| Qwen 35B | 2 | 0.035168 | 0.061696 | 0.205952 | 1.554400 | 6.219904 |
| Qwen 35B | 4 | 0.047232 | 0.065024 | 0.089824 | 0.572832 | 2.293728 |
| Qwen 35B | 8 | 0.285376 | 0.341024 | 1.086240 | 9.630912 | 45.941631 |
| Qwen 122B | 2 | 0.034912 | 0.065024 | 0.291616 | 2.240768 | 9.053600 |
| Qwen 122B | 4 | 0.043488 | 0.059936 | 0.129888 | 0.776128 | 3.079712 |
| Qwen 122B | 8 | 0.251392 | 0.369600 | 34.627457 | 12.616096 | 53.684063 |

## graph，第 1 轮

![Dispatch + Combine，绝对 ms](total_ms_graph_round1.png)

![Dispatch，绝对 ms](dispatch_ms_graph_round1.png)

![Combine，绝对 ms](combine_ms_graph_round1.png)


## eager，第 1 轮

![Dispatch + Combine，绝对 ms](total_ms_eager_round1.png)

![Dispatch，绝对 ms](dispatch_ms_eager_round1.png)

![Combine，绝对 ms](combine_ms_eager_round1.png)


## graph，第 2 轮

![Dispatch + Combine，绝对 ms](total_ms_graph_round2.png)

![Dispatch，绝对 ms](dispatch_ms_graph_round2.png)

![Combine，绝对 ms](combine_ms_graph_round2.png)


## eager，第 2 轮

![Dispatch + Combine，绝对 ms](total_ms_eager_round2.png)

![Dispatch，绝对 ms](dispatch_ms_eager_round2.png)

![Combine，绝对 ms](combine_ms_eager_round2.png)

## 如何对应 Expert 计算图

例如总数为 8192 token 时，EP2/EP4/EP8 每卡分别收到 32768 / 16384 / 8192 个 token–expert 行；驻留 Expert 数分别为全局 Expert 数除以 EP。需要使用对应的单卡 Expert 行数、驻留/激活 Expert 数、padding、同一分块和同一精度，再讨论省下的 Expert 时间能否抵消通信。本实验没有直接测得 EP/PP 的端到端盈亏平衡点；不能把两个组件在不同配置下的独立最慢值相加当成实测端到端时间。

## 原始数据与复现

- [交互表格](report.html)：可切换轮次、Eager/Graph、Dispatch/Combine/合计；所有格子都是固定事件 1 的原始值。
- [每格原始事件 1](absolute_event01.csv)，[全部 30 次、每次关键 rank](all_critical_events.csv)，[全部 rank 原始事件](all_rank_events.csv)。
- [配置](config.json)，[模型尺寸来源](geometry_sources.json)，[科学验证](comparison.json)，[独立原始事件审计](independent_audit.json)，[清理证据](cleanup.json)，[软件包元数据](environment_metadata.json)，[来源](sources.json)，[归档清单](manifest.json)。
- 原始运行目录：`/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_token_comm_surface_20261029/runs/r4`；冻结 harness 在其中 `frozen/harness/`。所有 native、wrapper、配置和 harness SHA256 见 frozen_inputs.json。
- CPU 重画：`/home/bxb1/conda/bin/python /data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_token_comm_surface_20261029/report_tools/build_report.py --run /data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_token_comm_surface_20261029/runs/r4 --out OUTPUT_DIR`。GPU 复跑使用新目录执行冻结 prepare.py，再按生成的 launch.json 通过 Guard 启动。
