# A100 两卡／八卡：通信域与 EP、PP 的部署取舍

日期：2026-10-26。**只读取已经保存的历史数据、重算及作图，没有申请 GPU 或运行新性能实验。**

## 1. 术语与准确负载

| 项目 | 本报告的确切含义 |
|---|---|
| EP2、EP8 | 同层专家分布到两张／八张卡；不是两个／八个独立副本。两卡 Attention 为 TP1×DP2；八卡为节点内 TP4×DP2。 |
| PP2、PP2×EP4 | 两个模型层阶段。两卡 PP 每阶段本地 EP1；八卡 PP 每节点一个阶段、节点内 EP4、Attention TP4。 |
| 两卡 Prefill | 8 请求；每条没有已缓存上下文，新增输入 4096 token，生成 1 token；整批 32768 新输入。 |
| 两卡 Decode | 8 请求；每条 GPU 已有真实模型生成的 4096-token 缓存，新增输入 1 token，生成 128 token；不计前缀构建、没有 CPU HiCache 换入。 |
| 八卡 Prefill | 4 请求；每条无旧缓存、新输入 8192 token、生成 1 token；整批同为 32768 新输入，但请求分布不同。 |
| 八卡 Decode | 4 请求；每条从 128 个新输入 token 开始，生成 512 token；冷输入，不是已有长缓存的纯 Decode 起点。 |
| TTFT／TPOT／端到端 | 请求放行至首输出／首输出至末输出除以输出数量减一／放行至末输出。Prefill 仅输出一 token，TPOT 不适用。 |
| 吞吐 | Prefill 计输入 token/s，Decode 计输出 token/s，越大越好；不能由平均请求延迟的倒数推得整批吞吐。 |
| 纯通信 token | 单次发给专家的唯一输入 token，不是旧缓存长度，也不是一定等于请求数。D+C 为一次完整 Dispatch＋Combine 事件跨度，不是请求 TPOT。 |

## 2. 完整推理：有类似实验，但 A100 不是 H100 那种反转

两卡主组是 Qwen3.5-122B-A10B FP8，A100069 GPU0/1、NV12；W8A16 Marlin、BF16 激活/KV、Triton Attention；DeepEP normal NVLink；Graph/overlap 关闭。固定两槽 PP，同一对卡顺序运行，各三轮。
八卡为 GLM-4.7-AWQ INT4/Humming，168/169 各四卡；同旧 native 9dc34887，pair 内 NVLink、pair 外及跨节点 RDMA。Prefill eager、Decode Graph；每点一个正式四请求 cohort。
八卡已统一节点内 Prefill chunk1024，EP 全局2048／PP全局1024；不能称为相同全局 step 或相同 GEMM 形状。

| 部署／模型 | 阶段 | 配置 | 吞吐 token/s ↑ | 平均 TTFT 秒 | 平均 TPOT 毫秒 | 平均端到端 秒 |
| --- | --- | --- | --- | --- | --- | --- |
| 两卡 Qwen122B | prefill | EP2 | 8670.75 | 2.3224 | 不适用 | 2.3224 |
| 两卡 Qwen122B | decode | EP2 | 72.08 | 0.2780 | 109.6631 | 14.2052 |
| 两卡 Qwen122B | prefill | PP2 | 8185.11 | 2.3892 | 不适用 | 2.3892 |
| 两卡 Qwen122B | decode | PP2 | 97.26 | 0.1739 | 81.3724 | 10.5082 |
| 八卡 GLM INT4 | prefill | PP2xEP4 | 1943.53 | 10.7101 | 不适用 | 10.7101 |
| 八卡 GLM INT4 | decode | PP2xEP4 | 101.92 | 0.4131 | 38.2922 | 19.9805 |
| 八卡 GLM INT4 | prefill | EP8 | 1322.77 | 18.2535 | 不适用 | 18.2536 |
| 八卡 GLM INT4 | decode | EP8 | 100.06 | 0.4378 | 39.0455 | 20.3901 |

![完整推理归一化比较](e2e_communication_domain.png)

- 两卡 122B：两槽 PP Decode 吞吐比 EP 高 **34.92%**；Prefill 则低 **5.60%**。
- 两卡 35B BF16 的补充对照：PP Prefill **31841.16** 对 EP **27644.14**（+15.18%）；PP Decode **121.76** 对 EP **90.80**（+34.09%）。与 122B 是不同精度和内核，不是模型大小单变量消融。
- 八卡 GLM：PP Prefill **1943.53** 对 EP **1322.77**（+46.93%）；Decode **101.92** 对 **100.06**（+1.85%）。四请求、单 cohort 的小 Decode 差距不足以证明稳定领先。
- 因而，**现有 A100 完整推理不能证明“跨节点后 PP 相对收益必然变大”，也没有 H100 那种 EP2 Decode 领先、跨节点 EP8 落后的明确翻转。**

### 为什么不能换一个旧 PP 数字来制造反转？

同一 Qwen122B 旧四槽 PP Decode 只有49.04 output token/s，低于 EP 的72.08；但两槽 PP 是97.26。槽位调度足以改变结论，所以本主图固定两槽而不挑选四槽。
后续 Q02 发送就绪优化的 Prefill 同进程从8160.97提升到9391.74 input token/s；那轮没有重测 EP，Decode 输出等价未通过，只保留 Prefill 局部优化结论，未混入本图。
Qwen122B 原 EP r7 测量、缓存和清理完成，但原 Guard 包装器误捕获 SystemExit(0)，状态保持失败；指标已独立复算。后续两槽 PP r8 通过。35B 的配对完整验证通过。上述操作状态与性能证据分别保留，不补写成功记录。

## 3. 同库纯通信：节点内到底走了什么，比标签更关键

最新 T01 扫描：065/167 同型号 ConnectX-6 Dx，GLM 通信向量宽5120、BF16、160专家、每 token 命中8专家；确定性均衡路由，没有模型、Attention、Expert GEMM 或 KV。
每点三次预热、三次正式采样；先取同次迭代 rank 最大跨度，再取中位数。GLM RDMA／CE 共享 c3405cad 原生库；实际同调用源行数、路由、容量、Graph 模式均匹配。

这里用**两种 EP 宽度单次调用输入 token 相同**的对照。EP4 每卡的 token–专家配对数为 EP8 的两倍，并非相同每卡工作量；完整 PP 半批对照保留在原 T01，不假造 PP 吞吐。

单位毫秒，越小越好。下表节点内为065，跨节点为065/167；另一节点独立结果完整保留在 CSV 和右图。

| 单次输入 token | 节点内 EP4 RDMA | 跨节点 EP8 RDMA | 节点内 EP4 CE | 跨节点 EP8 CE |
| --- | --- | --- | --- | --- |
| 8 | 0.1852 | 0.2846 | 0.2396 | 0.6021 |
| 128 | 1.1025 | 0.8796 | 0.4235 | 1.1069 |
| 1024 | 8.4070 | 6.3670 | 1.3348 | 7.0528 |
| 4096 | 35.4867 | 26.6117 | 4.6483 | 31.0398 |

![同库通信策略与输入量](communication_transport_curves.png)

- 8-token 小消息，RDMA 策略节点内0.1852／跨节点0.2846 ms，跨节点约1.54倍；该小点 CE 更慢。
- 1024-token、CE 策略节点内1.3348／跨节点7.0528 ms，跨节点约5.28倍；但 RDMA 策略为节点内8.4070／跨节点6.3670 ms，方向相反。
- CE 只替换同节点 NVLink pair 之外的 PCIe 搬运，跨节点仍走 RDMA；不能把 EP8 CE 称为跨节点 PCIe。EP4 的全本地 drain 优化与 EP8 远端完成路径也不同。
- 节点内、两节点的资源聚合、source 分配、QP／专家所有权和同步协议不同；不能把整段时差只叫作物理线缆延迟，事件跨度还包括 CPU 提交、打包、等待和还原。
- 旧9dc库同全局1024-token基线也出现过节点内8.7050～8.8947／跨节点EP8 6.5529 ms；该旧基线同节点 pair 外同样走RDMA。不能把它描述为全NVLink对跨节点。

## 4. 这份证据支持到哪里

**可以支持：EP／PP 的选择依赖实际通信域、传输实现、消息量和本地 Expert 执行成本。** H100 的节点内全NVLink与A100的“两组NVLink pair＋PCIe”不是同一种本地拓扑。CE让A100大消息节点内交换明显受益，但小消息、跨节点与完整调度都需另看。

**不支持：** 跨节点一定让PP获益更多；节点内一定比跨节点快；用不同模型的端到端差值唯一归因于网络；把组件中位数直接相加预测TPOT。完整推理之间还改变了模型、精度、内核、Graph、请求数、缓存起点和输出长度。

## 5. 文件与可复查入口

- [全部12行请求指标](full_metrics.csv)：包括补充35B，均值／中位数／P99；两卡每点24请求、八卡每点4请求，P99只描述样本。
- [6组吞吐对照](e2e_comparisons.csv) · [GLM两节点20行同调用通信表](communication_equal_call.csv) · [旧RDMA基线](old_rdma_communication.csv)。
- [原始路径、逐文件SHA256](source_manifest.json) · [独立离线校验](validation.json)。源表及短报告小副本在sources/，没有权重、KV或原始大trace。
- [T01全部模型／两种批次边界](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/single_layer_ep4_ep8_matched_20261024/ce_rdma_r1/summary.md)；其通信与独立真实Expert联表仍是组件估算，不是完整PP观测。
- [H100对应报告](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_comm_domain_comparison_20261026/REPORT.md)。

复算与画图：`MPLCONFIGDIR=/tmp/a100-comm-domain-mpl PYTHONDONTWRITEBYTECODE=1 /home/bxb1/conda/bin/python /data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/a100_ep_pp_comm_domain_comparison_20261026/build_report.py`。
