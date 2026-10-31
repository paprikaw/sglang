# 优化记录：通信库、Attention、Expert、流水调度与缓存加载

更新：2026-10-27。[返回实验总览](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/README.md)。

这份文档汇总本项目截至目前可追溯的实现、性能优化及未成功尝试。它不是“所有优化同时开启”的配置单：不同模型、通信库、CUDA Graph 状态和缓存起点必须分别读取。原始实验、失败状态和逐请求结果保持不变；本次仅整理已有证据，没有启动新 GPU 实验。

## 1. 先说明术语和计时边界

| 名称 | 本文准确含义 |
|---|---|
| Prefill | 处理本轮新增的输入 token。每项另外说明是否已有缓存前缀，不能把已缓存长度当成又做一次 Prefill。 |
| Decode | 逐步生成输出。已有 GPU 缓存、从 CPU HiCache 恢复、以及首轮补算尾部是不同起点。 |
| Dispatch／Combine | 将 token 发给目标专家／将专家输出返回并按路由权重合并；不是 TP 的 All-Reduce。 |
| TP／DP／EP／PP | 张量并行／独立 Attention 请求组或模型副本（按具体实验说明）／专家并行／按模型层切分的流水并行。PP 阶段数与循环槽位数不是同一概念。 |
| CE、SM、RDMA | CE 是 GPU Copy Engine 搬运；SM 路径使用 GPU 计算单元执行搬运；RDMA 指网卡参与的内存访问。本项目“RDMA 对照”一般仍保留 NVLink pair 内的 NVLink。 |
| TTFT／TPOT／端到端 | 请求放行至首输出／首末输出间隔除以输出数减一／请求放行至末输出。输出只有 1 token 时 TPOT 不适用。 |
| 单层／完整通信／请求级 | 单层表为每卡每次层执行毫秒；完整通信表包括 Dispatch、Combine 及其必需等待；请求级另含排队、缓存换入和流水调度。不能混为同一单位或直接相加。 |
| 验证通过 | 必须注明范围。通信 payload 正确、同一 PP 基线输出不变、无损 MTP、Guard 操作通过是不同判断。 |

## 2. DeepEP：先让 A100 路径正确工作

| 改动 | 原来的问题及实现 | 已验证范围／限制 |
|---|---|---|
| A100 跨节点 CPU postsend 协议修复 | PR517 路径与 NVSHMEM 3.4.5 的代理协议不一致；修正 ready-head 顺序、发送队列生产者索引与 doorbell 通知。保留进程内 AH `static_rate=0` workaround；使用 CPU NIC handler 绕过当时 GPU NIC 映射失败。 | 原生跨节点通信与 TP4 reduction 等 384 个 rank-case 检查通过；不是驱动升级，也不是已证明原生库支持任意 SGLang 混合 EP/TP 布局。修复源提交 `af5177ce`。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/deepep_ep2_tp4_debug_20261001/summary.md) |
| GLM AWQ 与低延迟布局衔接 | 改用 Humming 的 INT4 masked expert 路径接收 DeepEP 专家布局；后续完整加载器修复 AWQ qzeros／group-scale 元数据。不重新量化原 checkpoint。 | 真实单层专家、DeepEP Dispatch→GEMM→Combine 和相应 Graph 检查通过；这首先是兼容性修复，不能单独当成同计算后端的性能消融。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm_awq_humming_ep8_a100_20261001/summary.md) |
| 启用节点内 P2P | 映射同节点 GPU 的 NVSHMEM peer 指针，NVLink pair 内走 NVLink、pair 外可走 PCIe，远端保留 RDMA。PCIe peer 缺少原生原子操作不等于不能读写 payload。 | 1200 个 rank-case 检查通过，含空源／不均匀输入及禁 RDMA 回退的诊断。跨节点不能使用“禁止全部远端回退”的诊断 binary。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/deepep_intranode_p2p_20261002/summary.md) |

**版本必须同时记录 Python wrapper、实际加载的 `.so` 哈希和路由选项。** 历史 wrapper 标成 `05d4615` 的整模型结果，实际 native 为 `b0c15ccb…`，不能据此称为 CE 或真正合并访存实现。当前各版本的精确路径、完整哈希、开关及接受范围见 [证据与原始来源](/home/bxb1/.codex/skills/llm-experiment-guard/references/deepep_backends.md)。

## 3. PCIe 通信：做了哪些优化，什么时候有效

### 3.1 SM 合并访存及双卡对照

保留原有低 SM launch 和向量化设计，优化 PCIe payload 访问；同时修正 RDMA 对照初始化：不能只设置环境变量，构造 Buffer 后也要核验实际映射。下面固定同一节点 GPU2↔GPU0、跨 NVLink pair 的 SYS 路径，只测通信。BF16、hidden size 5120、top-8，CUDA Graph 关闭；预热 5 次，正式 32 次，先取每次较慢 rank，再算中位数。

| 每张源 GPU 的输入 token 数 | 原 RDMA，毫秒 | 同候选 binary 的 RDMA，毫秒 | 优化 PCIe，毫秒 |
|---:|---:|---:|---:|
| 1 | 0.1075 | 0.1055 | 0.0788 |
| 16 | 0.3328 | 0.3287 | 0.1925 |
| 256 | 4.1779 | 4.3331 | 1.9640 |

这些是同节点网卡路径与 PCIe 的受控比较，**没有证明比所有跨节点 RDMA 都快**。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/deepep_pcie_vs_rdma_20261006_r2/summary.md)

### 3.2 Copy Engine、去重、匹配顺序与通知协议

SM 搬运并不总能有效利用 SYS 路径。CE 方案先动态取得实际发送计数，再打包并用 `cudaMemcpyAsync` 搬运，目标端还原原始专家布局；不把动态计数、打包、等待或 ACK 移到计时外，也不靠容量 padding 替代真实数据量。

| 迭代 | 具体改动 | 该轮每源 GPU 256 token 的完整通信，毫秒 |
|---|---|---:|
| r3 | 动态 CE 代替跨 pair 的 SM payload 搬运 | 16.7899 → 2.5180 |
| r3 | 同一 token 发往同一目的 GPU 的多个专家时只跨 PCIe 传一次，在目标恢复各专家槽位 | 2.5180 → 2.0470 |
| r4 | 发送顺序匹配，避免两个源同时优先冲向同一接收端 | 2.0389 → 1.4787 |
| r5 | 仅在所有 peer 都本地可映射时，去除不需要的远端 drain | 1.5077 → 1.3871 |
| r7 | 专家完成通知随本地 frame 携带；先消费元数据，再 ACK 允许复用 | 1.3645 → 1.3322 |

各行是**各轮自己的对照**，不能串乘成额外加速。最终 r7 同节点、同路由、同一完整 benchmark 的结果如下：

| 每张源 GPU 的输入 token 数 | 全局唯一 token 数 | RDMA，毫秒 | 合并访存 SM PCIe，毫秒 | CE PCIe，毫秒 |
|---:|---:|---:|---:|---:|
| 1 | 4 | 0.1764 | **0.1403** | 0.2347 |
| 16 | 64 | 0.6529 | 1.0532 | **0.3412** |
| 64 | 256 | 2.1965 | 4.0755 | **0.5407** |
| 256 | 1024 | 8.5668 | 16.7004 | **1.3322** |

设备为 node168 四张 A100，160 专家、均衡 top-8、每 rank 容量 256；CUDA Graph 关闭，每档预热 3 次、正式 3 次。没有 Attention 或 Expert GEMM。source256 的 CE 比 RDMA 快约 **6.43 倍**；source1 则更慢。搬运校准最高 18.50 GB/s，四卡匹配顺序每 sender 17.31 GB/s；这是接近本机该路径实测裸拷贝水平，不是达到 PCIe 端口 31.51 GB/s 理论上限。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/deepep_prefill_pcie_optimization_20261010/summary.md)

### 3.3 扩展到 EP8 与真实模型

| 验证 | 负载、边界 | 结果及限制 |
|---|---|---|
| 早期动态 CE EP8 | 065／169、8×A100；全局 1024 token、24 个活跃路由地址槽；无模型、CUDA Graph 关；同 Buffer 交错 3 次 | 完整通信均值 CE **5.342** 对 RDMA **6.548 ms**，低 18.42%。包括计数回读、pack/unpack、RDMA drain 和 ACK；不是只测 memcpy。只覆盖这组形状。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/deepep_ep8_dynamic_copy_engine_20261007_r2/summary.md) |
| 后续 EP4／EP8 形状扫描 | GLM／Hy3 通信形状；全局唯一 token 8→4096 逐次翻倍，同时保留“同整批”和“同单次输入”两种比较 | CE 在节点内大消息受益明显，EP8 跨节点收益不稳定，不能把节点内结果直接推广到远端。整批 4096、EP4 每次 2048 token 时，065 上 GLM 形状 RDMA／CE 为 **17.7161／2.4376 ms**。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/single_layer_ep4_ep8_matched_20261024/ce_rdma_r1/summary.md) |
| GLM 完整 Prefill | 168／169、8×A100；4 请求，每条无旧缓存、新输入 8192、输出 1；每节点 local chunk 1024，CE native `c3405cad…` | PP2×EP4 为 **4092.66 input tok/s、TTFT 5.079 s**；EP8 为 **1635.45、14.731 s**。两个 CE 布局各一正式批；不是本轮同 native 的 CE/RDMA 消融。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_ep8_pp2ep4_ce_prefill_20261010_r4/summary.md) |

实现约束：本地仍保留 NVLink pair；只有跨 pair 的本地 payload 用 PCIe CE，远端继续 RDMA。EP8 必须保留远端完成顺序，不能照搬 EP4 的全本地 skip-drain。r7／EP8 扩展分别为 native `b6cdfc99…`／`c3405cad…`。动态 CPU count 和 CE 提交路径未获 Decode CUDA Graph 资格，额外 staging 在所测几何约 803 MiB／rank，须计入显存预算。

**没有实现通用的运行时自适应 CE／RDMA 切换。** 源 token 数阈值是每 rank Dispatch 输入行数，不是请求并发或全局 token 数；不能让不同 rank 随意选不兼容的协议。

## 4. Attention：修复 MTP 验证阶段的不合适路径

GLM 的一次验证只带少量候选 token，但旧路径按 Extend 方式扫描长前缀，Attention 成本异常升高。受限修复对符合条件的因果验证复用并行扫描 KV 的 Decode 路径；每个候选位置的可见长度仍不同，不允许看见后续候选。其它 mask／形状回退原实现，不把所有 Prefill 强制变成 Decode。

| 同语义 Bank／同请求形状的历史对照 | 旧验证 Attention → 修复后，毫秒／每卡／目标 Graph | 输出吞吐 token/s | 平均 TPOT，毫秒 |
|---|---:|---:|---:|
| EP8 | 78.27 → 17.55 | 96.50 → 131.59 | 69.72 → 46.74 |
| PP2×EP4，第 1／第 2 阶段 | 39.14／39.16 → 8.75／8.77 | 88.28 → 128.56 | 77.18 → 52.79 |

条件：065／167、8×A100，8 请求，每条逻辑旧上下文 4096 token；原生 CPU HiCache 实际缓存 4032 token，计时内补算 64 个旧位置并新增 1 token，输出 64；MTP 开、目标验证 3 个位置、CUDA Graph 开；DeepEP pair 内 NVLink、pair 外 RDMA，没有 CE。旧实现来自同 Bank 历史数据，不是本轮重跑控制组。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_mtp_verify_fixed_ep8_pp2ep4_20261023/RESULTS.md)

内核路径改善已测到；但 EP／PP 输出逐 token 等价、MTP 与不开 MTP 的严格无损等价**仍未通过**。本记录不把它写成通用无损 MTP 修复；优化在报告所用入口中显式开启。

## 5. PP：不只压通信内核，还要减少反馈与调度依赖

### 5.1 GLM 的 MTP Decode 优化链

本节均为上一节的 8 请求语义缓存负载，PP2×EP4、两个各 4 请求的槽位。除层数分配外不减少模型层、专家或输出预算；源 Bank 导入和权重加载不计时，原生 HiCache→GPU 换入计时。通信库和专家内核不变。

| 改动 | 为什么改 | 输出 token/s ↑ | 平均 TPOT，毫秒 ↓ | 验证与范围 |
|---|---|---:|---:|---|
| 尾阶段本地消费已有反馈 | 原先尾阶段已有结果，却还等首阶段转回；保留必要的尾→首反馈与激活交接 | 128.603 → 132.586 | 52.818 → 50.976 | 与相应 PP 基线输出完全一致。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_pp_stage_balance_20261023/RESULTS.md) |
| 按 MTP 实际工作平衡层数：46／46 → 50／42 | 草稿计算只在尾阶段，目标层数各一半不等于总工作各一半 | 132.586 → 140.726 | 50.976 → 47.398 | 同一语义 Bank 可按逻辑层重排；不是所有 batch 的通用最优分层。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_pp_stage_balance_20261023/RESULTS.md) |
| 精简并无损合并反馈包 | 去掉 3 个不消费字段，将 7 个整数张量无损打包为一个 GPU byte packet，减少传输调用 | 141.013 → 141.881 | 47.286 → 46.991 | 同进程原版→候选→候选→原版，完整输出一致；prefill 不变。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_pp_feedback_payload_ab_20261024/REPORT.md) |
| 复用准确的 CPU 序列长度 | 草稿规划前已取得同一份准确长度，去除之后重复的 GPU→CPU 回读，保留必需读取 | 141.465 → 142.863 | 47.125 → 46.643 | 同进程配对，值核对和输出一致；不省略接受结果处理。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_pp_seq_cpu_reuse_20261024/REPORT.md) |

每行使用本轮自己的基线，不能把不同轮次当作一个固定机器上的连续加速公式。上述接受范围是**保持已有 PP 输出不变的局部调度优化**；跨布局无损 MTP 资格仍是另一项未解决问题。均为显式启用的冻结 runtime 适配，不代表共享引擎全局默认已更新。

### 5.2 Qwen PP2 Prefill：数据就绪后再发布发送通知

原发送端先通知元数据，接收端可能提前启动 NCCL kernel 等数据，干扰生产端计算。候选先等待已经接入当前 CUDA 流的生产事件，再调用原生发送函数。没有改模型、精度、张量或路由；等待仍在请求时间内。

同节点 069 GPU0／1、NVLink、Qwen3.5-122B-A10B-FP8（A100 W8A16 Marlin）；8 请求，每条无旧缓存、新输入 4096、输出 1；两槽 PP2、每阶段 24 层，Attention TP1／Expert EP1。CUDA Graph 和 overlap schedule 均关闭；同进程两策略均预热，正式原版→候选→候选→原版，独立采 GPU trace。

| 指标 | 原 PP2 | 发送就绪候选 |
|---|---:|---:|
| 输入吞吐，token/s | 8160.97 | **9391.74** |
| 平均 TTFT，秒 | 2.3980 | **2.1383** |
| 第 1 阶段 Attention，毫秒／层，含投影 | 9.2318 | **7.9523** |
| 其中投影 GEMM，毫秒／层 | 6.3800 | **4.9118** |
| 第 1 阶段路由 Expert，毫秒／层 | 8.3115 | **6.9984** |

Prefill 吞吐提高 **15.08%**；第 2 阶段变化很小。约 301 ms 的最长 NCCL 驻留缩至约 0.1 ms，表示提前等待被移到生产端就绪事件前，不是 NVLink 物理传输快了几千倍。正式 32 条 Prefill 的完整 token 一致；这是当前已预热两槽 Prefill 的局部验收。

**Decode 不能据此宣称提速或退化：** 原版自身重复与候选自身重复都输出不稳定，未通过等价判定。候选保持实验私有，未默认用于 Decode。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q02.md)

<!-- BEGIN Q03 H100 -->
### 5.3 H100：原生 Graph 与线性 Attention 状态缓存预算

Qwen3.5-122B 的 H100 对照启用原生 FP8 W8A8 DeepGEMM、FA3、Decode Graph；EP2 Decode 使用低延迟 DeepEP。PP 始终两个等大槽位，Prefill 发送就绪局部 A/B/B/A 约 +1.08%；没有把未验收的 Decode 候选打开。

排查发现 KV 尚有空间时，沿途保存的中间线性状态也会挤掉完整前缀。仅将 PP 状态池增至 128 时，KV 逻辑容量为 475712 token，且自身容量点仍不能完整命中；最终用原生每路径状态剪枝、PP 每阶段 84 个状态，KV 为 750848 token，16×42176 的基础前缀及全部正式请求命中通过。此处同时调整了状态保留策略和预算，是缓存容量／公平性修正，不是 DeepEP 内核吞吐提速。

同工作量 Decode 仍是 EP2 更快，PP 吞吐低约 20%–23%；不能因 PP 的层内通信减少就认定端到端一定占优。PP 在该引擎版本不支持 overlap，跨代还同时改变 FP8 计算和 Graph 路径，不能全部归因于 GPU。

[协议、完整指标、profile 和测试证据](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q03.md)。
<!-- END Q03 H100 -->

<!-- BEGIN Q05 H100 PREFILL OPT -->
### 5.4 H100 35B：按负载选择 Prefill chunk，并验收发送就绪

同节点178、PP2 GPU2／3，Qwen3.6-35B-A3B BF16。8请求各无旧缓存、新输入4096时，chunk4096→8192使吞吐77992.5→86284.9（+10.63%），整批420.14→379.77ms；8个正式cohort输出逐请求一致。不是增大请求数，也没有少算输入。

32请求各GPU已缓存89,984、新输入512时，保留chunk4096，复用A100的“生产端数据就绪后再发布发送通知”：吞吐28438.3→29598.2（+4.08%），TTFT403.48→389.68ms；四个相同输入cohort输出一致。候选仍为显式启用的实验私有适配。

长缓存不能照搬8192；2048／1024也更慢。独立profile显示2048把forward从5次增至9次，Attention累计量基本不变而Expert成本上升，NCCL驻留更短不等于端到端更快。有界5ms合批也未采用。每策略只有两次，未重测充分调优EP2；本轮没有Decode或TPOT结论。[全部数据、负结果和两阶段profile](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q05.md)。
<!-- END Q05 H100 PREFILL OPT -->

### 5.5 H100 35B Decode：提前下一槽位的真实 GPU 计算

本轮固定第 6 节的通用小批量 Expert GEMM，EP2／PP2 都启用相同内核家族，再优化 PP 的阶段交接和 token 反馈调度。同节点178、同一 NVLink GPU0／1、Qwen3.6-35B-A3B BF16；全局8请求，每条真实 GPU 已缓存4096、新增1、生成128 token，temperature=0、ignoreEOS、无推测解码。EP2 为 Attention DP2／TP1、MoE EP2；PP2 为两个20层阶段、两个固定B4槽位、Attention TP1／MoE EP1。每配置5轮正式 cohort，关闭 profiler；前缀构建、预热和验证不计时。

| 配置 | TPOT，ms/token ↓ | 输出吞吐，token/s ↑ |
|---|---:|---:|
| EP2，同等通用 GEMM 优化 | 7.657 | 890.4 |
| PP2，同等通用 GEMM 优化 | 9.303 | 804.9 |
| PP2，提前发送 activation | 8.793 | 849.4 |
| PP2，下一槽位提前准备并发射 | 7.184 | 1023.1 |
| PP2，扩大受保护的提前发射区间 | **6.998** | **1048.3** |

实现包含三步：forward 返回后按原 producer event 提前提交 activation；收到真实 GPU token 后提前准备下一步 KV 位置／序列长度，同时保留旧输出处理 snapshot；在等待旧输出 D2H 和 CPU 处理前发射下一槽位的原生 forward，正常调度到该槽位时复用同一 result／event。保留采样、GPU token relay、发送完成和 Graph buffer 复用依赖，避免重复 forward／发送／FIFO 入队；activation 和两个 token relay 使用独立消息通道。宽窗口从已生成2个 token 后启用，结束前及 Mamba/GDN 状态边界保留保护。

相对同轮 PP 基线，最终 TPOT 降低 **24.8%**、吞吐提高 **30.2%**；5轮 TPOT 均值范围 **6.989–7.013 ms/token**，结论为约7ms。独立稳态 profile 每 rank 丢弃前20个 Graph、再选16个：PP 模型 Graph 合计 **6.110→6.106 ms**，Expert GEMM **66.30→66.39 µs／GPU／层**，而前向交接与反馈区间合计 **4.936→0.727 ms**。收益来自计算与 CPU 输出处理重叠。该区间包含调度、对端依赖及其他槽位工作；profile 与正式 TPOT 分属不同执行，不能把区间当作纯 NVLink 传输或直接拼成正式 TPOT 分解。

最终各 PP 调度版本的 **20,480个正式输出 token 逐位置一致**；payload／event／槽位生产消费与排空检查、12项CPU测试通过。验收范围是静态 PP2、两组固定B4、无 stop／penalty／logprob／speculation 的 Decode；动态到达与任意batch未验收。三轮冷 Prefill4096→1仅作基线控制。实现保留为实验私有 adapter，共享引擎默认未改。

[完整报告与复现入口](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_graph_gap_optimization_20261027/REPORT.md) · [性能与依赖区间图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_graph_gap_optimization_20261027/pp_decode_scheduling.png) · [逐轮优化记录，含无收益尝试](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_graph_gap_optimization_20261027/optimization_history.csv) · [冻结调度实现](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_graph_gap_opt_20261027/runs/shared_gemm_final_r8/code/gap_schedule.py)。

## 6. Expert GEMM：减少小批次执行碎片化，并在 EP／PP 同等启用

### 6.1 问题定位与实际改动

H100 35B 的 PP 每层本地驻留256个 Expert，但所测 B4 Decode 每次平均仅触及约11个、产生32个有效 token–Expert assignments。控制同一真实输入、所选权重、路由和有效计算量，并保持共享权重 bank 的地址不变，仅增大传给原生 kernel 的本地 Expert 维度时，层0的 Expert-only Graph 在 E32／E64／E128 下为 **47.16／56.48／70.15 µs**。这说明不活跃维度的布局／执行处理也有成本；不等于所有 padding 的 GEMM FLOPs 都实际执行。[碎片化定位、同输入内核对照与原始数据](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_expert_fragmentation_20261026/summary.md)。

最终按实际本地输入 token 数选择路径：**小形状 T≤4 使用现有 BF16 small-tile Triton GEMM，大形状保留 DeepGEMM**。EP 使用同一 `invoke_fused_moe_kernel` 内核家族，并增加 GPU 索引适配，将 DeepEP 分组 buffer／每专家有效行数映射给共享内核；适配开销包含在计时内。EP 保留原生 masked SwiGLU 和 DeepEP 加权 combine；两边均保留原有路由、权重精度和 Attention。

### 6.2 双方应用通用优化后的公平对照

独立于第5.5节的早一轮对照：同一 H100178 NVLink GPU0／1、完整 Qwen3.6-35B-A3B BF16、全局8请求、每条 GPU缓存4096＋新增1→输出128；每配置5轮正式测量。Expert 列来自独立原生 Decode Graph，单位为 µs／GPU／实际层调用；EP 新增索引 kernel 另有约3.22 µs／层，已计入请求指标。

| 布局 | 原生→共享小批量 GEMM，µs／层 ↓ | 原生→优化 TPOT，ms/token ↓ | 原生→优化输出吞吐，token/s ↑ |
|---|---:|---:|---:|
| EP2 | 59.60→47.74 | 7.956→7.663 | 860.5→886.9（+3.06%） |
| PP2 | 94.39→57.31 | 11.184→9.277 | 666.3→790.2（+18.61%） |

这属于双方均可采用的通用内核优化；第5.5节以两边都启用该优化为基线，再验证 PP 专属调度收益。两轮 PP 请求分组和采样区间分别记录，各轮数值不拼接成连续加速公式。

真实路由、有效 assignment 索引和同输入层输出检查通过；BF16 内核替换不保证与原 DeepGEMM 完整生成逐 token 相同，也没有任务质量结论。第5.5节的输出一致性仅针对 **GEMM 固定后的 PP 调度版本**。Prefill 大形状仍走 DeepGEMM，该通用优化未建立 Prefill 提速结论。

[公平对照完整报告与数值验证](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_ep_pp_fair_expert_20261026/summary.md) · [完整性能指标](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/analysis/metrics.csv) · [公平对照与组件图](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/deliverables/fair_ep_pp_comparison.png)。

## 7. 权重／KV Bank：缩短准备时间，也修正测量公平性

| 实现或流程修正 | 解决的问题 | 本文能支持的结论 |
|---|---|---|
| 统一 MemoryBank | 权重和 KV 不再各自假设全部 CPU 内存可用；统一条目、布局与 CPU／cgroup／tmpfs 联合准入。Guard 负责租约，WeightLoader 负责读张量 | 职责已独立到 `sglang_memory_bank`，旧 KV 包名是兼容入口，不是另一套策略。不是增加物理内存。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/sglang-memory-bank/README.md) |
| 单一合成前缀池与请求视图 | 不为每个总上下文／请求数组合重新运行 Prefill、重建完整 Bank；在不可变池上建立不拥有 payload 的请求视图 | 用于不评价生成质量的容量实验；不意味着请求共享 GPU KV 页，也不免除真实 CPU→GPU 传输。MTP 接受率实验必须另用模型生成的语义 KV。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/sglang-memory-bank/README.md) |
| 按层窗口权重 Bank，显式有界流式 fallback | 完整 CPU 模型副本放不下，或 EP8／PP2 每次都要重建不同布局的缓存 | 存完整层的所有专家，消费时选择 EP／TP／PP 视图；缺层才走已准入的有界流式路径。部分窗口不能标为完整模型 READY；每卡 GPU 仍只加载该布局所需权重。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/hy3_layer_window_bank_20261019/README.md) |
| 源池预置原生 HiCache，再统一开始计时 | 避免把自建 CPU 前缀池读取／格式转换算成 HiCache 恢复，夸大副本和 PP 的差距 | 两组均从“CPU HiCache 已就绪、目标 GPU KV 为空”开始；计时保留原生 H2D、排队和计算。早期口径结果仍保留，不与新口径合并。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_h100_hicache_prefill_retest_20261019_v1/summary.md) |

统一库 README 对通用前缀池的验证仍明确区分 CPU fixture／接口测试和真实集群运行；具体性能结论以对应实验的真实恢复证据为准，不把接口存在当作所有模型／布局均支持。

## 8. 保留的失败、负收益和未完成项

| 尝试 | 当前结论 |
|---|---|
| 强制旧 SM PCIe 承担大 Prefill | 完整 GLM 明显退化；这不是 CE。总览 P05 保留 PP 485.59、EP8 206.20 input tok/s，不能因名字也有“PCIe”就与 CE 混用。 |
| CPU 代理通知批量、线程亲和、错峰、RDMA-first／drain 等迭代 | 有的修复正确性或改变等待分布，但未建立跨 workload 的稳定收益；不能用来承诺混合 PCIe 必胜 RDMA。详细 runtime 仍在原实验目录。 |
| 早期手工 MTP microbatch | 实现了固定请求组／槽位生命周期，诊断显示长 NCCL 等待减少；但原版和候选都重复输出不稳定，报告为 correctness_unconfirmed，不计为已验证提速。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_pp_microbatch_atomic_20261023/RESULTS_DIAGNOSTIC.md) |
| 提前尾阶段反馈、复用 Prefill 草稿 | 前者未改善正式 TPOT；后者退化且同 PP 输出只有 4／8 相同，已拒绝。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_pp_feedback_payload_ab_20261024/REPORT.md) |
| 固定形状 Prefill layer Graph | 有计时，但 profile 写盘因磁盘配额失败，整轮未验收；不能宣称已解决 GLM 第一槽的启动等待。见总览 O04。 |
| CE r6 的 ACK 顺序 | 即使测试一度通过，源码复核仍发现 metadata／ACK 竞态；结论使用修复后的 r7，不采用 r6。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/deepep_prefill_pcie_optimization_20261010/summary.md) |
| 看似通信长尾，实际是测试 CPU 准备触发 GC | 复现样本 28.573 ms 中，rank 起跑偏差约 20.959 ms，Dispatch DMA 仍约 1.969 ms。把参考构造移到 barrier 前可消除该次污染；诊断开 profiler，不能据此报告生产加速。[证据与原始来源](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/deepep_dispatch_tail_diagnosis_20261008_r1/summary.md) |
| H100 35B compact DeepGEMM、新 route GEMV 和继续调小 tile | 同实际输入的 compact／GEMV 均慢于所选 small-tile Triton；tile 微基准略快未转化为稳定完整模型收益。未选入最终方案。[证据与候选数据](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_expert_fragmentation_20261026/summary.md) |
| H100 35B 仅取消尾阶段 echo、拆分通道或增加 activation stream | 各轮自己的对照未显示稳定额外收益；提前 metadata 有小幅收益，主要改善出现在进一步提前真实 GPU forward 后。独立通道／stream 是最终原型的组成部分，不把单项收益重复计数。[逐轮数据](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_graph_gap_optimization_20261027/optimization_history.csv) |
| H100 35B 异步 depth1＋共享无类型 channel | 消息顺序变化后无法可靠区分 activation 与反馈，候选停顿并已终止清理；部分计时不计为有效结果。另一个 typed-depth1 目录仅准备未运行，不能报告性能。[全部尝试及状态](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_graph_gap_optimization_20261027/manifest.json) |

## 9. 当前应怎样使用这些结论

1. **大消息节点内通信**优先考虑已验证 CE；小消息与 Decode Graph 保留各自有资格的 SM／RDMA 路径。CE 不是全局自动开启项。
2. **GLM MTP 的 PP 开销**不能只看 DeepEP：还要看验证 Attention、尾阶段草稿工作、反馈与 CPU 回读。后几项优化没有更换 DeepEP。
3. **Qwen 两槽 Prefill**已有输出一致的发送就绪收益；第5.2节122B候选的历史 Decode 可重复性问题仍按原记录保留。H100 35B 的固定B4 Decode lookahead 已独立验收，不能把这份资格外推到其它模型或动态batch。
4. **Expert 与调度分开评价：**通用 GEMM 改进应同时给予 EP／PP。PP2 约7ms的新增收益来自两槽位调度重叠；部署成本模型还应记录未被隐藏的 CPU／反馈依赖，不能只比较有效 FLOPs、单层 Expert 或通信字节数。
5. **所有后续比较**沿用总览中的准确负载和缓存起点，保存实际 native 哈希、显式开关、无 profiler 请求测量、独立 profile 与正确性状态。不同轮次的收益不直接累加。

本文件随有意义的优化继续更新；每项链接原报告／源清单，CSV、trace 和原始失败证据不重复搬运。精确文件哈希与本次链接校验见同目录 `optimization_sources.json`／`optimization_validation.json`。

<!-- begin qwen35b_a100_nvlink_regression_20261027 -->
## A100 NVLink上的PP lookahead复测（2026-10-27）

完整35B BF16，A100069物理GPU0/1实测NV12；EP2 Attention DP2/TP1，PP2两个20层stage、两个最多B/2请求槽。双方均为原生Triton Expert；DeepEP normal BF16，CUDA Graph及overlap schedule关闭。六组workload、每配置5轮，共15个计时点和30份rank profile；3次暖机，使用真实GPU-hot4096+new1→128 Decode及冷Q128/Q4096→1 Prefill。

PP9相对原生PP0的Decode吞吐变化：B8重复输入−0.75%、B32重复输入+0.36%、B32多样输入+0.73%。本轮调度增益均在±1%内，H100的lookahead收益不能直接照搬；原生PP相对EP的Decode吞吐高34.78%／37.42%／38.46%。通用Expert内核在双方相同，PP9只改变调度。

独立B32多样输入profile（µs／卡／实际层调用）：routed Expert EP263.26／PP320.02，Dispatch+Combine EP99.14／PP0，Attention EP136.36／PP135.84。实际分组、PP0/PP9全128-token输出及通信payload hash验证通过；EP／PP长Decode序列的分歧另列，不等同于任务质量验证。GDN状态池采用已验证预算，避免B32容量排队。Guard、清理、释放通过。

[完整报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_a100_nvlink_regression_20261027/report/REPORT.md) · [算子分项图](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_a100_nvlink_regression_20261027/report/components.png) · [指标](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_a100_nvlink_regression_20261027/analysis/metrics.csv)
<!-- end qwen35b_a100_nvlink_regression_20261027 -->
