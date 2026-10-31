# EP2 ↔ PP2 原地切换的可行性评估

日期：2026-10-29。性质：论文定向核查、当前代码检查与配置推算；没有运行新的 GPU 实验，也没有测得 EP/PP 切换延迟。

结论：技术上可行。主要增加的复杂度是整层所有权、KV/GDN 状态、请求调度和流水执行图的转换，不能仅用“要重新搬整个模型”判断其代价。均分层、均分专家的 EP2 ↔ PP2，跨卡搬运一半 routed-expert 权重即可。在线收益必须覆盖实际切换暂停，并满足目标布局的显存与延迟约束。

## 1. Moebius 的前提需要修正

核查版本：[Moebius arXiv v1](https://arxiv.org/html/2606.26607v1)，本地 PDF 17 页。

- §2.1 / PDF p3：比较的是 Attention TP + Expert TP 与 Attention DP + Expert EP。
- §3.1–3.2 / p5：Attention 布局和请求/KV 所有权都会改变。默认预留两套 Attention 权重，避免切换时搬这些权重；KV 仍然迁移。Figure 5 已目视核对。
- §4.2–4.4 / pp6–8：共享固定地址存储，保留两种运行时和 CUDA Graph，搬运变更所有权的数据。
- §6.4 / pp10–11：8×H200、Qwen3-235B-A22B 的切换为 215–434 ms。这是其特定实现和负载的结果，不能当作两张 H100 上 EP/PP 切换的预测。

因此，“Moebius 只搬 Expert、不改变 Attention/KV”的前提不成立。它提供的设计思路可以借鉴，论文并未实现或验证 PP↔EP。

## 2. 用四层、两组专家说明所有权变化

以下对应我们现有 EP2 的 Attention TP1/DP2，以及 PP2 的每阶段 Attention TP1、Expert EP1。每层专家分成 X、Y 两组；请求分成 A、B 两组。

| 状态 | GPU 0 | GPU 1 |
|---|---|---|
| EP2 Expert 权重 | 第1–4层的 X | 第1–4层的 Y |
| PP2 Expert 权重 | 第1–2层的 X+Y | 第3–4层的 X+Y |
| EP2 活跃缓存 | 请求 A，第1–4层 | 请求 B，第1–4层 |
| PP2 活跃缓存 | 请求 A+B，第1–2层 | 请求 A+B，第3–4层 |

GPU 0 把第3–4层 X 发给 GPU 1；GPU 1 把第1–2层 Y 发给 GPU 0。保留的权重无需离开本卡，也不需要从磁盘重新加载。

缓存同理：GPU 0 发出 A 的后半层缓存，GPU 1 发出 B 的前半层缓存。状态完整且格式兼容时，不需要重算 Prefill。两端 Attention 都是 TP1，不必切分/重组 attention heads；但层归属、请求服务方式和 KV 的本地地址映射改变了。

Attention/GDN 等非 routed-expert 权重还存在方向不对称：EP2 原本每卡都有全层副本，EP→PP 可直接保留本阶段需要的部分。PP→EP 若先前释放了另一半层的权重，就需要补齐。预留这些权重可以减少反向搬运，但必须计入显存预算。Embedding、LM head、共享 Expert、router 和量化元数据也要分别处理。

## 3. 实际模型对应的绝对权重字节量

按本地实际配置，计算文本主干 routed-expert 三个投影矩阵的逻辑权重：

`W_E = 3 × 层数 L × hidden H × expert intermediate I × expert 数 E × 每元素字节数`。

均匀 EP2 与均分层 PP2 之间：每卡发送 `W_E/4`、接收 `W_E/4`；两个方向合计网络传输 `W_E/2`。后者只计发送一次，不把同一数据的发送和接收重复相加。

| 模型及精度 | L / H / I / E | 全模型 routed-expert 权重 | 每卡发送；并接收同量 | 全组两个方向合计 |
|---|---|---:|---:|---:|
| Qwen3.6-35B-A3B BF16 | 40 / 2048 / 512 / 256 | 60 GiB | 15 GiB | 30 GiB |
| Qwen3.5-122B-A10B FP8 | 48 / 3072 / 1024 / 256 | 108 GiB | 27 GiB | 54 GiB |

这里是配置推算，不是实际分配器占用。122B 未计 FP8 scales、padding/packing；两者都未计非 routed-expert 模块、额外草稿模型和运行时。35B 使用20+20层，122B表中使用24+24层；不等长阶段需重新按每卡实际发送量计算。

**仅作带宽敏感性示例：假设每 GPU 单向有效传输达到 200 GB/s（十进制），并且双向能同时达到该速度**，则只搬 routed-expert 的理想传输时间为：

- 35B：`15 × 2^30 / (200 × 10^9) = 80.53 ms`。
- 122B：`27 × 2^30 / (200 × 10^9) = 144.96 ms`。

200 GB/s 不是本轮实测带宽，也不是把 NVLink 宣称的双向总带宽直接代入。实际暂停还包括流水收尾、KV/GDN、局部重排、同步与提交；此前 token dispatch/combine 的小消息曲线不能替代大块权重交换测量。

## 4. KV 与 GDN 可能改变结论

设全组活跃、唯一的 Attention KV 总字节数为 K；层和缓存 token 负载均衡、无额外副本且存储格式相同的理想情况，每卡需要发送 `K/4`。不均衡情况下用实际请求×层的归属计算，关键路径取较慢 rank；不能只按请求数量均分。

两种模型都是混合 Attention/GDN：35B 有10个 full-attention层、30个linear层；122B有12个full-attention层、36个linear层。假设 KV 为 BF16、2个 KV heads、head_dim=256，full-attention KV 分别为20 KiB/token与24 KiB/token，不应按所有层均为普通 Attention 计算。

| 假设场景，不是新测量 | 全组 full-attention KV | 均衡时每卡需发送的 KV |
|---|---:|---:|
| 35B，32请求，每请求8192 cached tokens | 5 GiB | 1.25 GiB |
| 35B，32请求，每请求131072 cached tokens | 80 GiB | 20 GiB |
| 122B，32请求，每请求8192 cached tokens | 6 GiB | 1.5 GiB |

这些只算 KV，不证明该场景能放入 GPU。35B 的长上下文例子还需扣除全部权重、GDN、图、临时区与缓存容量余量。122B 若32请求各131072 tokens，单是108 GiB routed权重与96 GiB BF16 KV就超过两张80GB GPU的总显存，不能作为全GPU驻留候选。

GDN 的卷积状态、循环状态需要按请求×层迁移到同一逻辑生成位置；请求长度、采样状态、slot/page表也要一致。若启用 MTP，还需处理草稿状态。历史长上下文数据涉及 CPU HiCache，不能把其全部缓存假设为 GPU resident；CPU KV/GDN 检查点、前缀引用和未完成回写均需单独协议。非活跃 prefix cache 可以选择失效，活跃状态则需要保留、迁移或明确承担重算成本。

## 5. 真正的工程难点与当前实现边界

建议的切换事务：

1. 检查目标显存和未来收益；暂停接收新工作进入本次切换。
2. 排空流水中的 microbatch 到安全边界，使每个请求的 logits/采样/缓存位置一致。无需让所有请求长度相同，也无需让请求全部生成完成。
3. 双方准备接收槽位，迁移变更所有权的 Expert、KV/GDN 和请求元数据；采用不会提前覆盖尚未发送源数据的分块计划。
4. 校验数据与映射，提交新的布局版本，启用已准备的通信组、运行时和执行图，再恢复调度。

EP 和 PP 的层数、层形状、collective/P2P 调用、调度依赖均不同，不能直接重用同一张 CUDA Graph。可借鉴预建双运行时、双图和共享稳定地址存储；但 PP 的层槽位大小/数量变化，需要重新设计别名布局与安全搬运顺序，不能宣称 Moebius 的布局不加修改即可使用。

当前只读检查的 SGLang HEAD 为 `74734b15fbfd96681b07663243309fef2b24bee5`；工作树并非全净，相关 `memory_pool.py` 有本地修改。本报告的 sources JSON 保存关键文件内容哈希。

- 已有弹性 PP 的层驻留、KV/GDN transport 和拓扑安装代码，可复用部分能力。
- `hybrid_kv_transport.py` 明确要求双方 allocator/slot identity 一致，当前不支持 slot remapping。EP 的两个 DP rank 可各自把不同请求放在同一 slot 编号；合并到 PP 请求空间时，必须新增请求和 GDN 槽位重映射，不能只搬同编号的数据。
- `hybrid_config.py` 当前还排除 prefix cache、HiCache、量化KV和 speculative decoding，并要求固定请求容量；首个原型应尊重这些已知边界。
- `scheduler_pp_mixin.py::_pp_install_topology` 要求覆盖现驻留 PP group，并在 microbatch ring 的 fence 后安装；这属于现有 PP 内的变化。
- `qwen3_5.py` 的 elastic PP 量化白名单为 `None` 与 `moe_wna16`；当前122B FP8不能直接套用该弹性路径。这不影响已有静态 FP8 PP 测量。
- 这些代码的存在不证明已经支持或验证了 EP2↔PP2 的 Expert 分片、DP请求、流水与全部活跃状态联合转换。

真正原地执行不必保留两份完整 Expert 权重，但需要交换 scratch、两套运行时元数据和可能预留的非 Expert 权重。两个静态布局各自能放下，不代表切换中间态也能放下。失败恢复必须在丢弃源数据前设计，不能边覆盖边假定可无成本回退。

## 6. 何时值得切换

同一完成工作量下，只有未来收益超过切换成本才有回报：

`Σ未来步骤 [T_旧布局 - T_新布局] > C_switch + 预测误差余量`。

`T` 必须包含完整执行、PP bubble 与暴露通信，不能把单层 Expert/通信图直接当作全模型的每步收益。Prefill 还受分块和请求组成影响；Decode 还受活跃 batch、context、缓存位置影响。

示例而非实测：暂停300 ms、切换后同一批请求每轮 Decode 节省2 ms，至少150轮才到收支平衡；若还要付同量反向切换成本，则至少300轮。如果请求的 ITL 预算为50 ms，300 ms 暂停即使最终吞吐回本，也不能忽略该次延迟违约。

因此不建议逐token按阈值来回切换。控制器至少应观察：目标布局的实际 step cost、要迁移的活跃状态字节数、目标显存、预计有利区间持续时间，以及暂停预算；结合迟滞与最短驻留时间。

## 7. 建议推进顺序

- 优先用已有静态 EP/PP 实例做新请求分流，验证“稳定、可预测、足够大的优势区间”是否存在。四张卡可安排一个EP2和一个PP2，但要与同样四张卡的最佳同构池比较；不能忽略混合实例的容量成本。请求尽量保持在原实例，避免活跃缓存搬迁。
- 若只有两张卡且必须复用，先做无活跃请求或小缓存条件下的低频切换，再做带请求原地切换。35B BF16、两张同NVLink域H100、GPU驻留短KV、无HiCache/MTP，是较小的首个实现范围。不是新开两个完整模型进程。
- 最小验证记录两个方向的权重交换、KV/GDN交换、流水排空和恢复耗时，以及显存峰值、恢复后的logits/序列状态、长尾ITL与净完成时间。验证已有生成工作不丢失、不重复；浮点路径不同用合适数值容差，不能要求所有长序列天然逐token一致。
- 在这个范围成立后，再扩展长上下文、CPU HiCache与122B FP8。GLM/HY3延续此前至少EP4约束：候选是EP8↔PP2×EP4，而不是纯PP；两张卡也不能同时实现独立并行轴上的PP2×EP2。

## 可追溯来源

- [实际35B部署、Attention/Expert拓扑与未测动态切换项](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_five_layouts_hicache_20261028/two_gpu_three_layouts_analysis/REPORT.md)
- [122B静态EP/PP与CUDA Graph匹配复核](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_qwen122b_pp_refine_20261027/report/REPORT.md)
- [35B配置](/data/gpfs/projects/punim2715/huggingface/Qwen3.6-35B-A3B/config.json)
- [122B配置](/data/gpfs/projects/punim2715/huggingface/Qwen3.5-122B-A10B-FP8/config.json)
- [当前Qwen模型弹性PP入口](/home/bxb1/vllm_workbench/sglang_autoscaling/sglang/python/sglang/srt/models/qwen3_5.py:1339)
- [当前PP拓扑安装与安全边界](/home/bxb1/vllm_workbench/sglang_autoscaling/sglang/python/sglang/srt/managers/scheduler_pp_mixin.py:131)
- [推算与内容哈希](EP_PP_INPLACE_SWITCH_FEASIBILITY_20261029.sources.json)

检索标记：ep_pp_inplace_switch_state_audit_20261029。
