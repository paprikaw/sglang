# Q01 补充：为什么 Prefill 的吞吐、请求延迟和 Attention 时间不同向变化？

2026-10-25，仅重新解析已有 trace，没有启动 GPU 或生成新的性能测量。

## 1. 先说明术语与比较范围

本报告不用单字母代替请求数或上下文长度。**Prefill** 是 8 个请求，每条没有已缓存上下文，新输入 4096 token，生成 1 token；合计 32768 个新输入 token。**EP2** 表示同层专家跨两卡；**PP2** 表示模型分成两阶段，每阶段一张卡。**槽位**是调度器轮转的批次位置，不是阶段数，也不是同时执行的 GPU 数。

**TP** 是张量并行，**DP** 是不同请求之间的数据并行。**TTFT** 从请求放行到首输出；此处只输出 1 token，因此与端到端延迟相同。**吞吐**取整批输入 token 数除以整批完成时间，再对三轮取平均。**P99** 是三轮共 24 条请求的最近秩 99 百分位，本样本里等于最大值。**TPOT**只适用于首个输出之后的生成，本 Prefill 场景不适用。

**Attention 活动**包括线性投影和注意力核心，不只是注意力配对计算。**Marlin** 是本轮 W8A16 矩阵计算内核；**NCCL** 是 PP 阶段收发使用的 GPU 通信库；**SM** 是 GPU 执行线程块的计算单元。1 ms 为一毫秒；1 KiB 为 1024 字节。

这里的“重叠比例”是 Attention 的 GPU 活动区间与 NCCL kernel 区间的交集除以前者。它**不是通信占端到端延迟的比例**，也不是 GPU 空闲比例。NCCL kernel 的驻留可以包含等待，不代表这段时间都在搬运数据。

## 2. EP2 没有使用 Attention TP2

| 配置 | Attention 实际布局 | 每卡负责的层 | Full Attention 每卡每层的头数 |
|---|---|---:|---|
| EP2 | TP1×DP2；每卡处理不同请求，专家仍参与 EP2 | 全部 48 层 | 32 个 Query head、2 个 KV head |
| PP2，两槽或四槽 | 每阶段 TP1；同一请求依次经过两阶段 | 连续 24 层 | 32 个 Query head、2 个 KV head |

EP2 启动参数的 `tp_size=2` 与 `enable_dp_attention=True, dp_size=2` 共同作用，原生代码计算 Attention TP 大小为 2÷2＝1。因此，Attention 没有被拆成两卡各算一半的头。

相同八请求下，每卡 Attention 的总工作量也基本对应：EP 每卡 4 请求×48 层；PP 每卡 8 请求×24 层，都为 192 次请求层调用。本次每条 Prefill 输入块都是 4096 token，没有使用更长的 PP 上下文。不能把 PP Attention 变慢解释成“每卡多了一倍 Attention 工作”。

配置证据：[EP2 实际服务信息](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/runs/qwen122_ep2_pp2_r7/ep2/server_info.json)、[PP 服务信息](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/runs/qwen122_ep2_pp2_r7/pp2/server_info.json)。原生解析公式见 [dp_attention.py](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/glm47-ep16-canonical-profile-20261004/python/sglang/srt/layers/dp_attention.py:326)。

## 3. 四槽提高了后续完成速度，但推迟了较早请求的输出

以下为三轮请求测量，不开 profiler。首个输出列是各轮最早输出时间的均值；整批完成列是各轮整批耗时的均值。吞吐按每轮计算后平均，不能拿平均整批耗时精确倒推平均吞吐。

| 配置 | 输入吞吐 token/s ↑ | 首个请求输出 秒 ↓ | 整批完成 秒 ↓ | 请求 TTFT 均值 秒 ↓ | 请求 TTFT P99 秒 ↓ |
|---|---:|---:|---:|---:|---:|
| PP2，两槽 | 8185.11 | 0.776 | 4.003 | 2.389 | 4.020 |
| PP2，四槽 | 8810.58 | 1.212 | 3.719 | 2.503 | 3.728 |

四槽相比两槽：输入吞吐增加约 7.64%，平均请求 TTFT 增加约 4.75%，P99 反而降低约 7.28%。所以“其他指标都没有更好”并不准确：尾部改善了，但较早请求等得更久。

直观地说，两槽较早交出第一份结果，之后约每 0.46 秒完成一个请求；四槽较晚开始交结果，之后的完成间隔大多缩短到约 0.32–0.38 秒。整批的最后一个请求更早完成，平均请求却可能更晚完成。吞吐看最后完成边界，平均延迟看全部请求的完成分布，二者不是同一个目标。

原生调度也不只是“多两个位置”：额外异步深度非零时，取回旧输出的逻辑在当前批次 launch 前调用；深度为零时在 launch 后调用。它改变了输出缓冲、收发依赖与计算重叠方式。因此本对照不能把所有变化单独归因于槽位数量。见 [scheduler_pp_mixin.py](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/glm47-ep16-canonical-profile-20261004/python/sglang/srt/managers/scheduler_pp_mixin.py:555)。

## 4. Attention 的慢点会跟随 NCCL 重叠的位置变化

下表是独立 profiler 采样，每卡每次实际层调用的 Attention GPU 活动均值。不是请求 TTFT。每份 trace 有 192 次 Attention 层调用；Full 与线性类型按实际调用加权。

| 配置／位置 | Attention 毫秒／层 | Attention 活动与 NCCL 重叠比例 |
|---|---:|---:|
| EP2，第一张卡 | 7.591 | 0% |
| EP2，第二张卡 | 7.679 | 0% |
| PP2 四槽，前半阶段 | 7.831 | 0% |
| PP2 四槽，后半阶段 | 8.644 | 31.3% |
| PP2 两槽，前半阶段 | 9.146 | 58.3% |
| PP2 两槽，后半阶段 | 7.838 | 0% |

四槽时后半阶段较慢，两槽时前半阶段较慢。慢点随调度与收发重叠位置移动，而不是固定在某一组模型层上。这里的 0% 仅表示没有 NCCL 与 Attention 重叠，不表示 EP2 没有 DeepEP 通信。

## 5. 主要异常在投影 GEMM，不是 Full Attention 核心同样变慢

以四槽 PP 后半阶段第 35 层为例，按相同 kernel 名称、线程块布局、层号和组件内调用次序匹配。每项有 5 次不重叠、3 次重叠观测；表内是对应 kernel 时间中位数。投影的维度来自固定的相同输入形状和操作位置，未另行采集全部 kernel 参数指针。

| 第 35 层的同一操作 | 不与 NCCL 重叠 | 与 NCCL 重叠 ≥90% | 变化 |
|---|---:|---:|---:|
| Marlin 投影，同形状调用第 1 次 | 0.7718 ms | 1.2679 ms | +64.3% |
| Full Attention 核心 `_fwd_kernel` | 5.0535 ms | 5.0619 ms | +0.17% |
| Marlin 投影，同形状调用第 5 次 | 1.3846 ms | 2.2620 ms | +63.4% |

这更符合“通信驻留干扰某些投影矩阵计算”，而不是“PP 改用了慢很多的 Attention 核心”。它不是说所有 Full Attention 核心都完全不受影响，也不能从一层外推一个确定的端到端收益。

## 6. 一个具体的资源冲突机制

原 trace 的 A100 属性是 108 个 SM、每 SM 167936 字节共享内存，即 164 KiB。投影 Marlin 的代表性 kernel 启动 108 个线程块，每块 256 线程，记录的共享内存需求为 166912 字节，即 163 KiB。长驻留 NCCL 收发 kernel 每块需要 96384 字节，即约 94.1 KiB。

这两种块需要的共享内存相加超过 164 KiB，**不能在同一个 SM 上同时驻留**。Marlin 的每块 255 个寄存器／线程、NCCL 的每块 640 线程×96 寄存器／线程也显示出很重的资源需求。不能把这种通信当成零 SM 开销的纯硬件拷贝。

一个合理推断是：Marlin 启动的块数恰好等于 SM 数，若部分 SM 被通信块占住，部分计算块需要晚一轮执行，少量占用也可能放大整个矩阵操作的尾部延迟。这能解释投影 kernel 明显拉长的方向；本轮没有采集实际 SM 驻留或频率计数器，也没有做移除通信的因果 A/B，尚不能精确量化该机制贡献了全部差额中的多少。

PP 的部分 NCCL kernel 驻留约 300 ms，其中包括等另一阶段算完或反馈就绪；**不是 NVLink 搬一次激活需要 300 ms**。问题可能是提前提交了一个长期等待但占执行资源的收发 kernel。

## 7. 当前结论与下一步最小验证

当前最明确的是：Attention 没有 TP 大小差异；每卡 Attention 总量相当；槽位改变会移动 NCCL 重叠与慢点；同层投影在重叠时明显变慢，Full Attention 核心的代表样本基本不变。专家工作集成本仍然存在，但不足以单独解释这些 Attention 变化。

后续若优化，应先在已有单次输入形状上验证“延后提交未就绪的接收”、更轻的收发等待，或给通信留出资源的 Marlin 启动配置。每种方案都应同时检查正确性、稳态吞吐和单请求延迟；暂不把这些建议当成已取得的加速。本轮没有启动这些新实验。

## 8. 数据入口与限制

- [逐次 Attention 活动与 NCCL 区间交集 CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_prefill_diagnosis/attention_scopes.csv)。
- [按层／kernel 形状／调用次序匹配的观测 CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_prefill_diagnosis/matched_kernel_overlap.csv)。
- [trace 哈希、硬件属性和 kernel 资源记录](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_prefill_diagnosis/summary.json)；[复现本次 CPU 分析的脚本](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/analyze_qwen_prefill_tension.py)。
- [Q01 完整性能与口径说明](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q01.md)。原测量只有三轮，profile 为独立采样；重叠与变慢的关联不是经过硬件计数器隔离的因果证明。
