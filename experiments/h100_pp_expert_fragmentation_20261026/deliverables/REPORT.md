# H100 PP Expert execution：定位、优化与完整模型验证

实测确认：PP 小微批次的 Expert 成本有一部分来自执行布局，能够改善。保持同一对 H100、完整 BF16 模型和 PP2 调度，仅对本地 token 数不超过 4 的 Expert 调用选择小 tile Triton，Decode 吞吐 **666.8→805.7 output token/s（+20.83%）**，TPOT 降低 18.66%，请求端到端延迟降低 17.23%。原生 Graph 中 Routed Expert GEMM 平均时间降低 **38.83%**，Attention core 接近。

本轮优化没有稳定的 Prefill 收益；优化后 PP Decode 仍比 EP2 低 6.66%。所有策略保持 BF16，但生成 token 不完全一致，因此当前结果是有数值差异的性能实验，尚不是已完成质量评估的生产部署方案。

## 1. 相同资源与明确负载

| 项目 | 本轮协议 |
|---|---|
| 模型 | 完整 Qwen3.6-35B-A3B，BF16 权重与激活；FP32 GDN 状态 |
| GPU | spartan-gpgpu178，同一对 GPU0/1，H100 80GB NVLink；已有作业 33646188；各 arm 顺序运行 |
| EP2 | MoE EP2；Attention DP2 × TP1；全局参数 tp2/dp2/ep2/pp1 |
| PP2 | MoE EP1，Attention TP1；每 stage 20 层；两个流水槽位，每个微批次最多 4 请求 |
| Prefill | 8 请求；每条无旧缓存，新增 4096 input token，输出 1；总新增 32768 token |
| Decode | 8 请求；每条已有真实模型生成的 GPU 缓存 4096 token，再新增 1 token，输出 128；ignore EOS |
| 缓存 | 原生 GPU radix prefix；没有兼容的混合 GDN CPU checkpoint，未使用 CPU restore 或合成 KV |
| 调度与 chunk | PP ready-send 关闭；PP chunk4096；EP 全局8192经 DP2 分为每实例4096；本轮不改变这些配置 |
| 内核 | H100 原生 BF16 DeepGEMM，FA3；EP 原生 overlap；PP 原生 overlap 不支持；Decode CUDA Graph，Prefill eager |
| 计时 | 缓存准备、预热、编译、数值验证和 profile 均在正式计时之外；原生各 3 轮，最终 adaptive 各 5 轮 |
| 输入组成 | 固定 tokenized 文档提示，每请求不同起始 token、主体相同；这是受控工作负载，尚未覆盖多样化服务流量 |

Decode 预热覆盖总请求数 2/4/6/8，对应每个本地执行实例的 Graph 形状 1/2/3/4。分析重新检查了每条正式请求的缓存计数和输出长度。

## 2. 完整模型结果

Prefill 吞吐是新增 input token/s，Decode 吞吐是 output token/s，越高越好；延迟单位为 ms，越低越好。吞吐列是各 cohort 的均值 ± 样本标准差；端到端、TTFT、TPOT 是逐请求均值。TPOT = (端到端−TTFT)/(输出数−1)，不是把整个请求时间除以输出数。

| 配置 | 阶段 | 吞吐均值 ± SD | 吞吐中位数 | 请求端到端 | TTFT | TPOT | 轮数 |
|---|---|---:|---:|---:|---:|---:|---:|
| EP2 native | Prefill | 63,308.73 ± 4,363.49 | 65,593.66 | 396.11 | 396.11 | — | 3 |
| PP2 native | Prefill | 78,608.53 ± 176.15 | 78,508.29 | 255.48 | 255.48 | — | 3 |
| PP2 adaptive | Prefill | 74,679.43 ± 8,449.89 | 78,411.35 | 282.19 | 282.19 | — | 5 |
| EP2 native | Decode | 863.20 ± 2.23 | 863.03 | 1185.41 | 176.24 | 7.946 | 3 |
| PP2 native | Decode | 666.79 ± 1.10 | 667.14 | 1532.34 | 112.97 | 11.176 | 3 |
| PP2 adaptive | Decode | 805.67 ± 10.62 | 807.51 | 1268.33 | 113.80 | 9.091 | 5 |

最终 adaptive Prefill 的五轮吞吐为 **78,511、78,522、78,389、78,411、59,564**。最后一轮 cohort 为 550.13 ms，另外四轮约 417–418 ms。全部保留，均值相对原生 PP 下降 5.00%；中位数 **78,411.35 vs 78,508.29**。慢轮原因未确定，不能据此宣称 Prefill 完全无退化。Prefill 的实际 T4096 Expert 调用仍走原生路径。

[完整指标及逐请求中位数/P99](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/full_metrics.csv) · [每轮结果](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/rounds.csv) · [逐请求计时](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/per_request.csv)

![End-to-end and components](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/end_to_end_and_components.png)

## 3. Expert 问题真实存在，Attention 不是本轮提速来源

以下是实际 Decode CUDA Graph 中每卡、每次实际层调用的平均 GPU kernel 时间，单位微秒；两 rank 各选择 16 个稳定的本地 B4 forward，丢弃前 4 个。Attention 仅包含 core，Expert 仅包含 routed GEMM，不包含全部投影、共享专家与路由。

| 配置 | Attention core | Routed Expert GEMM | Dispatch | Combine |
|---|---:|---:|---:|---:|
| EP2 native | 13.07 | 59.56 | 15.36 | 17.95 |
| PP2 native | 12.50 | 94.19 | 0.00 | 0.00 |
| PP2 adaptive | 12.33 | 57.62 | 0.00 | 0.00 |

PP Expert **94.19→57.62 µs**，接近 EP 的 59.56 µs；PP Attention **12.50→12.33 µs**。这支持“先优化 Expert 执行”的方向。上述是独立 kernel scopes，不能相加成请求延迟；PP 没有逐层 Dispatch/Combine，但仍有 stage 边界与反馈通信。stage0 的 NCCL 驻留从 3.475→2.323 ms/所选 forward，包含等待和 overlap，不能直接当作网络传输时间或与 Expert 收益相加。

Prefill 独立全模块 profile 中，原生 EP/PP 的 Expert 累计约 119.69/175.02 ms/卡，Attention 约 126.55/125.16 ms/卡。EP 的 Dispatch+Combine 累计约 165.59 ms/卡。PP 虽有较慢的 Expert，仍通过省去逐层跨卡交换在该 Prefill 负载获益。Prefill 模块累计计时与 Decode core 计时范围不同，不混成同一占比图。

## 4. 等有效计算量对照：不活跃 Expert 维度也会增加成本

真实 PP 的每层有 256 个 resident experts。路由记录显示，本地 Decode T4 只有平均 **11.1 个活跃 Expert**（8–14），总共 32 次 token–Expert assignment；T4096 Prefill 平均活跃 193.15 个（123–230）。这些活跃数与本轮提示组成有关。

原生 masked 预处理为每个本地 Expert 分配 `m_max=(T//256+1)*256` 的 M 容量。因此 T4 仍使用 M 容量256。它表明布局容量较大，不能据此声称所有 padding 对应的 GEMM FLOPs 都实际执行了。

更强的控制实验固定同一真实层输入、所选权重、路由、有效计算量，使用同一个128-Expert权重 bank 的相同地址前缀，仅改变传给原生 kernel 的本地 Expert 维度；全局 num_experts=256 保持不变，expected-M tuning hint 保持1。Expert-only CUDA Graph 时间：

| 真实层来自 | E32 | E64 | E128 | 原始 E256 参考 |
|---|---:|---:|---:|---:|
| 0 | 47.16 | 56.48 | 70.15 | 100.30 |
| 1 | 60.54 | 67.16 | 80.65 | 104.73 |

E32/64/128 是严格共享权重地址控制，输出与 DeepGEMM 参考完全相同；原始 E256 使用原模型权重 bank，仅作参考。权重收集准备不计入此内核实验，正式优化不做这种权重重排或复制。

这说明 PP 的 Expert 劣势不全是不可避免的有效计算量差异，执行布局对不活跃 Expert 的处理本身就能产生额外成本。

![Fragmentation mechanism](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/fragmentation_mechanism.png)

## 5. 实现和几轮优化

| 轮次 | 方法 | 观察与取舍 |
|---|---|---|
| r1 | 原生 EP2/PP2、真实路由与同输入 kernel 对照 | PP 小形状原生约100 µs，已有小 tile Triton约35–42 µs |
| r2 | PP 本地 T≤4 选择 Triton，较大输入走原生 DeepGEMM | Decode 795.65 vs r1 PP 674.90 tok/s；后续以重新测量的原生参考确认 |
| r3 | 调整 M/N/K tile；同时测 compact DeepGEMM 与新 route-indexed BF16 GEMV | M16N64K128 单层略快，但完整模型没有稳定进一步收益；该轮后续 logprob 传输失败，整轮 Guard 标为 failed，计时保留为探索数据，不进入最终主结果 |
| r4 | 修复仅验证阶段的 PP logprob 元数据传输，重测 EP/PP；增加等有效工作量的 resident 维度控制 | 获得最终原生参考与因果证据 |
| r5 | 回到默认 small-tile Triton，5 轮确认 | Decode 805.67 tok/s；Expert GEMM −38.83%；Prefill 慢轮全部保留 |
| r6/r7 | 原生/优化各3轮原子批量概率验证 | 去除客户端到达分组差异；数值差异重复出现，见下节；没有计入性能结果 |

同实际 T4 输入的候选 kernel 对照，stage0/stage1 微秒：masked DeepGEMM 98.30/102.52；compact DeepGEMM 86.27/93.15；新 route GEMV 43.07/47.07；默认 small-tile 34.66/42.35；调 tile 后33.06/41.48。route GEMV 避免 token-row padding 并融合 up+SwiGLU，但仍慢于小 tile Tensor Core 路径，因此没有选入最终方案。只减少 padding 并不足以保证最快。

最终策略在 [fragmentation.py](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/fragmentation.py)：只作用于 PP2，按实际本地 T≤4 切换现有 BF16 Triton runner。没有改路由专家数、权重精度、Attention、流水槽位，也没有跨微批次等待合批。每个形状在 Graph capture 时确定路径。更大的输入继续原生 DeepGEMM。

## 6. 数值验证和结果适用范围

单层真实输入对照通过有限值和相对 L2 阈值1.5%的实验检查；Triton 相对 DeepGEMM 约 **0.27%–0.33% L2**，新 GEMV约0.23%–0.27%，compact DeepGEMM在这些输入上完全相同。这不是完整生成的质量评估。

最初并发单请求 API 的原生重复也受到到达分组影响，因此另测原子 batch API：一次提交8条输入，每条生成8个 token，连续3轮。原生 PP 自身重复 **64/64位置完全一致，top10 logprob 差为0**。优化版三个重复也一致，但相对原生仅有 **55/64位置仍处于完全相同的生成历史**；其中下一 token 相同 **53/55（96.36%）**，top10 平均重合9.65，交集 token 的平均绝对 logprob 差 **0.08916**，最大 **0.74750**。后续分歧轨迹的位置没有被当作同条件概率比较。

因此不能称为逐 token 等价或已验证无损。完整128-token性能运行也可能沿不同轨迹执行；相同实际输入的单层对照才是隔离 kernel 成因的直接证据。本轮证明了性能机会，生产采用仍需任务质量评估和数值容差标准。

[固定批次概率比较](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/analysis/probability_validation_batched.json) · [位置级记录](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/analysis/probability_positions_batched.csv) · [原并发验证保留](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/analysis/probability_validation.json)

r3 的验证失败来自该归档引擎将嵌套 CUDA logprob Tensor 作为 Python pickle 传回前一 PP stage；接收时在对方已满 GPU 创建第二个 CUDA context，导致 OOM。私有 adapter 只在未计时的 V 验证阶段把嵌套元数据转到 CPU，顶层 Tensor 仍保留原生 typed transport；正式性能路径不做此 CPU 转换。r4/r5/r6/r7 后续验证与清理均正常结束。

## 7. 对研究问题的影响

最有价值的新发现是：**部署决策中的 Expert 成本需要显式包含 local token 数、resident/active Expert 数和 kernel/layout；仅以有效 FLOPs 或模型总参数预测，会把一部分可优化的实现成本误认为 PP 的固有代价。**

这个负载下优化后 PP 的逐层 Expert 成本已经接近 EP，但完整 Decode 仍较慢，说明下一步还需定位流水依赖、反馈、边界等待和其他算子在关键路径上的作用。当前结果还没有证明新的跨槽位 Expert 合批调度，也不能据此宣称此机制本身的研究新颖性。更大的 batch、更多样路由和跨通信域，属于后续泛化范围。

## 8. 复现与来源

源代码 commit：`870269702285d94bdd926a50cc29ad1a20bcf60c`；不可变 tag：`experiment/sglang/glm47-ep16-canonical-profile-20261004-cpu-validated`。冻结引擎源码当前 clean。模型 revision：`995ad96eacd98c81ed38be0c5b274b04031597b0`。DeepEP wrapper：`/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.third_party/deepep-pcie-coalesced-20261006`；native：`/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/runtime/deepep_h100_custom_ep8_20261007_r2/native12/python/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so`，SHA256 `d9b26ac86748517800ce9f9734d865be20d089d473d9d4f6d66dde98c673d76b`。[通信库哈希及实际环境](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/communication_provenance.json)。全部启动参数、每个私有 adapter 哈希、具体环境和 GPU 映射保存在各 run 的 launch_manifest/config；本报告与快照不复制授权文件、权重或 KV payload。

[最终优化运行来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/runs/r5_adaptive_confirm/launch_manifest.json) · [原生参考来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/runs/r4_reference_fixed/launch_manifest.json) · [数值/计时汇总 CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/analysis/metrics.csv) · [GPU分项 CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/analysis/components.csv) · [同输入内核 CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/analysis/microbench.csv) · [路由记录 CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/analysis/routing.csv)

离线重算，不启动 GPU：

```bash
python3 experiments/h100_pp_expert_fragmentation_20261026/analyze.py
python3 experiments/h100_pp_expert_fragmentation_20261026/validate_probabilities.py --batched
/home/bxb1/conda/bin/python experiments/h100_pp_expert_fragmentation_20261026/plot.py
python3 experiments/h100_pp_expert_fragmentation_20261026/finalize.py
```

新 GPU 复现先重新核对 booking、Guard 节点及 CPU 权重 bank，使用新的 run 目录。`prepare.py --mode native --arms ep2 pp2` 为参考；`--mode triton_small --arms pp2 --repeats 5` 为最终策略，再交由 `launch.py --run <new-run>` 执行 Guard。现有 run 的冻结代码不修改。

共执行7次 Guard，另有1个只准备未启动的目录。6次实验完整结束，r3 后续验证失败但清理正常。全部自有模型进程退出、7次租约均释放，后台工作可恢复；没有修改共享引擎默认或 gateway。[逐轮清理、释放和冻结哈希审计](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/run_audit.json)。

本次小型归档：[experiment-manager snapshot](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_expert_fragmentation_20261026/manifest.json)；原始大 trace 和日志仍在本实验 runs 目录，快照通过路径与哈希关联。


后续离线分析：[NVIDIA GEMM原则下的Expert理想模型与PP2端到端估算](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/HEADROOM_ESTIMATE.md)；其中目标数据为估算，不是新GPU测量。
