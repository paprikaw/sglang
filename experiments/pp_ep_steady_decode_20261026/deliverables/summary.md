# PP / EP 部署决策：本轮补测与汇报材料

完成了 Qwen3.6-35B-A3B BF16 在同一对 A100、同一对 H100 上的 EP2 / 两槽 PP2 稳态 Decode 补测，并补齐较长的 Attention / Expert / 通信分项。结合已有模型和 workload 数据，制作了 12 页中文 PPT。结论支持同时比较专家执行和通信成本；尚未建立跨节点阈值或自动预测算法。

- [可编辑 PPT](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/deliverables/PP_EP_deployment_decision_20261026.pptx)
- [PDF 预览](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/deliverables/PP_EP_deployment_decision_20261026.pdf)
- [全部图表与数据](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/deliverables)
- [逐请求数据](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/deliverables/data/requests.csv)、[三轮数据](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/deliverables/data/rounds.csv)、[逐卡分项](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/deliverables/data/components_per_rank.csv)
- [协议](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/deliverables/data/protocol.json)、[来源哈希](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/deliverables/data/source_hashes.json)、[每页讲稿与来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/deliverables/speaker_notes.md)

## 本轮原生端到端结果

各配置 3 轮，每轮同时发出 8 请求。每条真实 GPU 已缓存 P=4096，本轮输入 Q=1，生成 128；总旧缓存 32768、本轮新输入 8、每轮输出 1024 token。相同语义提示词；无 CPU KV 换入、无 synthetic KV。前缀准备、权重加载、JIT、预热和 profile 均不计入正式计时。吞吐越高越好，TPOT / TTFT 越低越好。

| 平台 | 布局 | 输出 token/s，均值 ± 三轮标准差 | 三轮范围 | 平均 TPOT ms | 平均 TTFT ms |
|---|---|---:|---:|---:|---:|
| A100 | EP2 | 91.40 ± 0.88 | 90.38–91.95 | 86.258 | 248.65 |
| A100 | PP2 | 123.10 ± 0.58 | 122.44–123.56 | 64.271 | 139.91 |
| H100 | EP2 | 858.86 ± 5.20 | 855.46–864.84 | 7.962 | 180.34 |
| H100 | PP2 | 668.65 ± 0.81 | 667.73–669.23 | 11.149 | 112.11 |

A100 上 PP 的 TPOT 降低 **25.5%**；H100 上 EP 的 TPOT 降低 **28.6%**。H100 此负载下 PP 的 TTFT 较短、EP 的持续生成 TPOT 和吞吐更好，因此仍需按部署目标选择。三轮不能代表长期稳定性或生产尾延迟；本轮保留了固定 EP 后 PP 的顺序。

A100 保留 Triton Expert / Triton Attention、Graph 关、overlap 关及 DeepEP normal BF16；H100 保留 BF16 DeepGEMM、FA3 / Triton、原生 Decode Graph 和 EP overlap，PP overlap 不支持。这是各平台原生配置的对比，不能归因为只更换 GPU。

## Attention / Expert / 通信对照

下面单位均为 **ms / 卡 / 一次实际层调用**。EP 每卡 40 层，PP 每卡 20 层；均筛选实际本地 batch=4 的 Decode。每配置 3 个独立 profile，每个 profile 两个 rank，各丢弃最初 4 个 Decode forward 后取 24 个完整 forward。共 36 份 rank trace、864 个 rank forward、25920 次实际层调用。误差条按三份独立 profile 的卡均值计算；不是把同一 trace 的重复层当作独立样本。

| 平台与执行范围 | 布局 | Attention | Routed Expert / GEMM | Dispatch | Combine |
|---|---|---:|---:|---:|---:|
| A100 native | EP2 | 0.104289 | 0.098284 | 0.070400 | 0.040195 |
| A100 native | PP2 | 0.103435 | 0.105473 | 0.000000 | 0.000000 |
| H100 native | EP2 | 0.013019 | 0.060818 | 0.015275 | 0.017886 |
| H100 native | PP2 | 0.012522 | 0.093013 | 0.000000 | 0.000000 |
| H100 eager_diagnostic | EP2 | 0.057625 | 0.072559 | 0.407722 | 0.117511 |
| H100 eager_diagnostic | PP2 | 0.056504 | 0.115225 | 0.000000 | 0.000000 |

**统计范围必须区分：** A100 native 与 H100 eager diagnostic 的 Attention 含投影、Expert 为模块；H100 native 使用实际 Graph replay，只识别 Attention 核心和 Routed GEMM，其余投影 / Shared Expert / routing 等留在 other。H100 eager 诊断使用同一批加载权重、单独走已有 eager fallback，验证没有整模型 Graph replay；不得把它的绝对时间当成原生 Graph 模块耗时。完整表另外保留 Shared Expert、other 和标准差。

A100 的 Attention 变化 **-0.8%**，Routed Expert **+7.3%**；后者 PP 三份 profile 标准差较大，应表述为接近、略增。EP 逐层 Dispatch + Combine 明显，PP 免去这部分专家交换而最终 TPOT 更低。H100 原生 Graph 的 Attention 核心变化 **-3.8%**，Routed GEMM **+52.9%**；eager 模块诊断分别 **-1.9%** / **+58.8%**，支持关注专家执行效率的变化，但并未隔离工作集、padding 和路由复用各自的因果贡献。

PP 仍有阶段边界通信。A100 native / H100 native 的 NCCL 驻留约 **0.037 / 1.745 ms / 卡 / 所选 stage forward**。该指标来自选定 CPU 窗口内 NCCL GPU 活动的时间并集，包含等待并可能重叠计算，不能解释为纯 NVLink 载荷时延，不能与上表每层数值相加。GPU 分项本身也可能重叠，不是独占 TPOT 百分比分解。

## 已有三维证据如何配合本轮结果

历史统一 moderate 对照中，PP 相对 EP 的三轮均值吞吐变化如下；两卡、8 请求，Prefill P=0 / Q=4096 / O=1；Decode P=4096 / Q=1 / O=128。历史运行用各自原生配置；没有和本轮请求记录混成一个平均值。

| 模型 | 平台 | 阶段 | PP 相对 EP |
|---|---|---|---:|
| 35B BF16 | A100 | prefill | +15.18% |
| 35B BF16 | A100 | decode | +34.09% |
| 35B BF16 | H100 | prefill | +14.15% |
| 35B BF16 | H100 | decode | -25.08% |
| 122B FP8 | A100 | prefill | -5.60% |
| 122B FP8 | A100 | decode | +34.92% |
| 122B FP8 | H100 | prefill | +6.39% |
| 122B FP8 | H100 | decode | -19.92% |

H100 冷 Prefill 中，35B / 122B 的 PP 额外 Routed + Shared Expert 活动约 **66.1 / 117.8 ms/卡**，而 EP Dispatch + Combine 约 **132.7 / 129.0 ms/卡**；Attention 在两布局间接近。这是“本地专家代价会吃掉通信收益”的机制线索。35B BF16 与 122B FP8 的架构、精度、后端不同，加上小模型 Prefill 的三轮波动，不能推出越大的模型必然越偏 EP。已有 Q07 有更细的跨模型审查。

H100 35B 的 workload 例子：8 请求中等负载 PP Prefill / Decode 吞吐 +14.2% / −25.1%；16×179968 GPU 缓存、各新增 512 的长 Prefill为 +11.7%，对应 Decode −18.6%；32×89984 缓存、各新增 512 时为 −4.5% / −18.0%。后两点旧缓存总量同为 2879488，但请求构成和总新增输入不同，不是单变量 batch 扫描。长上下文 EP 实际本地 batch8 Graph 中，Attention 核心约 10.095 ms、Routed GEMM 2.076 ms，因此 Attention 的绝对成本不能忽略。

A100 122B Prefill 还存在 Attention 干扰反例：两槽 PP Attention 比 EP 约高 11.2%；同层同形状 Marlin 投影与 NCCL 高度重叠时 0.7718→1.2679 ms，而 Full Attention 核心 5.0535→5.0619 ms。支持把通信驻留、等待及调度干扰作为独立考虑项；这仍是相关性证据，没有移除通信的因果 A/B，也未测精确 SM 利用率。

## 给部署决策的简洁用法

1. 用权重 + KV + 运行缓冲筛掉不可行布局，并标明实际 EP 通信域和 PP 边界。
2. 对候选测 Expert 工作集 / 输入复用对应的执行变化、Dispatch / Combine、阶段边界和等待；Attention 保留为对照，并检查长上下文与通信干扰。
3. 在目标 workload 上短测原生 TTFT / TPOT / 吞吐，按 SLO 确认。通信节省大且 Expert 惩罚小倾向 PP；Expert 惩罚或流水依赖更大倾向 EP。判断针对未被重叠隐藏的关键路径成本，不能直接相加独立内核时间。

这份材料是有测量支持的候选筛选方法，尚不是已校准的自动部署预测器。本轮只涉及节点内 NVLink 双卡，不提供跨节点 EP / PP 的切换阈值，也没有补造一、二、四卡全容量网格。

## 核验、失败记录和复现

最终采用 a100_r2 / h100_r2；两者原始 Guard postcheck、canonical CPU 权重缓存审查、拥有进程清理均通过，GPU 资源已释放。两平台完整冻结引擎 commit为 870269702285d94bdd926a50cc29ad1a20bcf60c；额外运行适配器逐文件哈希位于各 launch_manifest。A100 NV12 与 H100 NV6 是各节点 nvidia-smi 观测标签，并非实测带宽。

原始失败不覆盖：a100_r1 在 CPU 准备阶段失败，没有 GPU 测量；h100_r1 首轮 EP TTFT 有明显异常且 eager 开关钩在未使用的 facade 上，保留原始记录。修正为实际 DecodeCudaGraphRunner 的资格检查后，H100 r2 扩展本地 1/2/3/4 形状预热，并完整重测两个布局各三轮；没有只剔除旧轮中的慢样本。最终 eager trace 均确认无 Graph replay。A100 先前短 profile 的 Expert 方向与本轮较长共同提示词采样不同，旧数据保留；本轮不把两者合并为稳定的普适趋势。

运行前查询的 CPU KV 注册表没有该 Qwen 混合模型的兼容项，使用真实 GPU 原生前缀；没有导出、搬运或伪造不兼容 GDN checkpoint。现有后台资源管理器正常，本轮没有执行被自动审批拒绝的独立 backfill_trigger 动作。

PPT 文字和形状可编辑，图表另存 PNG / 矢量 PDF，数据为 CSV / JSON。已核验 12 页、所有对象边界和文字高度，并逐页检查布局预览；当前环境无 PowerPoint / LibreOffice，PDF 由同一布局的高分辨率页面生成，未声称通过 Office 实机渲染。PPT 使用 Noto Sans CJK SC；接收端字体回退可能影响排版，PDF 可直接展示。原始大 trace、权重、KV、admin key、私有 config 和 lease token 均留在原运行位置，不进入分享包。

原始补测目录：[/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/pp_ep_steady_decode_20261026/runs)。轻量快照：[/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_a100_h100_pp_ep_steady_decode_20261026](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_a100_h100_pp_ep_steady_decode_20261026)。每页 notes 和 sources.json 指向图表的原始来源。
