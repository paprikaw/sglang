# PP／EP 优化回归 Benchmark v1

整理日期：2026-10-27。用途：每轮优化后，用同一组模型、GPU、部署和输入检查收益及回退。[返回实验总览](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/README.md)。协议定义、历史结果和后续补测分别保留来源；本清单生成脚本只做离线整理。

## 1. 固定模型和部署

每行分别在 **A100、H100** 上运行；同一行的 EP／PP 使用相同 GPU 数、同一卡组和相同全局请求数。

| 模型档位 | 模型及权重格式 | 固定资源 | EP 配置 | PP 配置 |
|---|---|---|---|---|
| 小 | Qwen3.6-35B-A3B，BF16 | 2 GPU，NVLink pair | MoE EP2；Attention DP2×TP1 | PP2×EP1；每 stage 1 GPU、20 层 |
| 中 | Qwen3.5-122B-A10B-FP8 | 2 GPU，NVLink pair | MoE EP2；Attention DP2×TP1 | PP2×EP1；每 stage 1 GPU、24 层 |
| 大 | GLM-4.7-AWQ，INT4 | 8 GPU，2 节点×4 GPU | MoE EP8；Attention DP2×TP4 | PP2×EP4；每 stage 1 节点、Attention TP4、46 层 |

PP 均固定两个槽位，每槽最多 B/2 个请求；不把 PP stage 数、槽位数和 Attention TP 混用。GLM 的 PP stage 内仍有 EP4 Dispatch／Combine，不能记为零。以上是固定资源比较，不是最低部署卡数或所有 degree 的穷举。

跨模型保留既有 checkpoint 精度；记录每层专家权重字节数、专家数、top-k 和实际活跃专家数。A100 的 FP8 checkpoint 可能走 W8A16／Marlin，H100 走 W8A8／DeepGEMM，需记录真实路径；GPU 间结果用于平台部署比较。

## 2. 固定 11 个 workload

`B` 是全局请求数；`P` 是每请求计时开始前已有的 GPU 缓存；`Q` 是每请求新增输入；`O` 是输出 token 数。相同模型的优化前后、EP／PP 使用相同 token IDs、请求分组和输入哈希。

| 阶段 | B | P | Q | O | 输入组成 | 点数 |
|---|---|---:|---|---:|---|---:|
| Decode | 8、16、32 | 4096 | 1 | 128 | 同源文档，复用现有 repeated 生成方法 | 3 |
| Decode | 32 | 4096 | 1 | 128 | 不同主题，复用现有 diverse 生成方法 | 1 |
| Prefill | 8、32 | 0 | 128、512、4096，完整交叉 | 1 | 同源文档 | 6 |
| Prefill | 32 | 0 | 4096 | 1 | 不同主题 | 1 |

每个 Attention 实例的 Prefill token budget 固定 4096；记录实际 chunk 和每 rank 的源 token 数，不能把 B×Q 当成单次 Dispatch 输入。Decode 使用真实模型生成的 GPU-hot KV／GDN 状态；前缀准备不计入 Decode，不能用冷输入128→输出512代替。

完整矩阵：**3 模型 × 2 GPU 类型 × 2 部署 × 11 workload = 132 个组合／版本**。优化前后用同一矩阵比较；一次只有 B8 的结果属于定点验证。统一 workload 在 A100／122B／GLM 上的兼容性、容量及优化适配仍需逐项验证，不能把未运行当通过。

## 3. 每个点保存什么

| 数据 | 必需内容 |
|---|---|
| Prefill 端到端 | 整批 input tok/s、逐请求 TTFT／E2E、整批 makespan |
| Decode 端到端 | 整批 output tok/s、逐请求 TPOT、TTFT、E2E；不能仅靠吞吐判断稳态 Decode |
| Expert | routed Expert GEMM 时间；本地 token 数、活跃专家数、token-expert assignments |
| 通信 | Dispatch 和 Combine 分开；PP activation 交接、反馈和 stage 时间另列；EP1 的 MoE 交换不存在，记 0；缺失测量标未测 |
| Attention | 同阶段相同口径；尽量同时保留 core 和含 projection 的 module，避免混用 |
| 统计及复现 | 每次 3 轮预热、5 轮 profiler-off 正式计时，保存逐轮／逐请求数据、均值和标准差；profile 独立采集 |

单层分项统一标成 µs／卡／实际层调用，并保留层调用次数、rank 和时间窗口。Prefill 与 Decode 的后端、Graph、shared-expert overlap／TBO、真实网络及 native binary 哈希必须记录。通信 kernel 时间可能有等待／重叠，不能直接拼成 E2E。

通用 kernel 优化在 EP 和 PP 同时启用；PP 调度优化额外比较“PP 原调度／PP 新调度”，EP 保留其已验证优化。新旧版本按相反顺序交替运行。先验证同布局优化前后输出，再独立记录 EP／PP 数值与任务质量；改变内核时不能只凭同模型名判断等价。

## 4. 最新优化已经覆盖哪里

| 模型 | A100 | H100 |
|---|---|---|
| 35B BF16 | 有历史 EP2／PP2，最新优化待适配／重测 | **11 workload 的两种部署均有计时：22／22** |
| 122B FP8 | 有历史数据，最新优化待适配／重测 | 有历史数据，最新优化待适配／重测 |
| GLM INT4，8 GPU | 有历史数据；统一 GPU-hot workload 待测 | 有历史数据；统一 GPU-hot workload 待测 |

按计时覆盖统计，当前 **22／132 已有最新参考结果，110／132 待验证**。35B 的 4 个 Q128／Q512 Prefill workload（共 8 个部署组合）仍有 EP／PP 数值差异：有性能数据，但不能标成质量验证通过。

H100 当前参考：双方相同 T≤16 Expert selector；Decode 的 PP 使用 policy9 lookahead；Prefill 仍以 PP 原生 policy0 为默认参考。policy1 early-send 的全部结果另存，因未见稳定收益，不把它视为默认优化。

@@A100_SECTION@@

### 最新 H100 35B Decode

GPU-cached P4096＋Q1→O128；每配置 5 轮。下表吞吐包含首 token 的影响，TPOT 单独保留。

@@DECODE_TABLE@@

B32 时 PP 吞吐接近 EP，而 TPOT 仍更高；benchmark 会同时检查这两个指标，不把两者混为一项收益。

### 最新 H100 35B Prefill

P0→Q→O1；每配置 5 轮。下表为同一主矩阵中的 PP native，不混入独立复测。完整 TTFT／E2E 和 early-send 消融在结果 CSV 中。

@@PREFILL_TABLE@@

B8/Q4096 主矩阵有波动；独立 10 轮复测 EP／PP native 为 64,788.8／77,814.0 input tok/s，early-send 为 77,926.9。该组单列，不能与主矩阵拼成新均值。Q4096 跨布局只验证了首 token；短输入数值问题须继续保留为未解决。

## 5. 下次完整验证的入口

- [机器可读协议](benchmark/suite_v1.json)：模型、部署、11 个 workload、计时规则。
- [132 个组合及逐点状态](benchmark/cases.csv)、[六组覆盖汇总](benchmark/coverage.csv)。
- [最新 H10035 全部 33 行指标](benchmark/latest_h10035_results.csv)：包含 EP、PP 原生和 PP 调度消融；其中 22 行对应主矩阵。
- [最新 H10035 Expert／Dispatch／Combine／Attention 分项](benchmark/latest_h10035_components.csv)、[11 组 EP／PP 对照](benchmark/latest_h10035_comparisons.csv)。Decode 的 Attention 为 core，Prefill 为含 projection 的 module，CSV 明确区分。
- [来源与 SHA256](benchmark/sources.json)、[离线核验](benchmark/validation.json)、[重新生成清单的脚本](benchmark/build_benchmark.py)。脚本只整理已有结果，不启动 GPU；原始逐请求、trace 和 PP 边界时间从来源实验的 analysis/requests.csv、profile_cycles.csv 进入。

执行时为每轮优化创建独立 run ID／source tag，保存优化前后版本及 workload 哈希，按 Experiment Guard 和 CPU-KV checkpoint 规范选择与验证资源。跑完后每个组合记录 measured／failed／unsupported／pending；OOM 或不支持须保留原因，不静默缩小 B、Q 或 P。新增配置或改变输入长度时另起协议版本。
