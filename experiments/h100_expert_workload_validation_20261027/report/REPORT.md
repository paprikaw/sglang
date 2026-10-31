# H100：Expert 小 tile 优化能否推广到其他 workload？

**可以推广到本轮测到的每卡16 token。** PP2在每卡4/8/16 token重复文档上的吞吐分别提升18.2%、15.4%、11.9%；16 token多主题输入提升11.7%。同一优化也提升EP2，双方各自选择较快候选后，本轮四组均为EP2吞吐更高。

随着batch扩大，PP2相对收益下降，但Expert内核改善仍存在：每卡16 token重复文档下，PP2 GEMM由138.8降至100.4µs/层。保持16 token改成多主题后，PP原生GEMM升至212.5µs，优化后仍为170.7µs。因此仅用token数描述workload不足；实际路由和Expert执行形状也是profiler应测的特征。

本轮暂停 TBO 适配，只验证此前公平应用于 EP2、PP2 的 Expert 优化。没有启用此前 PP lookahead/feedback 调度原型。

## 实验范围

完整 Qwen3.6-35B-A3B BF16，同一节点 spartan-gpgpu178、同一对 H100 NVLink GPU0/1；整节点 Guard 租约、两 GPU 实际运行。EP2 为 Attention DP2×TP1、MoE EP2；PP2 为两个20层 stage、两个原生流水槽位。全局 B8/B16/B32 对应目标每卡 Decode 输入4/8/16 token，PP 的 microbatch 使用原生空闲期 API 设为 B/2。

前三组保留此前重复文档组成；第四组保持 B32，改成32个不同主题。所有请求均有不同的首 token，避免请求之间共享前缀。每条 Decode 请求使用真实 GPU 缓存4096 token，再输入1 token、生成128 token；temperature0、ignore EOS。独立冷 Prefill 为每条4096新输入→1输出。每配置、每 workload、每阶段5个正式 cohort，保留全部轮次。

原来的已归档策略只在本地 T≤4 启用。本轮为检验该内核的适用范围，将双方的门限同样扩到 T≤16；内核实现、形状对应的默认 tile 选择、BF16精度、路由和 DeepEP dispatch/combine 均沿用原实现。原始 T≤4 策略在 T8/T16 会回退到 native，本轮不是宣称原策略自动覆盖了大形状。EP 原生 shared-expert overlap 保持开启，TBO 关闭。

## Decode 端到端结果

吞吐为全部请求生成的 output token / cohort makespan，均值±样本SD；TPOT和E2E为请求均值，单位ms。TPOT按首token到末token除以127计算。

| Workload | 配置 | Native output tok/s ↑ | Small-tile output tok/s ↑ | 吞吐变化 | TPOT ms ↓ | E2E ms ↓ |
|---|---|---:|---:|---:|---:|---:|
| b8_repeated | EP2 | 858.9 ± 6.1 | 886.6 ± 3.1 | +3.23% | 7.958 → 7.676 | 1191.5 → 1154.2 |
| b8_repeated | PP2 | 671.9 ± 3.2 | 794.2 ± 8.8 | +18.20% | 11.099 → 9.236 | 1520.7 → 1286.6 |
| b16_repeated | EP2 | 1477.2 ± 8.1 | 1549.5 ± 1.6 | +4.90% | 9.335 → 8.882 | 1384.8 → 1320.1 |
| b16_repeated | PP2 | 1125.0 ± 3.3 | 1297.9 ± 6.7 | +15.37% | 13.296 → 11.380 | 1815.7 → 1573.7 |
| b32_repeated | EP2 | 2486.8 ± 3.9 | 2574.2 ± 3.2 | +3.51% | 11.357 → 10.950 | 1643.7 → 1587.7 |
| b32_repeated | PP2 | 1892.0 ± 8.0 | 2117.6 ± 3.8 | +11.92% | 15.930 → 14.122 | 2157.8 → 1927.5 |
| b32_diverse | EP2 | 2186.0 ± 2.6 | 2282.5 ± 2.3 | +4.42% | 13.141 → 12.522 | 1870.2 → 1791.1 |
| b32_diverse | PP2 | 1733.2 ± 5.3 | 1936.4 ± 6.1 | +11.73% | 17.486 → 15.528 | 2355.9 → 2108.4 |

PP2 在重复文档的每卡4/8/16 token三档吞吐变化分别为 +18.20% / +15.37% / +11.92%；16 token多主题为 +11.73%。EP2 对应为 +3.23% / +4.90% / +3.51% / +4.42%。

![End-to-end and Expert](workload_generalization.png)

双方都给相同内核选择机会后，各自保留本轮平均吞吐更高的实现；若EP的native更快，比较时保留native。下面是两种已测试候选的比较，不代表穷尽调优的上限。

| Workload | EP2较快候选 / output tok/s | PP2较快候选 / output tok/s | PP / EP |
|---|---:|---:|---:|
| b8_repeated | triton_small / 886.6 | triton_small / 794.2 | 0.896× |
| b16_repeated | triton_small / 1549.5 | triton_small / 1297.9 | 0.838× |
| b32_repeated | triton_small / 2574.2 | triton_small / 2117.6 | 0.823× |
| b32_diverse | triton_small / 2282.5 | triton_small / 1936.4 | 0.848× |

## 真实路由和 Expert 时间

活跃专家数来自独立 eager 诊断中的真实路由，取每层第一个匹配目标本地输入数的调用，不是全部128步的平均，也不是预设32或20。该诊断包括cached单token Extend与Decode。Assignments是到达该卡的token–expert配对。GEMM时间来自另一轮原生 CUDA Graph profile，每rank跳过前4个稳定Decode forward，再选16个；两rank均值，单位µs/GPU/实际层调用。它不含全部路由、激活、索引和流水等待，不能直接相加代替E2E。

| Workload | 配置 | Active experts native→opt | Assignments native→opt | Expert GEMM µs native→opt ↓ | GEMM降幅 |
|---|---|---:|---:|---:|---:|
| b8_repeated | EP2 | 10.86 → 10.88 | 32.0 → 32.0 | 59.49 → 47.61 | +19.96% |
| b8_repeated | PP2 | 10.38 → 11.47 | 32.0 → 32.0 | 93.88 → 63.83 | +32.01% |
| b16_repeated | EP2 | 11.75 → 11.78 | 64.0 → 64.0 | 62.89 → 51.07 | +18.80% |
| b16_repeated | PP2 | 12.70 → 12.78 | 64.0 → 64.0 | 108.98 → 74.62 | +31.52% |
| b32_repeated | EP2 | 15.82 → 15.82 | 128.0 → 128.0 | 87.08 → 75.26 | +13.57% |
| b32_repeated | PP2 | 13.40 → 13.40 | 128.0 → 128.0 | 138.76 → 100.44 | +27.61% |
| b32_diverse | EP2 | 28.57 → 28.59 | 128.0 → 128.0 | 148.00 → 133.23 | +9.99% |
| b32_diverse | PP2 | 14.75 → 14.88 | 128.0 → 128.0 | 212.53 → 170.69 | +19.68% |

![Actual routing](routing_geometry.png)

## Prefill 对照

Cold Prefill大形状继续走native DeepGEMM；小 tile 策略本身不预期改善该部分。吞吐为新增input token/s，E2E单位ms，所有波动轮次均保留。

| Workload | 配置 | Native input tok/s ↑ | Small-tile input tok/s ↑ | E2E ms native→opt ↓ |
|---|---|---:|---:|---:|
| b8_repeated | EP2 | 66795.9 ± 7830.4 | 67763.5 ± 8162.1 | 384.0 → 378.4 |
| b8_repeated | PP2 | 78366.4 ± 418.1 | 74076.1 ± 10070.3 | 256.8 → 289.0 |
| b16_repeated | EP2 | 71963.4 ± 4376.5 | 71021.7 ± 3340.5 | 608.6 → 614.2 |
| b16_repeated | PP2 | 80439.0 ± 4318.5 | 82624.9 ± 150.9 | 466.4 → 444.0 |
| b32_repeated | EP2 | 70904.3 ± 2441.5 | 73686.7 ± 93.4 | 1100.2 → 1046.5 |
| b32_repeated | PP2 | 84950.0 ± 136.1 | 85289.7 ± 111.4 | 819.5 → 816.0 |
| b32_diverse | EP2 | 72311.9 ± 98.3 | 72802.1 ± 153.8 | 1067.2 → 1058.5 |
| b32_diverse | PP2 | 84045.7 ± 1293.0 | 85041.4 ± 62.2 | 832.1 → 816.9 |

## 检查与可复现材料

正式计时不含模型初始化、JIT、图捕获、prefix准备、路由诊断、数值比较、概率验证或profiler。所有正式请求逐条核验精确缓存数及输出长度。每rank使用真实权重检验T4/T8/T16的EP索引覆盖、block边界、空专家及全部专家活跃；实际路由另做EP层输出对照与PP两stage的相同输入 native/Triton 对照，实验相对L2门限1.5%。独立8-token top-10概率验证仅在相同生成历史处比较；没有声明长生成逐token一致或任务质量等价。

引擎继续使用冻结commit `870269702285d94bdd926a50cc29ad1a20bcf60c` 和原不可变tag；模型revision `995ad96eacd98c81ed38be0c5b274b04031597b0`。运行代码、模型/通信来源、Guard/cache/cleanup检查和全部冻结输入哈希随manifest保存。不存在兼容的Qwen混合GDN CPU-KV checkpoint，沿用真实模型产生的GPU radix前缀与GDN状态。

逐配置、逐轮、逐请求、各rank分项、真实路由及概率对照分别保存在 `analysis/metrics.csv`、`cohorts.csv`、`requests.csv`、`components_per_rank.csv`、`routing.csv`、`probability_positions.csv`。`comparisons.csv`为本页逐行对照。大profile与原始SSE仍在各run原路径，归档保留哈希引用。
