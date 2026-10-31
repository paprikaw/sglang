# A100 NVLink: EP2 and PP2 regression points

B32多样输入的独立profile中，routed Expert为EP 263.3 / PP 320.0 µs，Dispatch+Combine为 99.1 / 0 µs，Attention为 136.4 / 135.8 µs。PP本地Expert更慢、Attention接近、专家交换减少，与本轮端到端PP更快的方向一致；这些GPU分项不直接相加还原TPOT。

已完成6组workload、15个计时点。PP9相对PP原生调度的变化为 decode_b8_repeated: -0.75%；decode_b32_repeated: +0.36%；decode_b32_diverse: +0.73%。具体PP／EP取舍与所有延迟见下表。

完整 Qwen3.6-35B-A3B，BF16 权重／激活／KV、FP32 GDN。节点 spartan-gpgpu069 的物理 GPU0/1，租约内实测为 NV12。EP2 的 Attention 为 DP2×TP1，Expert 为 EP2；PP2 为两个各含20层的 TP1/EP1 stage、两个最多B/2请求的槽位。Decode实际每槽B/2请求，Prefill按每Attention实例4096-token预算分块。两种布局使用相同原生 Triton Expert 家族；A100 DeepEP normal BF16、Triton Attention，CUDA Graph 与 overlap schedule 均关闭。

Decode：每请求真实 GPU 缓存4096 token + 新输入1 token，生成128 token；Prefill：无缓存，新增128或4096 token，生成1 token。B为全局请求数。重复型与32主题多样输入沿用 H100 构造，输入指纹可逐项核验。权重加载、缓存准备、JIT和profile不计入正式性能；每配置3次暖机、5轮正式cohort。PP0为原生调度；PP9为提前发射下一槽的调度优化，Expert算子保持相同。

吞吐按每轮总token数除以整批从首个请求开始至最后输出的时长，再取5轮均值；Decode计输出token，Prefill计本次新输入token。TTFT为客户端首token延迟，TPOT为首末token间隔除以127，E2E为客户端请求完成时间；延迟取全部正式请求均值。固定请求组的入队等待计入延迟。

| Workload | EP2 tok/s | PP2 native tok/s | PP2 lookahead tok/s | PP reference / EP |
|---|---:|---:|---:|---:|
| decode_b8_repeated | 91.55 | 123.39 | 122.46 | 1.338× |
| decode_b32_repeated | 328.80 | 451.83 | 453.47 | 1.379× |
| decode_b32_diverse | 326.43 | 451.97 | 455.28 | 1.395× |
| prefill_b8_q128_repeated | 4238.37 | 6882.40 | — | 1.624× |
| prefill_b8_q4096_repeated | 27245.06 | 31337.20 | — | 1.150× |
| prefill_b32_q4096_diverse | 30109.66 | 33654.67 | — | 1.118× |

| Workload / layout | TTFT ms | TPOT ms | E2E ms | Throughput SD |
|---|---:|---:|---:|---:|
| decode_b8_repeated / ep2 p0 | 168.56 | 86.40 | 11141.92 | 0.69 |
| decode_b32_repeated / ep2 p0 | 240.80 | 95.79 | 12406.21 | 2.33 |
| decode_b32_diverse / ep2 p0 | 241.82 | 96.50 | 12496.83 | 1.58 |
| prefill_b8_q128_repeated / ep2 p0 | 182.98 | — | 182.98 | 11.48 |
| prefill_b8_q4096_repeated / ep2 p0 | 730.72 | — | 730.72 | 35.35 |
| prefill_b32_q4096_diverse / ep2 p0 | 2320.74 | — | 2320.74 | 81.31 |
| decode_b8_repeated / pp2 p0 | 115.81 | 64.30 | 8282.23 | 0.41 |
| decode_b8_repeated / pp2 p9 | 114.88 | 64.82 | 8346.41 | 1.63 |
| decode_b32_repeated / pp2 p0 | 162.58 | 69.94 | 9044.81 | 2.65 |
| decode_b32_repeated / pp2 p9 | 160.56 | 69.71 | 9013.35 | 2.09 |
| decode_b32_diverse / pp2 p0 | 160.90 | 69.93 | 9041.81 | 2.09 |
| decode_b32_diverse / pp2 p9 | 161.47 | 69.42 | 8977.41 | 1.17 |
| prefill_b8_q128_repeated / pp2 p0 | 125.47 | — | 125.47 | 98.32 |
| prefill_b8_q4096_repeated / pp2 p0 | 637.95 | — | 637.95 | 68.84 |
| prefill_b32_q4096_diverse / pp2 p0 | 2076.15 | — | 2076.15 | 42.71 |

独立profile将GPU kernel经CPU launch correlation归到实际层和算子；Decode丢弃前20个forward，选之后16个；Prefill覆盖整次冷输入cohort。以下对两rank取均值，单位为每卡每次实际层调用的微秒。Attention含投影及核心；Expert分开routed GEMM和其他路由执行开销，shared Expert另存CSV。PP为EP1，Dispatch/Combine无跨卡专家交换。GPU区间统计不能直接相加还原客户端TPOT。

| Workload / layout | Expert total µs | Expert GEMM µs | Dispatch + Combine µs | Attention µs |
|---|---:|---:|---:|---:|
| decode_b8_repeated / ep2 p0 | 107.14 | 91.39 | 135.82 | 104.40 |
| decode_b32_repeated / ep2 p0 | 181.30 | 163.70 | 102.95 | 136.13 |
| decode_b32_diverse / ep2 p0 | 263.26 | 245.25 | 99.14 | 136.36 |
| prefill_b8_q128_repeated / ep2 p0 | 495.83 | 463.27 | 159.14 | 351.66 |
| prefill_b8_q4096_repeated / ep2 p0 | 2009.15 | 1730.52 | 920.95 | 2705.38 |
| prefill_b32_q4096_diverse / ep2 p0 | 2220.78 | 1912.17 | 961.90 | 2759.02 |
| decode_b8_repeated / pp2 p0 | 109.78 | 101.40 | 0.00 | 103.77 |
| decode_b8_repeated / pp2 p9 | 110.35 | 102.03 | 0.00 | 103.93 |
| decode_b32_repeated / pp2 p0 | 204.39 | 190.93 | 0.00 | 135.75 |
| decode_b32_repeated / pp2 p9 | 203.93 | 190.49 | 0.00 | 135.65 |
| decode_b32_diverse / pp2 p0 | 320.02 | 306.32 | 0.00 | 135.84 |
| decode_b32_diverse / pp2 p9 | 318.91 | 305.20 | 0.00 | 135.45 |
| prefill_b8_q128_repeated / pp2 p0 | 901.63 | 871.70 | 0.00 | 349.36 |
| prefill_b8_q4096_repeated / pp2 p0 | 2241.88 | 2064.60 | 0.00 | 2790.75 |
| prefill_b32_q4096_diverse / pp2 p0 | 2268.70 | 2091.39 | 0.00 | 2830.27 |

同布局PP0/PP9在所有正式轮次核验完整输出序列一致，并核验实际B/2请求组、发送接收payload hash、每轮lookahead确实执行且所有权完全回收。EP/PP的输出一致率单独列出，避免把不同BF16归约顺序造成的生成分歧当作已经通过的质量验证。

| Workload | EP/PP complete request outputs equal |
|---|---:|
| decode_b8_repeated | 3/8 |
| decode_b32_repeated | 9/32 |
| decode_b32_diverse | 5/32 |
| prefill_b8_q128_repeated | 8/8 |
| prefill_b8_q4096_repeated | 8/8 |
| prefill_b32_q4096_diverse | 32/32 |

本轮覆盖35B的6个workload、12个部署组合，另有3个PP原生Decode对照，共15个计时点。A100原生eager/Triton/DeepEP-normal与已有H100 Graph/FA3/low-latency结果属于不同硬件原生执行配置；跨GPU结果可用于部署选择，不能将差值只归因于GPU带宽。其他模型与其余workload仍待补测。

初次公共Guard不支持Qwen而在准入前停止；一次NVLink检查遇到ANSI颜色码，在模型启动前停止。随后较小的GDN状态池在B32暖机触发精确缓存检查，运行已清理；该次已完成的B8数据作为诊断保留，不混入本表。最终采用EP全局168／每rank84、PP每stage164个GDN状态，保留每条路径最终prefix state，避免容量排队或前缀淘汰。最终结果只取此处完整的新运行。

[吞吐图](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/a100_pp_ep_regression_20261027/report/a10035_nvlink_r4/throughput.png) · [Expert／通信／Attention图](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/a100_pp_ep_regression_20261027/report/a10035_nvlink_r4/components.png)

[Metrics CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/a100_pp_ep_regression_20261027/analysis/a10035_nvlink_r4/metrics.csv) · [Components CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/a100_pp_ep_regression_20261027/analysis/a10035_nvlink_r4/components.csv) · [Recomputed result](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/a100_pp_ep_regression_20261027/analysis/a10035_nvlink_r4/result.json) · [Configuration](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/a100_pp_ep_regression_20261027/runs/a10035_nvlink_r4/config.json) · [Topology](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/a100_pp_ep_regression_20261027/runs/a10035_nvlink_r4/topology_qualification.json)
