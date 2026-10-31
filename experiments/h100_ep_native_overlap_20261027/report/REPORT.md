# EP2：SGLang 原生通信/计算 overlap 复测

当前冻结版本的 Qwen 专用 shared-expert/DeepEP overlap 默认开启。此前约7.66ms的 EP2 基线已包含这一路；本次显式关闭/开启并核验实际 stream 分支。

同一 H100178 NVLink GPU0/1；Qwen3.6-35B-A3B BF16；AttentionDP2/TP1，MoEEP2；同一通用小批次 Triton GEMM。全局B8，每请求真实GPU-cached4096+new1→output128；五个正式 profiler-off cohort/配置。前缀构建、warmup、stream审计和独立profile不计入正式计时。

| 配置 | TPOT ms/token ↓ | 五轮均值范围 ms | 输出 tok/s ↑ | 实际 TBO subbatch1 |
|---|---:|---:|---:|---|
| Shared OFF | 8.413 | 8.398–8.439 | 824.4 | False |
| Shared ON | 7.664 | 7.657–7.676 | 889.8 | False |

原生 shared overlap 开启后，TPOT 降低 **8.9%**，吞吐提高 **7.9%**。

![Native overlap measurements](native_ep_overlap.png)

## 实际重叠及组件

原生 Qwen 路径将独立的 shared-expert MLP 放到 alt stream，主 stream 执行 routed experts 的 dispatch、GEMM、combine；最后通过事件依赖合并。相同 token 的 dispatch→routed GEMM→combine 依赖仍保留。两 rank 的 B4 Graph 捕获记录分别确认：OFF 的 shared expert 在主 stream，ON 在独立 stream。审计只记录首次 warmup/capture，不增加正式 Graph replay 的 Python 操作。

| 配置 | Expert GEMM µs/层 | Dispatch µs/层 | Combine µs/层 | Graph ms | D/C与其他计算并行 µs/层 |
|---|---:|---:|---:|---:|---:|
| Shared OFF | 53.69 | 14.16 | 18.70 | 8.270 | 0.00 |
| Shared ON | 53.19 | 15.12 | 18.62 | 7.587 | 12.15 |

组件与并行区间来自独立 profile：每 rank 丢弃前20个Graph，选后16个。并行时间是 DeepEP dispatch/combine GPU kernel 区间并集与其他计算 kernel 区间并集的交集，包含 shared expert 及其他独立计算，不等于净节省时间；不可直接从正式 TPOT 中相减。

## 通用 SBO / TBO 开关

- `SGLANG_ENABLE_QWEN_DEEPEP_SHARED_OVERLAP=1`：当前模型可用且默认开启，本次已显式测试0/1。
- `--enable-single-batch-overlap`：当前源码在SM90初始化时明确抛出不支持错误，原因指向当前 sgl-deep-gemm wheel；本次没有绕过该检查。
- `--enable-two-batch-overlap`：单独配置测试，是否实际拆分以 dispatcher subbatch0/1 的运行记录为准。当前Qwen3.5/GDN的完整模型forward是逐层循环，没有调用标准TBO模型调度入口。
- Shared ON + TBO flag：未形成有效性能点，阶段 `failed`，错误 `RuntimeError`；保留启动和清理证据。
  首个原生异常发生在 CUDA Graph 初始化的 `TboForwardBatchPreparer.filter_batch`：不支持拆分字段 `mamba_track_indices`, `mamba_track_mask`。日志原路径及SHA256记录在 `results.json`；本次没有修改模型或绕过字段检查。

## 来源与复现

[正式指标](metrics.csv) · [逐请求](formal_requests.csv) · [逐cohort](formal_cohorts.csv) · [逐Graph并行区间](profile_overlap_cycles.csv) · [来源、开关和stream证据](results.json)

所有成功实验均通过冻结输入哈希、真实GPU缓存/输出长度、完整SSE计数、Guard postcheck及owned cleanup校验。模型数学与GEMM未改；EP原生异步调度的输出可能有A/A差异，不将本实验当作任务质量评估。冷Prefill4096→1为每配置三轮基线控制，当前性能结论针对Decode。

- [Shared OFF](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_native_overlap_20261027/runs/native_shared_off_r1)：[冻结配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_native_overlap_20261027/runs/native_shared_off_r1/config.json)；[启动来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_native_overlap_20261027/runs/native_shared_off_r1/launch_manifest.json)。
- [Shared ON](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_native_overlap_20261027/runs/native_shared_on_r2)：[冻结配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_native_overlap_20261027/runs/native_shared_on_r2/config.json)；[启动来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_native_overlap_20261027/runs/native_shared_on_r2/launch_manifest.json)。
- [Shared ON + TBO flag](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_native_overlap_20261027/runs/native_tbo_on_r3)：[冻结配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_native_overlap_20261027/runs/native_tbo_on_r3/config.json)；[启动来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_native_overlap_20261027/runs/native_tbo_on_r3/launch_manifest.json)。
