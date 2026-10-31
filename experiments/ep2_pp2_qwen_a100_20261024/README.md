# 双卡 EP2 与 PP2：Qwen3.5-122B-A10B-FP8

## 实验边界

完整 48 层文本模型；不截层、不用虚拟专家替代模型。序列化权重约 118.42 GiB，单张 A100 80GB 放不下，双卡能够部署。两个配置固定同一节点、同一对 NVLink GPU0／1，顺序运行。

| 配置 | Attention | Routed Expert | 每 GPU 层数 | 每个已分配层的本地专家数 | GPU 间通信 |
|---|---|---|---:|---:|---|
| EP2 | TP1 × DP2 | EP2，专家不再做 TP | 48 | 128 | 每层 DeepEP normal Dispatch／Combine |
| PP2 | 每 stage TP1 | 每 stage EP1 | 24 | 256 | stage 边界传输、反馈和调度；没有远端专家 Dispatch／Combine |

模型包含 36 层 GatedDeltaNet 线性 Attention、12 层 Full Attention。Expert 是每层 256 个 routed experts、top-8，另有 shared expert。

两组同用 W8A16 Marlin：保留 E4M3 block-FP8 权重，BF16 激活和 KV。A100 不使用原生 FP8 Tensor Core。CUDA Graph 和 overlap schedule 均关闭。此轮测 DeepEP normal 原生 NVLink 路径，不代表之前 CE、PCIe 或 RDMA low-latency 路径。

EP2 的启动 chunk 参数为 8192，原生 DP2 解析后每卡为 4096；PP2 的参数为 4096。比较的是相同的实际本地 chunk，不是相同但会被不同方式解释的命令行数字。

## 相同的请求负载

| 场景 | 请求数 | 每条已缓存上下文 | 每条本轮新增输入 | 每条输出 | 计时边界 |
|---|---:|---:|---:|---:|---|
| Prefill | 8 | 0 token | 4096 token | 1 token | 客户端请求放行至首个输出 token |
| Decode | 8 | 4096 token | 1 token | 128 token | 请求放行至输出；另报首 token 后的 TPOT |

EP2 显式固定请求到两个 Attention DP rank，每个 4 请求。最终 PP2 使用 `pp_async_batch_depth=0`、每个 microbatch 最多 4 请求：原生循环槽数为 PP2＋额外深度0＝2。旧 r7 的 `pp_async_batch_depth=2` 实际为四槽，不能称为两个 slot。实际 shape 以原始 trace 的 `step[...]` 标注为准，不能仅凭参数声称流水已经打满。

Decode 上下文在本次进程内用真实模型计算、保存在原生 GPU prefix cache。不是合成 KV，不计前缀构建时间，不涉及 CPU HiCache 换入。本轮容量充足，用于隔离 EP/PP 执行差别，不能证明 KV 容量不足时谁更好。

## 测量与校验

1. Guard 全节点登记、监督租约，使用已有预约；实际只用 2 张卡。
2. 预载并复用 canonical CPU full-weight bank。
3. 新增 Marlin／DeepEP-normal 数据接口，在完整权重加载后，对真实一层执行固定输入、变化输入和跨卡路由验证。独立 all-gather／本地专家／all-reduce 参考只用于数值校验，不参与性能测量。
4. 预热后各测 3 轮，不启用 profiler；输出 TTFT、TPOT、端到端延迟的均值、中位数和 P99，以及 cohort 吞吐。
5. 另采 CPU＋GPU profiler。GPU kernel 通过 correlation ID 对应真实 CPU launch，再归属到 Attention、Expert、Dispatch、Combine。CPU 范围时间不当作 GPU 计算或网卡耗时。
6. 每个原始 trace 保存逐层、逐次调用数据；不同 forward mode、microbatch shape 分开。GPU 分项可重叠，不把相加值冒充关键路径。

## 文件入口

- `prepare.py`：冻结脚本、模型和原生库，生成 Guard 合同；不启动 GPU。
- `runner.py`：Guard controller、节点 worker、真实模型服务器及客户端计时。
- `runtime_adapter.py`：两组共享 Marlin 计算、EP2 DeepEP-normal 数据接口和数值校验。
- `analyze.py`：对原始 trace 做逐层 GPU 归因。
- `test_protocol.py`：CPU 回归测试；不代替 GPU 数值校验。
- `runs/`：不可变的每轮源码快照、配置、日志、测量和清理证据。

首次 `paired_r1` 在新验证代码调用关键字接口时失败，没有有效性能结果；实际 GPU 清理通过。首次 harness 的最终 progress／cleanup 写入顺序不满足 Guard，失败记录保留。后续 run 修正接口和记录顺序，未修改旧记录或放宽 Guard。

后续适配还修正了原生 cache-flush 的纯文本应答、旧 DeepEPMoE 容器绕过 quant-method 的计算分支，以及零接收 token 不能启动 CUDA Expert 归约的问题。数值校验直接经过真实 EP 模块 forward，加入所有 token 只发向一个 rank 的用例；空 rank 仍参与 Dispatch／Combine，只跳过不存在的 Expert 计算。PP 的原生 TP1 日志另补实际 TP0 身份到原始 loader 记录，不生成替代的 cache-hit 记录。

## 最终结果入口

[完整报告与统计表](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen_a100_20261024/results/summary.md)。EP2 使用 r7 的有效测量，最终两槽 PP2 使用 r8；各三轮、每轮八请求。Decode：EP2 为 72.08 output tok/s、TPOT 109.66 ms；PP2 为 97.26 output tok/s、TPOT 81.37 ms。Prefill：EP2 为 8670.75 input tok/s，PP2 为 8185.11 input tok/s。

两组共有八份原始 CPU＋GPU trace，完整逐层表见 `results/stage_metrics.csv`。分析只使用 CPU user annotation 定义层调用，不把 Kineto 镜像的 GPU user annotation 再算一次。所有原始失败运行保持不变。

r7 已完成四个测量点、八份 trace，cache 审计及 GPU 清理均通过；退出包装器误捕获 `SystemExit(0)`，所以原始 Guard 状态仍为失败，数据使用另附独立复算与验证。该错误已修复并有 CPU 回归测试；最终 r8 以 exit0 完成，Guard postcheck、缓存审计、GPU 清理均通过。未把任何旧失败记录改写成成功，也未因此重测 EP2。
