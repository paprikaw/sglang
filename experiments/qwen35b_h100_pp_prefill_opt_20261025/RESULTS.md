# H100 PP2 Prefill 优化：结果与分阶段证据

2026-10-25；Qwen3.6-35B-A3B BF16；H100 节点 178。这里只报告新输入处理，输出固定为 1 token；不是 Decode／TPOT 优化结果。

## 1. 两项有输出一致性证据的改善

| 明确的请求负载 | 原版 → 候选 | 新输入 token/s ↑ | 整批完成，毫秒 ↓ | 平均 TTFT，毫秒 ↓ |
|---|---|---:|---:|---:|
| 8 请求；每条没有旧缓存，新输入 4096 token | 每次 forward 最多 4096 → 8192 新 token | 77992.5 → **86284.9（+10.63%）** | 420.14 → 379.77 | 257.98 → 254.92 |
| 32 请求；每条 GPU 已缓存 89,984 token，再输入 512 token | 保持每次最多 4096 新 token；数据就绪后再发送 | 28438.3 → **29598.2（+4.08%）** | 576.14 → 553.59 | 403.48 → 389.68 |

第一项是改变原生 Prefill 每次执行的新输入预算，两个 PP stage 同步修改 `chunked_prefill_size` 和 `max_prefill_tokens`；没有减少请求或输入。第二项复用 A100 的发送就绪适配：等待生产端 CUDA 事件完成，再交给原生发送函数发布元数据，避免接收端过早占用 NCCL 执行资源等数据。等待仍包含在测量内。

**适用范围不同：** 长缓存的 8192-token chunk 比原 4096 更慢，不能统一放大；短缓存并发 32 的 8192 也没有改善。没有把以上两项收益相乘，或声称已经得到同配置下的 EP2 最优对照。

## 2. 测量协议和名词

- **已缓存历史 token：** 请求之前的真实模型 KV 与线性 Attention/GDN 状态，正式计时前已经在 GPU。不是再次做相同长度的 Prefill，也不是从 CPU HiCache 换入。
- **本轮新输入：** 正式计时中真正处理的 Prefill token；吞吐只计这些新 token，不把旧前缀重复计入。
- **每次 forward 的新 token 预算：** 调度器一次送入模型的所有请求新输入之和，不是请求数；受实际到达、请求粒度及缓存状态影响，并不保证每次恰好满。
- **TTFT：** 单个请求从发送至收到首个生成 token 的客户端耗时，包含调度、必要同步和通信；整批完成时间从 cohort 提交计时到全部完成。
- **端到端延迟：** 本轮都只生成 1 token，因此等于 TTFT，完整 CSV 同时保留 mean／median／P99；没有可报告的 TPOT。
- **PP2：** GPU2／3，各 20 层，Attention TP1，Expert EP1，原生两个循环槽位；Prefill 仍按预算分块，不是固定将 32 请求分成两个 16 请求一起计算。
- **EP2 参考：** 仅第一轮另用 GPU0／1，Attention TP1×DP2、Expert EP2、各卡完整 40 层；每点两次，波动明显，不作为稳定最优排名。

三个负载为：32 请求各缓存 4096、新输入512；32 请求各缓存89984、新输入512；8请求无缓存、新输入4096。均保留所有 40 层，真实 BF16 DeepGEMM、FA3／Triton；Decode Graph 开、Prefill Graph 关。复用完整 CPU 权重 Bank，但没有把权重 Bank 称为 KV Bank。每次缓存试验独立重建同初始 GPU 前缀，预热与前缀准备不计时，正式样本与 profiler 不重叠。

合批和发送就绪使用同进程原版／候选／候选／原版，每种两次。chunk 扫描使用 4096／2048／1024／8192／8192／1024／2048／4096。正式冷 Prefill 8 批 64 个输出逐请求一致；长缓存发送就绪 4 批128个输出逐请求一致；另保存短缓存所有 chunk 的同输入输出检查。只能支持已测的单 token 输出一致性，不代替长生成质量验证。

## 3. 全部正式指标

数值为每策略两批的均值；TTFT median／P99 来自该策略两批全部请求。小样本 P99 只是这批请求的高位值，不是稳定服务尾延迟。不同轮次各自有基线，不拼接成连续加速比例。

### 第一轮：有上限的输入合批

| 请求数 | 每条已缓存 token | 每条新输入 token | 配置／策略 | 新 token/s ↑ | 整批 ms ↓ | TTFT mean / median / P99，ms ↓ |
|---:|---:|---:|---|---:|---:|---|
| 8 | 0 | 4,096 | EP2，原版 | 73013.4 | 453.50 | 359.89 / 401.35 / 498.93 |
| 32 | 89,984 | 512 | EP2，原版 | 29551.9 | 557.89 | 543.78 / 512.71 / 601.26 |
| 32 | 4,096 | 512 | EP2，原版 | 51192.2 | 320.29 | 313.19 / 310.71 / 328.14 |
| 8 | 0 | 4,096 | PP2，原版 | 77297.5 | 423.92 | 260.83 / 260.87 / 424.93 |
| 8 | 0 | 4,096 | PP2，最多5ms合批 | 76797.8 | 426.68 | 263.30 / 263.19 / 426.98 |
| 32 | 89,984 | 512 | PP2，原版 | 29655.5 | 552.58 | 388.06 / 352.77 / 558.93 |
| 32 | 89,984 | 512 | PP2，最多5ms合批 | 29210.0 | 561.12 | 378.67 / 366.55 / 569.88 |
| 32 | 4,096 | 512 | PP2，原版 | 62249.6 | 263.20 | 190.14 / 176.00 / 262.75 |
| 32 | 4,096 | 512 | PP2，最多5ms合批 | 61767.1 | 265.26 | 171.64 / 178.22 / 265.89 |

### 第二轮：四种原生 chunk 预算

| 请求数 | 每条已缓存 token | 每条新输入 token | 配置／策略 | 新 token/s ↑ | 整批 ms ↓ | TTFT mean / median / P99，ms ↓ |
|---:|---:|---:|---|---:|---:|---|
| 8 | 0 | 4,096 | PP2，4096 新 token／forward | 77992.5 | 420.14 | 257.98 / 257.74 / 420.44 |
| 8 | 0 | 4,096 | PP2，2048 新 token／forward | 44584.8 | 734.96 | 435.11 / 434.96 / 735.87 |
| 8 | 0 | 4,096 | PP2，1024 新 token／forward | 23112.7 | 1417.76 | 818.91 / 819.16 / 1420.78 |
| 8 | 0 | 4,096 | PP2，8192 新 token／forward | 86284.9 | 379.77 | 254.92 / 260.38 / 379.69 |
| 32 | 89,984 | 512 | PP2，4096 新 token／forward | 29278.8 | 559.69 | 399.20 / 372.75 / 567.03 |
| 32 | 89,984 | 512 | PP2，2048 新 token／forward | 27555.7 | 594.75 | 375.05 / 370.44 / 603.28 |
| 32 | 89,984 | 512 | PP2，1024 新 token／forward | 19396.7 | 844.73 | 488.95 / 489.96 / 849.99 |
| 32 | 89,984 | 512 | PP2，8192 新 token／forward | 25477.6 | 650.31 | 475.14 / 478.07 / 718.15 |
| 32 | 4,096 | 512 | PP2，4096 新 token／forward | 61503.1 | 266.40 | 192.27 / 177.06 / 267.36 |
| 32 | 4,096 | 512 | PP2，2048 新 token／forward | 43790.4 | 374.16 | 237.82 / 232.15 / 375.65 |
| 32 | 4,096 | 512 | PP2，1024 新 token／forward | 22602.1 | 724.92 | 406.51 / 406.68 / 727.30 |
| 32 | 4,096 | 512 | PP2，8192 新 token／forward | 60995.5 | 268.77 | 225.05 / 201.39 / 274.60 |

### 第三轮：长缓存发送就绪

| 请求数 | 每条已缓存 token | 每条新输入 token | 配置／策略 | 新 token/s ↑ | 整批 ms ↓ | TTFT mean / median / P99，ms ↓ |
|---:|---:|---:|---|---:|---:|---|
| 32 | 89,984 | 512 | PP2，原版 | 28438.3 | 576.14 | 403.48 / 423.47 / 576.96 |
| 32 | 89,984 | 512 | PP2，数据就绪后发送 | 29598.2 | 553.59 | 389.68 / 360.50 / 557.31 |

## 4. 分阶段归因

以下全部是独立采集的长缓存 32 请求 Prefill，每条 GPU 已缓存89,984、新输入512、输出1。单位为**毫秒／每卡／整批**，不是毫秒／层，也不是请求 TTFT。Attention 含模块内投影；路由和共享 Expert 分开。内核累计可重叠，NCCL 驻留还包括等待，不能相加作为端到端时间。

| 独立采样 | 策略 | stage | forward 次数 | Attention | 路由 Expert | 共享 Expert | NCCL 驻留 |
|---|---|---:|---:|---:|---:|---:|---:|
| 输入合批 | baseline | 0 | 5 | 229.07 | 89.09 | 4.08 | 173.15 |
| 输入合批 | baseline | 1 | 5 | 227.70 | 89.51 | 4.07 | 22.58 |
| 输入合批 | coalesced | 0 | 5 | 223.16 | 86.95 | 3.98 | 148.06 |
| 输入合批 | coalesced | 1 | 5 | 223.37 | 90.49 | 4.04 | 26.00 |
| chunk 对照 | 2048 | 0 | 9 | 227.34 | 112.47 | 4.70 | 30.39 |
| chunk 对照 | 2048 | 1 | 9 | 226.96 | 134.25 | 4.74 | 2.45 |
| chunk 对照 | 4096 | 0 | 5 | 229.50 | 89.14 | 4.09 | 170.56 |
| chunk 对照 | 4096 | 1 | 5 | 227.28 | 90.85 | 4.05 | 26.00 |
| 发送就绪补采 | baseline | 0 | 5 | 223.50 | 87.38 | 3.99 | 146.57 |
| 发送就绪补采 | baseline | 1 | 5 | 222.32 | 88.51 | 3.99 | 20.88 |
| 发送就绪补采 | ready_send | 0 | 5 | 230.05 | 77.74 | 4.09 | 27.96 |
| 发送就绪补采 | ready_send | 1 | 5 | 229.12 | 88.19 | 4.12 | 1.45 |

缩小 chunk 的直接证据：4096 时共5次 forward，新增输入组成为512＋4096＋4096＋4096＋3584；2048时变成9次。Attention 总工作近似不变，路由 Expert 累计耗时上升，尽管 NCCL 驻留下降，实际整批仍更慢。该 profile 支持“更细分块有额外计算／执行开销”，没有测 HBM 读字节，不能据此断言精确的权重重复读取量。

发送就绪补采中，第1阶段NCCL驻留146.57→27.96ms，第2阶段20.88→1.45ms，与减少过早发布发送通知后的GPU等待相符。它不是NVLink载荷传输时间，也不能把减少的累计驻留直接当作端到端收益：正式整批只降低约22.55ms。补采原版首块为1024、新版首块为512，虽然总新token及forward次数相同，仍不能把全部差额严格归为单一NCCL内核因素。

有上限合批只把首块从512改为1024，总forward仍为5，三个负载都未获益，因此不采用。长缓存发送就绪的完整两阶段表来自单独补采 R5；R4最后一份末阶段 trace 没有等待写盘就清理，只保留为不完整诊断，不用它冒充完整因果表。正式计时来自原始 R1／R3／R4，未被 profiler 数字替换。

## 5. 图、源码、验证和限制

![四种 chunk 的吞吐和延迟](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen35b_h100_pp_prefill_opt_20261025/runs/h100_pp_prefill_chunk_r3/analysis/chunk_comparison.png)

![实际分块执行时间线；包含发射间隔](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen35b_h100_pp_prefill_opt_20261025/runs/h100_pp_prefill_chunk_r3/analysis/pipeline_timeline.png)

[全部请求指标 CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen35b_h100_pp_prefill_opt_20261025/analysis/full_metrics.csv) · [逐请求时序 CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen35b_h100_pp_prefill_opt_20261025/analysis/request_metrics.csv) · [逐卡分阶段 CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen35b_h100_pp_prefill_opt_20261025/analysis/stage_metrics.csv) · [原始路径、哈希与验证](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen35b_h100_pp_prefill_opt_20261025/analysis/validation.json)

[发送就绪完整诊断](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen35b_h100_pp_prefill_opt_20261025/runs/h100_pp_prefill_profile_r5/analysis/PROFILE.md) · [chunk 诊断](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen35b_h100_pp_prefill_opt_20261025/runs/h100_pp_prefill_chunk_r3/analysis/PROFILE.md) · [适配与测试源码](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen35b_h100_pp_prefill_opt_20261025)

三轮46个正式cohort、23组指标均通过缓存命中、输出长度与原始 Guard postcheck；补采另有2组完整profile；其控制器错误地按性能文件计为0／2，原始Guard postcheck因此失败，未改写为通过。两阶段导出、缓存及清理证据单独校验通过，不能把该补采称为原始Guard全通过。所有 GPU 进程按拥有者范围清理、租约释放；保留 CPU 权重 Bank。11 项 CPU 测试通过。改动保持在冻结的实验私有适配，不修改共享引擎默认值，也不影响正在使用的服务。

本轮只优化两个已测 Prefill 区间，不代表所有负载最优。特别是长缓存8192策略两批波动较大；没有扫描所有层分配、内核或请求到达模式。没有运行 Decode，不能外推 TPOT；也没有充分重测 EP2 最优参数，因此未声称已消除此前 PP 相对 EP 的全部差距。
