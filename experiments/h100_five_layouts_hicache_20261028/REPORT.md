# 35B：五种四卡部署，开启 HiCache

Qwen3.6-35B-A3B；每种部署合计 4×H100 80GB、节点内 NVLink。全局 64 请求，缓存前缀 P=8,256 或 130,304；Prefill 每请求新增 Q=512、输出 1 token，Decode 新增 Q=1、输出 128 tokens。每项 3 轮正式样本；吞吐越高越好。

所有部署启用原生 GPU/CPU HiCache，write_through/kernel/page_first；每张物理 GPU 的 CPU 池为 64 decimal GB Attention KV + 8 decimal GB GDN。GPU 前缀主动移出只发生在受控 CPU 重放开始前，传回 GPU 的时间包含在吞吐与延迟内。CPU 写入 fence 与 GPU 主动移出位于 cohort 计时外，因此这里是每批请求吞吐，不是包含批间准备间隔的连续服务吞吐。没有使用磁盘 L3 或 CPU checkpoint replay。

## 自然前缀重放，P=8,256

| 部署 | Prefill 新 tokens/s | Decode tokens/s | Decode TTFT 秒 | Decode TPOT ms | Prefill CPU 前缀比例 | Decode CPU 前缀比例 |
|---|---:|---:|---:|---:|---:|---:|
| EP4 | 85,767 | 5,474 | 0.252 | 9.75 | 0.0% | 0.0% |
| 2×EP2 | 99,446 | 4,961 | 0.207 | 11.21 | 0.0% | 0.0% |
| 4×EP1 | 123,670 | 5,373 | 0.144 | 10.66 | 0.0% | 0.0% |
| 2×PP2 | 105,066 | 4,148 | 0.133 | 14.30 | 0.0% | 0.0% |
| PP4 | 89,664 | 3,276 | 0.154 | 18.40 | 0.0% | 0.0% |

## 自然前缀重放，P=130,304

| 部署 | Prefill 新 tokens/s | Decode tokens/s | Decode TTFT 秒 | Decode TPOT ms | Prefill CPU 前缀比例 | Decode CPU 前缀比例 |
|---|---:|---:|---:|---:|---:|---:|
| EP4 | 33,360 | 2,292 | 0.559 | 23.65 | 0.0% | 0.0% |
| 2×EP2 | 17,406 | 1,553 | 1.430 | 17.98 | 53.1% | 57.8% |
| 4×EP1 | 14,336 | 636 | 5.876 | 5.26 | 99.0% | 99.5% |
| 2×PP2 | 33,966 | 1,537 | 0.980 | 21.73 | 44.3% | 49.0% |
| PP4 | 29,117 | 1,658 | 0.505 | 34.68 | 0.0% | 0.0% |

## 受控 CPU 前缀重放，P=8,256

| 部署 | Prefill 新 tokens/s | Decode tokens/s | Decode TTFT 秒 | Decode TPOT ms | Prefill CPU 前缀比例 | Decode CPU 前缀比例 |
|---|---:|---:|---:|---:|---:|---:|
| EP4 | 76,189 | 5,195 | 0.329 | 9.77 | 100.0% | 100.0% |
| 2×EP2 | 83,683 | 4,642 | 0.268 | 11.28 | 100.0% | 100.0% |
| 4×EP1 | 96,451 | 4,998 | 0.210 | 10.75 | 100.0% | 100.0% |
| 2×PP2 | 103,500 | 4,433 | 0.165 | 12.74 | 100.0% | 100.0% |
| PP4 | 92,745 | 3,828 | 0.166 | 15.47 | 100.0% | 100.0% |

## 受控 CPU 前缀重放，P=130,304

| 部署 | Prefill 新 tokens/s | Decode tokens/s | Decode TTFT 秒 | Decode TPOT ms | Prefill CPU 前缀比例 | Decode CPU 前缀比例 |
|---|---:|---:|---:|---:|---:|---:|
| EP4 | 12,432 | 1,230 | 2.262 | 22.94 | 100.0% | 100.0% |
| 2×EP2 | 14,126 | 1,327 | 2.105 | 17.78 | 100.0% | 100.0% |
| 4×EP1 | 14,379 | 636 | 5.876 | 5.26 | 100.0% | 100.0% |
| 2×PP2 | 24,827 | 1,405 | 1.268 | 22.96 | 100.0% | 100.0% |
| PP4 | 22,607 | 1,669 | 1.008 | 30.54 | 100.0% | 100.0% |

## 解释与范围

CPU 前缀比例为实际 native cached_tokens_details.host / (B×P)，其余命中来自 GPU；总命中和每轮明细见 full_metrics.csv。它不是 CPU 请求命中率或逐 rank 传输字节比例，CPU 命中请求数另存于 CSV。不同缓存模式使用不同生成分支与缓存历史，两模式的吞吐差不能单独归因于CPU传输。自然重放沿用该布局的原生缓存状态；这批合成请求不是生产 trace，命中率不能直接外推为生产正常命中率。

PP2 按 Prefill/Decode、前缀长度、缓存模式，分别从 native/early-send 与 microbatch 16/8/4 的完整组合中选择有效配置；PP4 使用支持四阶段的 native/early-send 与 microbatch 16/8/4 候选。tail-local feedback 在长上下文压力下触发 slot 复用错误，已排除。PP2 lookahead 的固定消息形状路径不适用本次实际 batch；探索性的动态形状改造也未通过输出校验，均未用于正式比较，依赖该路径的 policy10 尚未取得本负载资格。每个候选两轮 pilot，选择样本排除在正式三轮之外；Decode 候选要求输出与本布局 native control 相同。这里的“最优”指本次已测试的候选，不代表全参数空间最优。PP2 的层划分为20+20，PP4为10+10+10+10；本轮没有搜索其他层划分。

正式比较的五种布局全部在同一178节点完成，171节点的探索性候选单独留档且未混入正式结果。PP 使用 static memory fraction 0.92，预留原生 GDN 状态复制和通信工作空间；EP 为 0.94。PP Decode Graph 最大捕获 batch 为实际 microbatch 上限16。旧的未开启 HiCache 数据另存，不能与本表混成一个性能排名。

GPU KV token 池是本配置分配值；大上下文 P=130,304 为此前 EP4 容量附近的固定负载，本轮不会将其称为所有 HiCache 配置下新的绝对最大容量。缓存恢复、排队、重算都计入服务吞吐，Expert kernel 效率不能由端到端比例单独推断。

正式数据的 token 数、输出长度、吞吐、均值延迟均从逐请求样本独立重算；前缀 hashes 跨布局一致。Guard 后检、自有模型清理、租约释放和冻结输入 SHA256 均已通过。

## 原始数据

- summary.json：完整指标、容量、拓扑与 PP 选择。
- full_metrics.csv：每轮吞吐与实际 GPU/CPU 命中。
- validation.json：独立核验与稳定文件读入 SHA256 收据。
- [ep_178_r4](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_five_layouts_hicache_20261028/runs/ep_178_r4)：冻结配置、完整原始样本、调优 pilots、rank 证据、日志与 Guard 收尾。
- [ep_178_pp2_r1](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_five_layouts_hicache_20261028/runs/ep_178_pp2_r1)：冻结配置、完整原始样本、调优 pilots、rank 证据、日志与 Guard 收尾。
- [ep_178_pp4_r1](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_five_layouts_hicache_20261028/runs/ep_178_pp4_r1)：冻结配置、完整原始样本、调优 pilots、rank 证据、日志与 Guard 收尾。
