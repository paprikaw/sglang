# AgentX：数据、HiCache 与 P/D 动态配置验证

2026-10-30。研究对象仍是 **EP 与 EP=1（单卡／纯PP）之间的选择**；首先验证真实 P→D 交接的价值，再检验动态配比的增量。

**本轮结论：数据已下载，HiCache合并部署基线有效；P/D比较卡在GDN Decode前缀缓存支持，尚未得到P/D或动态配比收益。**

- 官方256k数据有393个会话、68,266次调用；完整原始数据和下载校验均已保存。
- 两节点共完成4个HiCache基线点、每点3轮，共480次请求；后续轮次平均复用48,486个GPU缓存token。
- 当前引擎拒绝GDN/Mamba模型开启P/D Decode radix缓存，EP2和PP2均在启动时失败，P/D有效性能点为0。没有关闭Decode缓存绕过要求。
- 已核验本实验进程清理；节点已交还现有backfill。下一步先处理缓存兼容性，再比较P/D和动态配比。

## 已取得的数据

官方完整版与256k版均已下载到 [数据目录](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/datasets/agentx_20261030)，保留全部原始 JSONL、数据卡和统计文件。下载总表包含文件大小、固定仓库 revision 和 SHA256；大文件均与 Hugging Face LFS SHA256 核对。另保存官方 harness 源码供协议审计，没有安装到共享推理环境。

| 数据集 | 会话 | 实际模型调用 | 平均输入 token | 平均输出 token | 理想保留时平均新输入 token |
|---|---:|---:|---:|---:|---:|
| 完整版 | 393 | 98,827 | 218,922 | 1,077 | 3,773 |
| 256k版 | 393 | 68,266 | 100,947 | 860 | 3,949 |

**表中都是原始文件重新统计的算术平均。** 新输入估算按会话内时间排序，只匹配在当前请求开始前已完成请求的输入块，假设不驱逐、不迁移、没有GDN状态端点限制；不是实际缓存命中率，也不是本机GPU测量。256k版这一估算下的 token 加权前缀复用约96.09%。

256k版有 175 个会话包含显式子agent，1,697 个子agent组，39,822 次嵌套调用。原始内容已脱敏为64-token块哈希；重建的合成输入可保留前缀关系和长度，但不能还原真实语义与专家路由分布。

完整版原始文件是98,827次模型调用，与方法页所写135,282不一致；本报告采用文件实数，并保留该差异。256k版68,266与数据卡一致。

[官方完整版](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126) · [官方256k版](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k) · [方法与限制](https://inferencex.semianalysis.com/agentx/methodology) · [官方回放器](https://github.com/SemiAnalysisAI/agentx-harness)

## 对研究假设的含义

这份数据适合做验证，但数据长度的变化本身不能证明动态配置收益。长context增加Attention与缓存压力；缓存命中后，新Prefill量可能只有几千token。P/D分离节省的干扰必须抵得过状态传输、缓存副本与固定P:D资源比例的成本。

Qwen35B原生context上限262,144，先使用官方256k版；不把完整版长请求静默截短。35B的10个全Attention层、2个KV头、head_dim256、BF16，对应每个上下文token约20KiB的KV（不含GDN）。256k版平均上下文的完整KV约1.93GiB/请求；如果两端都有有效前缀，只传增量可显著改变这笔开销。此处是模型尺寸估算，不是实测传输量。

**HiCache是后续主实验的必需配置。** GPU保留原生可用容量；CPU承接溢出。所有部署保持相同CPU预算，维持会话与实例/DP rank的亲和性，分别记录GPU命中、CPU命中/回载以及P→D传输。HiCache开启不等于发生过CPU回载。

## 当前四卡资格检查

模型为Qwen3.6-35B-A3B，权重/KV为BF16，GDN状态为FP32；固定引擎commit为`0f2f1990994774472b1089d2b0682f0bf926a6b4`。每台节点4张H100 80GB，节点内NVLink。EP2采用Attention DP2（每请求TP1）、MoE EP2；PP2采用TP1/EP1、层间PP2。

两台现有H100各运行一组独立、同节点、等四卡的对照：011为2×EP2合并服务对EP2 Prefill→EP2 Decode；171为2×PP2合并服务对PP2 Prefill→PP2 Decode。计划通过Mooncake实际交接KV/GDN状态；本轮Decode启动失败，尚未执行交接。跨节点结果不直接相减。

用户要求HiCache后，初次GPU-only配置已中止并排除。HiCache轮次每卡32 decimal GB host pool、每节点128GB（Guard保守预约128GiB），GPU static fraction0.95，不设人为max-total-tokens上限；PD Decode额外打开radix前缀缓存。

协议要求先检查P/D对照的串行贪心输出一致性，再运行小规模AgentX派生片段；本轮仅完成合并服务基线，P/D一致性检查未能执行。片段在GPU测量前选定：20个合格会话中，按新增输入/输出比均匀选8个，各保留连续5轮的原长度、块前缀和轮间等待；输入1K–64K、输出16–512。C1/C8各三轮，每轮开头重置以固定起点，会话内部不清缓存。首轮请求保留在全片段wall time中；另列后四轮请求平均TTFT/TPOT/E2E。

**这是兼容性与有限范围筛查，不是完整AgentX分数。** 不涵盖子agent DAG、长于64K的尾部、跨节点迁移、高并发饱和或动态比例控制；PP先用原生调度。不能据此宣布哪一类布局在AgentX上全面更快。

| Run | 状态 | 完整性能点 | 证据 |
|---|---|---:|---|
| 011_ep2_hicache_r2 | 合并基线完成；PD缺Mooncake，整轮失败 | 2 | [运行目录](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_feasibility_20261030/runs/011_ep2_hicache_r2) |
| 011_ep2_hicache_r3 | dependency_isolation_replaced；排除 | 0 | [运行目录](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_feasibility_20261030/runs/011_ep2_hicache_r3) |
| 011_ep2_hicache_r4 | PD Decode不支持GDN前缀缓存，失败 | 0 | [运行目录](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_feasibility_20261030/runs/011_ep2_hicache_r4) |
| 011_ep2_r1 | cancelled_by_user_protocol_correction；排除 | 0 | [运行目录](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_feasibility_20261030/runs/011_ep2_r1) |
| 171_pp2_hicache_r2 | 合并基线完成；PD缺Mooncake，整轮失败 | 2 | [运行目录](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_feasibility_20261030/runs/171_pp2_hicache_r2) |
| 171_pp2_hicache_r3 | 仅准备、未启动；排除 | 0 | [运行目录](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_feasibility_20261030/runs/171_pp2_hicache_r3) |
| 171_pp2_hicache_r4 | PD Decode不支持GDN前缀缓存，失败 | 0 | [运行目录](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_feasibility_20261030/runs/171_pp2_hicache_r4) |
| 171_pp2_r1 | cancelled_by_user_protocol_correction；排除 | 0 | [运行目录](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_feasibility_20261030/runs/171_pp2_r1) |

合并服务基线：后续请求的三轮算术均值（绝对ms），C是并发会话数，每格96个后续请求；首轮primer不计入本表：

| Run | 部署 | 并发会话 | TTFT | TPOT | E2E |
|---|---|---|---:|---:|---:|
| 011_ep2_hicache_r2 | colocated | c1 | 134.38 | 5.99 | 1448.01 |
| 011_ep2_hicache_r2 | colocated | c8 | 163.84 | 7.64 | 1826.73 |
| 171_pp2_hicache_r2 | colocated | c1 | 111.56 | 5.88 | 1400.88 |
| 171_pp2_hicache_r2 | colocated | c8 | 109.42 | 8.93 | 2045.97 |

**这些是两组独立节点的基线，不能用相邻行推导EP/PP反转或动态收益。** 共12轮、480次正式请求，其中384次为后续轮次。每个性能点后续请求GPU命中共4,654,656 token，平均48,486；host/storage命中均为0，未验证CPU回载。

原始Guard整轮postcheck保持failed：r2因随后PD未加载成功而缺PD权重加载日志，r4未完成预定性能点。已完成的合并服务部分另行核对冻结输入SHA256、4个副本的全部预期rank及canonical CPU权重来源、每条输出长度、三轮均值和清理回执；通过的是这部分独立核查，不是整个P/D对照。

[逐项审计和最终状态](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_feasibility_20261030/final_findings.json) · [011基线独立审计](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_feasibility_20261030/runs/011_ep2_hicache_r2/analysis_audit.json) · [171基线独立审计](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_feasibility_20261030/runs/171_pp2_hicache_r2/analysis_audit.json) · [完整均值和轮次](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_pd_feasibility_20261030/metrics.csv)

## 当前阻塞：GDN的P/D Decode前缀缓存

补齐并隔离既有Mooncake依赖后，两种布局均到达模型加载阶段，随后Decode在构建缓存时报同一个明确错误：

```text
--disaggregation-decode-enable-radix-cache is incompatible with Mamba/SSM models
```

这项限制同时存在于[固定版本源码](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/qwen35b-decode-ep-overlap-20261027/python/sglang/srt/mem_cache/kv_cache_builder.py:295)和[工作区源码](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/python/sglang/srt/mem_cache/kv_cache_builder.py:295)。[EP2失败日志](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_feasibility_20261030/runs/011_ep2_hicache_r4/cache/pd.r1.log)与[PP2失败日志](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_feasibility_20261030/runs/171_pp2_hicache_r4/cache/pd.r1.log)均已保存。

**这是当前实现的兼容性限制，不是P/D没有收益的证据，也不能推出GDN在原理上无法做P/D。** 合并部署的HiCache已运行；P/D Decode缓存还需要处理GDN recurrent state与前缀端点、状态生命周期/拷贝、恢复和交接一致性。不能只删除报错检查。

因此目前三个问题的答案分别是：数据可获得且适合构造实验；保留缓存条件下P/D是否有用仍未知；动态EP/EP1配比的额外收益仍未知。

## 缓存兼容后，两节点八卡的决定性实验

| 要回答的问题 | 必须比较 | 通过条件 |
|---|---|---|
| P/D分离是否有用 | 最佳同预算合并部署 vs 最佳固定P/D部署 | 同到达/会话集合和SLO下改善达标吞吐或会话完成时间，计入缓存与传输 |
| EP/EP1是否值得同时保留 | 固定同构P/D vs 固定异构P/D，并测同一阶段低/高活跃量 | 至少两个工况有可重复的相反赢家，超过测量波动 |
| 是否需要动态配比 | 最佳固定布局/比例、只调P:D比例、再加EP/EP1比例控制 | 对变化负载有额外收益，包含迁移、重配、合批缩水和缓存损失 |

首轮八卡候选保留2×EP4、4×EP2、4×PP2，以及容量允许时的8×单卡。P/D先固定每节点4卡，在通过状态交接资格检查的EP4、2×PP2和单卡副本组合中选择，再缩小比例搜索。当前[PP映射源码](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/qwen35b-decode-ep-overlap-20261027/python/sglang/srt/disaggregation/common/conn.py:712)要求Decode PP与Prefill PP相等或Decode PP=1；因此EP4→PP2不能直接上线，PP2→EP4也仍需验证不同rank与GDN状态映射。动态策略必须和最佳固定组合、只调P:D的策略比较；固定混合池的请求路由不能直接叫GPU比例重配置。

正式回放使用官方harness的依赖图、子agent并发与工具等待；低/中/高会话并发以及上升/下降负载段使用同一预定种子与会话集合。短跑只作资格检查，最终保留完整官方时长口径或明确标为派生实验。闭环快系统会走到不同请求，需同时报告固定工作量的会话完成时间，避免只比较不同工作集的tokens/s。

建议下一阶段预先固定工程筛查门槛：额外达标吞吐约15%以上或会话完成时间约10%以上、重复实验稳定，才值得继续实现复杂动态机制；这是投入判断标准，不是统计显著性或已获得收益。若只有P/D有效而动态增量很小，应收窄到固定异构P/D；若P/D无收益，保留合并部署作为可选动作。

## 可复现入口

- [冻结协议](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_feasibility_20261030/PROTOCOL.md)
- [数据下载校验](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/datasets/agentx_20261030/manifest.json)
- [完整数据统计脚本](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_feasibility_20261030/analyze_dataset.py)
- [所选片段及原始索引](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_feasibility_20261030/screening_windows.json)
- [统一实验索引](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/README.md)
