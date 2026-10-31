# 35B：两卡预算下的2×EP1、EP2与PP2

两卡候选应同时包括2×EP1（两个独立单卡实例）、EP2与PP2。所有数据来自本次Qwen3.6-35B-A3B BF16 / H10080GB实验，全部开启原生HiCache。

这是既有四卡、64请求实验的离线拆分：固定GPU对[0,1]与[2,3]各承担32请求；2×EP1的每个EP1承担16请求，EP2/PP2各实例承担32请求。对每个GPU对使用自己的最早请求开始至最晚请求完成重算吞吐，再对两个GPU对、正式三轮取均值。未简单除二，未挑选最快GPU对。EP2与2×EP1的每GPU对请求组成相同；PP2采用原实验轮转分配，完整64请求组成相同。

两个GPU对同时运行，共享节点CPU/内存；这是两卡组的观测值，不能代替独占两卡、承接全部64请求或固定到达率的实测。Prefill每请求新增Q512、输出1；Decode新增Q1、输出128。P为每请求缓存前缀。吞吐单位tokens/s，越高越好；Prefill只计新增输入，Decode计生成输出。CPU→GPU恢复与排队包含在请求时间中，批前准备不计入。

## 自然缓存重放

| P | 阶段 | 2×EP1 | EP2 | PP2 | CPU前缀token比例：2×EP1 / EP2 / PP2 |
|---|---|---:|---:|---:|---|
| 8,256 | prefill | 62,396 | 50,149 | 52,948 | 0.0% / 0.0% / 0.0% |
| 8,256 | decode | 2,702 | 2,505 | 2,094 | 0.0% / 0.0% / 0.0% |
| 130,304 | prefill | 8,921 | 11,868 | 17,286 | 99.0% / 53.1% / 44.3% |
| 130,304 | decode | 328 | 803 | 780 | 99.5% / 57.8% / 49.0% |

## 受控重放：前缀token全部从CPU命中

| P | 阶段 | 2×EP1 | EP2 | PP2 | CPU前缀token比例：2×EP1 / EP2 / PP2 |
|---|---|---:|---:|---:|---|
| 8,256 | prefill | 51,707 | 44,574 | 51,959 | 100.0% / 100.0% / 100.0% |
| 8,256 | decode | 2,556 | 2,405 | 2,303 | 100.0% / 100.0% / 100.0% |
| 130,304 | prefill | 8,950 | 8,220 | 12,706 | 100.0% / 100.0% / 100.0% |
| 130,304 | decode | 328 | 715 | 710 | 100.0% / 100.0% / 100.0% |

## GPU KV容量与策略含义

| 两卡布局 | GPU KV token池 |
|---|---:|
| 2×EP1 | 320,384 |
| EP2 | 3,199,872 |
| PP2 | 3,502,912 |

每GPU对短前缀总计264,192 tokens，长前缀总计4,169,728 tokens。短前缀自然重放全部命中GPU，2×EP1的Prefill与Decode均高于另两者；该条件下复制单卡实例是应保留的扩容候选。长前缀下，2×EP1约99%的前缀token从CPU命中，Decode吞吐明显下降；Prefill优先PP2，Decode中EP2与PP2接近。受控全CPU重放下，短前缀Prefill的2×EP1和PP2均约5.2万tokens/s，差异很小；长前缀Prefill仍以PP2更高。因此不能只按模型能否单卡放下选择拓扑，还需看KV工作集、缓存层、并发和阶段。

这补充并修正此前EP2/PP2二选一的适用范围：短前缀、GPU缓存充足时，不应默认一拿到第二张卡就重组为EP2或PP2。具体原地切换、KV/GDN迁移、在途请求以及缩容正确性和成本尚未实测；这些静态结果用于提出策略候选。

PP2使用本轮通过资格的原生/early-send与microbatch调优，lookahead/tail-local未取得本负载资格。EP2为Attention TP1/DP2、Expert EP2；PP2为Attention TP1、Expert EP1、20+20层；2×EP1各自Attention TP1/DP1、Expert EP1。每GPU CPU池64 decimal GB KV +8 decimal GB GDN；EP/PP static memory fraction为0.94/0.92。合成重复前缀不能代替生产命中率，受控CPU重放与自然重放的历史和生成分支不同，表间差异不能单独归因H2D。

## 核验与来源

30项布局/上下文/缓存/阶段指标，156个GPU对-轮次记录。正式热重放每项3轮，另有冷Prefill每项1轮保存在JSON/CSV中。核对请求顺序与endpoint映射、数量、输入输出token、计时、HiCache字段和原始SHA256；EP2/PP2指标与上一份两卡拆分一致。两个并发GPU对不是独立重复实验，未建立统计置信区间。

[逐GPU对逐轮数据](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_five_layouts_hicache_20261028/two_gpu_three_layouts_analysis/per_group_rounds.csv) · [指标、容量与来源哈希](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_five_layouts_hicache_20261028/two_gpu_three_layouts_analysis/summary.json) · [可复现分析脚本](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_five_layouts_hicache_20261028/two_gpu_three_layouts_analysis/analyze.py) · [原四卡报告](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_five_layouts_hicache_20261028/REPORT.md)
