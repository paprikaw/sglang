# 35B：两卡 EP2 与 PP2 的既有数据分析

结论：本轮每两卡实例 B32 的条件下，Prefill 优先 PP2；短前缀 Decode 优先 EP2；长前缀 Decode 的端到端吞吐接近，需要结合 CPU 恢复与首 token/后续 token 延迟目标选择。不能从 EP4 四卡结果直接推断 EP2 的收益。

## 数据范围

这是对已经完成的实验做离线拆分，没有启动新 GPU 实验。Qwen3.6-35B-A3B BF16，H100 80GB 节点内 NVLink，全部开启原生 HiCache。原实验同时运行两个独立两卡实例，四卡合计64请求；每个实例32请求。下面对每个实例按自身 `max(finished)-min(started)` 重算吞吐，再对两个实例和正式三轮取均值，不是将四卡总吞吐除以二。两实例共享同一节点的 CPU/内存资源，因此不能替代独占两卡、总64请求的实测。

EP2：Attention TP1/DP2、Expert EP2；PP2：Attention TP1、Expert EP1、20+20层。PP2使用本轮通过资格的原生/early-send与microbatch调优；lookahead/tail-local路径未取得本负载资格。每GPU CPU池64 decimal GB KV +8 decimal GB GDN；EP/PP static memory fraction分别0.94/0.92。

P是每请求缓存前缀。Prefill每请求新增Q512、输出1；Decode新增Q1、输出128。缓存恢复、排队均在请求计时内；批前缓存准备位于计时外。这是每批请求吞吐，不是包含批间准备的持续服务吞吐。

## 自然缓存重放：每个两卡实例吞吐

| P | 阶段 | EP2 tokens/s | PP2 tokens/s | 均值比较 | EP2/PP2 CPU前缀token比例 |
|---|---|---:|---:|---|---|
| 8,256 | prefill | 50,149 | 52,948 | PP2高5.6% | 0.0% / 0.0% |
| 8,256 | decode | 2,505 | 2,094 | EP2高19.6% | 0.0% / 0.0% |
| 130,304 | prefill | 11,868 | 17,286 | PP2高45.6% | 53.1% / 44.3% |
| 130,304 | decode | 803 | 780 | EP2高2.9% | 57.8% / 49.0% |

长前缀Prefill存在实例不均衡：EP2两个实例各轮均值为8,704与15,032，PP2为16,986与17,586 tokens/s；四卡聚合取两实例共同完成时间，因此其倍数与单实例吞吐均值的倍数不同。CPU比例是前缀token来源比例，不是请求命中概率。

## 全部前缀从CPU命中的受控重放

| P | 阶段 | EP2 tokens/s | PP2 tokens/s |
|---|---|---:|---:|
| 8,256 | prefill | 44,574 | 51,959 |
| 8,256 | decode | 2,405 | 2,303 |
| 130,304 | prefill | 8,220 | 12,706 |
| 130,304 | decode | 715 | 710 |

这张表CPU前缀token命中比例均100%，CPU→GPU恢复在计时内。长前缀Decode单实例均值715与710接近；两实例的EP2/PP2均值对分别是664/717和766/703，方向不一致，不能宣布稳定胜者。原四卡共同完成指标在此点为1327与1405 tokens/s；它反映整批尾部与实例不均衡，不能直接作为两卡数值。不同缓存模式的历史与生成分支不同，不能用表间差值单独估计H2D成本。

## Decode延迟：首token与后续token

| 模式 | P | EP2 TTFT s | PP2 TTFT s | EP2 TPOT ms | PP2 TPOT ms |
|---|---:|---:|---:|---:|---:|
| natural_replay | 8,256 | 0.207 | 0.133 | 11.21 | 14.30 |
| natural_replay | 130,304 | 1.430 | 0.980 | 17.98 | 21.73 |
| cpu_replay | 8,256 | 0.268 | 0.165 | 11.28 | 12.74 |
| cpu_replay | 130,304 | 2.105 | 1.268 | 17.78 | 22.96 |

EP2在本轮两个上下文的平均TPOT更低；PP2的TTFT更低。因此“Decode选谁”还取决于首token目标、输出长度和CPU恢复量。TPOT是请求平均每token时间，不是与其他布局等并发的单kernel延迟；它不能独立决定服务吞吐。

## 对扩缩容策略的含义

在本轮负载范围内，GPU预算1→2时，Prefill候选优先PP1→PP2；Decode在前缀主要驻留GPU且关注持续生成速度时优先EP1→EP2。长前缀CPU压力明显时，EP2与PP2整体Decode吞吐接近，应把预计CPU恢复、TTFT、TPOT和输出长度纳入决策。模型是否能放进单GPU只是可行性条件，不能单独决定拓扑。

每个两卡实例的GPU KV token池：EP2为3,199,872，PP2为3,502,912，PP2约多9.5%；本轮每实例长前缀合计4,169,728 tokens，均超过这两个GPU池。

这些静态结果支持候选方向，尚未测EP1→EP2/PP2的切换时延、KV/GDN迁移、在途请求以及缩容正确性；也未测试在仅两卡时保持总64请求的负载。因此不能将所提1/2/3/4卡原地路径称为已验证的在线策略。

## 证据

20项布局/负载/缓存/阶段指标，拆分104个实例-轮次记录；热重放每项两个并发实例×3轮，冷Prefill仅1轮。逐请求数量、输入/输出token、总计时及冻结映射已核对；原始文件哈希和映射源码保存于summary.json。两个并发实例不是额外的独立重复实验，未建立统计置信区间。

[逐实例逐轮数据](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_five_layouts_hicache_20261028/two_gpu_analysis/per_instance_rounds.csv) · [指标与来源哈希](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_five_layouts_hicache_20261028/two_gpu_analysis/summary.json) · [原四卡报告](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_five_layouts_hicache_20261028/REPORT.md)
