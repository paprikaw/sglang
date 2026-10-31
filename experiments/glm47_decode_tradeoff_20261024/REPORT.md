# A100 GLM：EP8 与 PP2×EP4 的 Decode 取舍

只复算保存的数据；本次没有启动 GPU。两种布局均使用 8 张 A100，Attention 在节点内 TP4。

## 1. 普通 Decode 的计算与通信预算

B8、P4096（Bank 实际保存4032）、Q1、64输出，两个B4 slot，CUDA Graph。
以下只统计实际 Decode Graph，不包含开头的残余 Prefill；语义上下文相同，但不宣称跨布局输出逐token完全相同。
PP 是两个半模型的分项工作量合计，不是 HTTP TPOT，也不是同一时刻的墙钟。Graph 外等待和 PP 边界尚未纳入。

| 分项 | EP8 ms | PP 路径合计 ms | PP−EP ms |
|---|---:|---:|---:|
| graph_span_ms | 89.9074 | 84.2531 | -5.6544 |
| attention_ms | 15.6454 | 15.5962 | -0.0493 |
| expert_ms | 22.7657 | 23.7524 | +0.9867 |
| communication_ms | 25.6056 | 18.7439 | -6.8617 |
| expert_routed | 7.0348 | 7.6583 | +0.6235 |
| dispatch_combine | 19.8154 | 13.5729 | -6.2425 |

通信仅算 Dispatch+Combine，节省6.2425ms，已经足以覆盖 Routed Expert 多出的0.6235ms。
含TP的模型内部通信节省6.8617ms，全部Expert/FFN多出0.9867ms，二者差额5.8750ms。
这说明计算/通信侧存在余量，不证明最终TPOT改善。插桩改变执行节奏，均值掩盖最慢rank，不能直接把余量当作服务延迟。

## 2. 稳态预测模型

固定全局B与请求组成；EP8使用两个Attention TP4组，PP使用两个B/2 slot。设总专家数E、top-k为k。
在生产D1 source scatter与均衡路由下，两者每卡每层实际Expert配对数均为S=B*k/8；PP每卡持有E/4个expert，EP8为E/8。
因此有用FLOPs相同，Expert数量、每Expert行数、padding和权重复用不同。不要因为PP slot减半就把它每卡的Expert工作也减半。

定义同一全局Decode轮次、同一计时协议：

    预计PP净收益 = 暴露通信节省 − Expert额外计算 − Attention差额 − PP额外开销

PP额外开销包含边界激活/反馈、CPU调度、阶段不均衡及无法隐藏的依赖；必须用时间线校准，不能将所有GPU空隙叫气泡。
如果只看目标Graph，两个slot的理想吞吐服务下界约为2*max(G0,G1)，不是G0+G1；反馈/依赖可能使实际周期更大。
本次短context普通Decode的Graph均值是EP=89.9074ms，PP两stage=40.7895/43.4636ms。
用这些诊断均值作说明，2*max=86.9271ms，与EP还差约2.9803ms。该数不是实测PP overhead或TPOT门槛，只说明阶段不均衡还会减少可兑现余量。

## 3. 长context普通Decode：收益去了哪里

独立cohort：B8/P81920/Q1、32输出，065/167；Graph OFF、overlap OFF。先恢复并预热GPU前缀，采稳态；不是首次HiCache换入计时。

| 分项 | EP8 ms | PP两stage核算 ms | PP−EP ms |
|---|---:|---:|---:|
| mean_microbatch_cycle_ms | 209.9652 | 211.3255 | +1.3603 |
| attention_ms | 112.1267 | 110.6395 | -1.4873 |
| expert_ffn_ms | 8.4894 | 9.1643 | +0.6748 |
| communication_ms | 69.9901 | 57.4036 | -12.5864 |
| remaining_ms | 19.3589 | 34.1182 | +14.7592 |

原记录：GPU无活动间隙16.07→29.04ms，增加12.97ms，其中forward内+8.31、forward间+4.66ms。不能据此认定全部来自固有pipeline bubble。
同一诊断实验的无profiler TPOT为EP201.162、PP205.379ms。CUDA-only profiler引入约6%扰动，以上预算只用于定位。
通信省12.59ms而Expert/FFN仅多0.67ms，仍未转为TPOT优势；当前GLM更应检查额外提交/同步/反馈依赖。

## 4. MTP Decode 单独记录

修复Verify Attention之后，目标Verify路径D+C节省8.7845ms，Routed Expert多0.9233ms。
这是EP8已有同Bank参照与50/42层PP的历史对照，非新鲜严格配对实验。未计draft、接受率、PP边界与反馈，不能除以名义草稿数得到TPOT。
正确比较应使用完整Verify+draft+反馈周期 / 实际接受输出token数，并验证语义/输出等价。较早错误Verify Attention后端的MTP数据没有纳入。

## 5. Expert token扫描：保留原口径，不与上表相加

单张A100真实GLM AWQ单层，Graph主表；S为每卡实际token–expert配对数。旧probe采用D4来源副本，原logical B=S/4，不是生产Decode请求B。
生产D1下top8、全局B8 Decode对应S约8；低于这个匹配扫描最低S32，因此不能直接外推。

| S | EP8 ms | PP slot平均 ms | PP−EP ms |
|---:|---:|---:|---:|
| 32 | 0.09318 | 0.09421 | +0.00102 |
| 128 | 0.19968 | 0.28467 | +0.08499 |
| 512 | 0.33843 | 0.37120 | +0.03277 |
| 2048 | 0.78541 | 0.81510 | +0.02970 |
| 8192 | 2.65421 | 2.63219 | -0.02202 |
| 16384 | 5.20448 | 5.25286 | +0.04838 |

大工作量下约1%的差异不足以认定胜者。小工作量的差异随活跃expert数、tile padding和调优改变，不保证随batch单调。只测单层热重放，不代表逐层冷权重流量。

## 判断

已有普通Decode实测支持：PP的通信节省能够超过其Expert计算代价。仍缺的是将剩余收益在稳态关键路径兑现。
当前证据不足以给出通用的B阈值；需要同一Graph/传输协议、相同P与全局B的完整Decode周期来校准PP额外开销。
长P增大Attention工作和KV需求，不直接增加每轮Decode的Expert/通信token数；不要把长P当成更大通信包。

## 来源

- [stage_activity_summary.json](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_semantic_bank_fourway_20261023_v8/runs/event_profile_followup_r1/ep8_mtp_off/stage_activity_summary.json) SHA256 `d28bf8a4784de230e0842acff996cd59c3dd0d168244d21aa7760beb39532a3c`
- [stage_activity_summary.json](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_semantic_bank_fourway_20261023_v8/runs/event_profile_followup_r1/pp2ep4_mtp_off/stage_activity_summary.json) SHA256 `cdf9335d880b06e38bd6923f118d920110d03c1369e1d8ba0f1e879a198643ad`
- [stage_activity_summary.json](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_mtp_verify_fixed_ep8_pp2ep4_20261023/runs/paired_r1/ep8_mtp_fixed/stage_activity_summary.json) SHA256 `8566ed2b244db8270f084b264f8e49001af65c738999a7adbd44bfa2f99dc5e7`
- [stage_activity_summary.json](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_pp_stage_balance_20261023/runs/balance_r1/pp2ep4_balanced/stage_activity_summary.json) SHA256 `1a1384e8f70947abd67c98a7a6cbb1d857ea629156c6fd08d23a5e14f90cb473`
- [comparison.csv](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_ep8_pp2ep4_expert_compute_20261021/runs/r2/analysis/comparison.csv) SHA256 `a54b1753a44dd6ba3f41e98d648cef33080a161c52e660c7a61605322b468375`
- [stage_metrics.csv](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_ep8_pp2ep4_cached_decode_components_20261022_r1/stage_metrics.csv) SHA256 `0aa4d2e0b18d3e613f5297ad33929e474121973b3c9389a2788385f56fa722f8`
- [overhead_analysis.md](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_ep8_pp2ep4_cached_decode_components_20261022_r1/overhead_analysis.md) SHA256 `997b9f5f3f915208517fe589c769c318684ed92a97205b1109fe30a6de11d531`
- [summary.md](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_ep8_pp2ep4_cached_decode_components_20261022_r1/summary.md) SHA256 `cef6f227c0b1fab40aa3a9d60976f556c22c53d2e9002c13db81f5643116115c`
- [EVENT_PROTOCOL.md](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_semantic_bank_fourway_20261023_v8/EVENT_PROTOCOL.md) SHA256 `644f5d0005e79810c031116aec15db4ae859cf316d622e3aca99335b21893766`
