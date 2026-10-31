# PP2 / EP2 Decode TPOT：离线时序诊断（2026-10-27）

结论：当前优化后PP2的剩余劣势，主要应从**模型Graph之间的stage交接、输出反馈与CPU调度开销**解释。GEMM差额不足以独立解释端到端差距。现有trace同时显示PP计算被压缩后，调度空档更加突出。

本次仅分析已保存结果，没有新增GPU测量。固定Qwen3.6-35B-A3B BF16，H100178同一NVLink GPU0/1，全局8请求；每请求缓存4096、新输入1、生成128。正式性能为5轮，profiler关闭；下述细分为另一次profiler采集，各rank丢弃前4个稳定B4 Graph后选16个，与原报告分项使用完全相同的Graph。

| 优化后配置 | Expert GEMM，µs/GPU/层 | 正式TPOT，ms | 正式Decode，tok/s |
|---|---:|---:|---:|
| EP2 | 47.74 | 7.663 | 886.9 |
| PP2 | 57.31 | 9.277 | 790.2 |

每层GEMM差约9.57 µs。按40层折算为约0.383 ms，而正式TPOT差为1.614 ms（PP高21.1%）。这是量级核对，不能当成从正式TPOT扣除的一项精确归因。EP每层dispatch+combine GPU事件累计约34.07 µs；PP省去逐层DeepEP交换，其整体模型Graph反而更短。

**同一profile内部的逐token闭环。** EP两rank共32个同批次相邻token周期；PP16个同批次周期，跨卡按`g_sk`和同值出现序号匹配，验证下一token回到相同槽位。每一个PP样本都满足stage0结束≤stage1开始<stage1结束≤同槽位下一token的stage0开始，四段严格相加。

| Profile中的同一批请求 | GPU模型Graph | 前向stage交接 | 回传/下一输入准备 | 完整周期 |
|---|---:|---:|---:|---:|
| EP2 | 7.400 ms（40层） | — | 0.093 ms（Graph之外） | 7.493 ms |
| PP2 | 2.694+3.074=5.769 ms（20+20层） | 2.572 ms | 2.335 ms | 10.676 ms |

PP模型Graph总计约5.77 ms，但同一组4请求必须跨两个stage，再等输出返回和下一输入就绪；两组4请求交替，一个请求的token间隔覆盖两个微步。EP每GPU同时推进本地4请求的40层，单个微步即覆盖全局8请求。**不能将PP一个20层forward的2.7–3.1 ms直接与EP的40层forward或请求TPOT比较。**

Profile里EP模型Graph覆盖各卡观察窗口约98.7%/98.8%；PP约50.5%/57.6%。这是Graph时间覆盖率，不是SM利用率；DeepEP等待已包含在EP的Graph之内，NCCL等待可能落在PP的Graph之外。

**源码与开关解释了这个模式。** 启动日志确认EP `disable_overlap_schedule=False`，PP为`True`，并明确记录“Pipeline parallelism is incompatible with overlap schedule.”。PP `pp_async_batch_depth=0`，`pp_loop_size=pp_size+depth=2`，实际两个B4槽位。[PP调度源码](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/glm47-ep16-canonical-profile-20261004/python/sglang/srt/managers/scheduler_pp_mixin.py:556)依次发射当前batch，收取/处理待退休batch的输出，`d2h_event.synchronize()`，然后才提交当前stage激活；[结果退休及发送位置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/glm47-ep16-canonical-profile-20261004/python/sglang/srt/managers/scheduler_pp_mixin.py:648)。同步本身可与当前GPU计算重叠，不能把CPU同步时长再次加到Graph上；关键是未被覆盖的stage交接和下一轮准备进入了请求关键路径。

**PP通信时间也需区分等待。** Trace中的激活消息为两个8192元素的BF16数组（合计32 KiB），回传token为4个int64（32 B）。rank0的NCCL kernel可驻留数ms，而rank1对应kernel仅约数µs，符合接收端等待结果就绪的模式。这不是对NVLink纯带宽的测量，也不能把整个NCCL驻留时间算成搬运这些字节的成本。原始消息shape与事件驻留统计保存在timing.json。

**正式TPOT不是流式计数误差。** 四个配置各40请求、每请求128个事件，合计20480个事件均每次增加一个completion token，没有重复计数；根据唯一token重算TPOT与原值完全相同。TPOT仅计算首token到末token的127个间隔，不含TTFT。

Profiler确实改变了时间：PP的profile周期为10.68 ms，正式TPOT为9.28 ms；EP分别为7.49/7.66 ms。因此，profile能证明等待位置和规模，不能声称正式9.28 ms里精确有4.91 ms调度开销。精确的profiler-off因果占比仍需要低干扰、按槽位记录GPU事件的受控测量。

下一步应优先优化PP的stage交接和autoregressive反馈闭环：让已就绪激活更早发送，将输出回传、D2H与CPU结果处理从下一批发射的关键路径移开，并联合选择微批次大小与在途槽位数。通用GEMM改进仍同时应用双方。两stage Graph时间也存在2.69/3.07 ms的不均衡；即便理想调度，单纯均分层数也不完全等于均分时间。假设保持这些stage执行时间、完全隐藏调度且有两个活跃B4槽位，模型给出的计算周期下界为`2×max(2.69,3.07)≈6.15 ms`；这只是条件模型，尚未实测达到。

[时间轴PNG](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/tpot_analysis_20261027/tpot_timeline.png) · [PDF](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/tpot_analysis_20261027/tpot_timeline.pdf) · [全部逐周期数据](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/tpot_analysis_20261027/profile_cycles.csv) · [计时、开关及来源哈希](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/tpot_analysis_20261027/timing.json) · [原公平对比报告](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/deliverables/REPORT.md)

![Profiled model execution and PP token feedback](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_fair_expert_20261026/tpot_analysis_20261027/tpot_timeline.png)
