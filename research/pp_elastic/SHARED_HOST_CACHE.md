# 同节点 PP 切换：复用 CPU HiCache

2026-11-13。已有 PR #10 已合入项目分支；本轮代码位于 `codex/pp-hicache-shared-host-20261113`，已有有限A100暖缓存往返及同源开关对照。设计背景见[用户澄清与方案](DISTRIBUTED_HICACHE.md)，完成状态见[统一清单](TODO.md)。

**怎样复用。** 启动预热完成、接收请求之前，按全局层号建立一份 memfd 共享 CPU 存储，各候选 worker 独立映射和注册。每个 worker 只操作当前负责层的视图；PP 改变时，底层历史 KV／GDN 留在原处。视图保留完整存储的真实行间距，MHA／GDN 的 kernel I/O 按该间距读写，不能假定本地层在每个 token 行中紧密排列。

**怎样交接。** 仍通过已有 NIXL 路径搬活跃 GPU KV，补齐最新 GDN 和请求进度。最终排空在途 HiCache I/O，核对原活跃阶段的前缀槽位和分配／回收状态完全一致，然后建立目标视图并共同发布布局。本路径不做历史 CPU 数据快照、内容扫描或 CPU→CPU 复制；仍需处理元数据和正常 CPU↔GPU 读写。槽位状态不一致时拒绝交接，不猜测对应关系。目标构造失败只销毁目标视图，提交后才退休源视图。

**启动方式。** 在已有合格的 native／HiCache／预热 Graph 配置中增加 `--elastic-pp-hicache-shared-host`。首版限定单节点、TP1／EP1、page_first、kernel I/O、L2 cache、overlap backing，以及候选权重 `offload`；不支持 L3、跨节点或独立暂停某个 worker 的 CPU 池。这是实验开关，默认关闭。

**内存代价。** 启动时以原活跃阶段的最小组件槽位容量建立全模型共享池；之后容量不随 PP 阶段数改变。初始化期间原私有池和新共享池短暂共存，必须有足够 CPU 余量。各进程虚拟地址可不同，独立持有描述符；创建者退出不会使已接入的映射失效。退休一个布局不能解除仍在使用的页锁定；正常退出时再释放本进程的整个共享池。候选 GPU 权重继续按原要求 offload，Graph／KV 常驻范围没有因此扩大。

**已有开销作为参照。** 上一版缩容最终暂停约87 ms，其中 CPU 快照／内容核验约39.8 ms；准备期各阶段先后占调度线程约33–37 ms。它们来自有限单轮实验，不能直接预测本方案会减少同样的时间。共享方案仍有 I/O 排空、元数据核验、活跃 GPU 状态收尾及 Graph 选择，需要真实往返测量。

**验证结果。** 修正版`d9a5731c1222`的1206项CPU检查全过。同A100／35B BF16／B1、每模式一次PP2→PP3→PP2，共享开关使扩／缩最后暂停105.2／88.5→43.8／39.0 ms、完整转换1.342／0.879→1.007／0.610秒；准备期最大token间隔74.1／96.2→26.5／58.0 ms。两个方向各2048输出与各自同进程暖参考及配对模式逐token一致；实际CPU前缀loadback、Graph replay、runtime capture=0及清理／正常release通过。历史CPU快照／发送／复制为0，底层存储身份不变，候选非活跃GPU权重物理页为0。不是重复均值或整体吞吐结论。[报告及全部绝对指标](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_hicache_shared_host_20261113/summary.md)。

**失败与限制。** 首轮启动因额外L3建组阻塞，未进入请求，254.904秒内正常清理；修正版按L2-only范围跳过该collective，并修正空阶段容量，失败保留。共享池full=7104／GDN=29、逻辑总量约1.875 GiB，固定为初始容量；槽位不一致会拒绝切换。尚无通用共享槽位分配、更多模型／batch、跨节点或冷暖生成等价资格。Graph／KV／GDN和worker仍可驻留，没有证明整卡归还。代码由主agent实施和自行检查，[PR #11](https://github.com/paprikaw/sglang/pull/11)保持Draft，独立review待补。
