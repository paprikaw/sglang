# 减少HiCache最终交接复制

2026-11-12。已有35B／A100／B1的CPU预打包往返，最终暂停约264 ms，Graph选择仅约2 ms。HiCache快照、传输和目标准备占大头；各rank存在同步等待，不能把每个rank的最大分项相加。

**实测结果。** 同源无fast-copy／开启fast-copy各一次正式往返，扩／缩暂停253.4／296.3→160.0／195.9 ms，客户端跨fence的token到达间隔281.8／310.3→184.8／210.6 ms。扩容接收GPU对应的CPU行导入111.0→20.7 ms，目标初始化31.1→30.8 ms，Graph选择约2 ms；改动减少的是重复复制，CPU缓存传输和目标初始化仍在。1130项完整CPU通过，两个方向各2048输出token一致，HiCache字节数及容量相同，候选GPU权重仍非驻留。[完整分项与来源](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_hicache_fence_copy_20261112/summary.md)。

**本轮改动。** 显式开启`--elastic-pp-hicache-fast-copy`（smoke为`--hicache-fast-copy`）。连续CPU行直接复制，任意行仍按原顺序gather／scatter；快照只保留一次独立副本。目标池由现有工厂重新创建，核对其storage与源不重叠后，写入新接收行时不再备份目标中尚未启用的旧内容。通用适配器默认仍保留undo；只有明确私有、未发布的目标可省略。源缓存一直保留至发布，准备或导入失败时丢弃目标。

**保持的条件。** HiCache及所有CPU前缀继续保留，仍检查完整暖缓存输出和最终CPU前缀恢复。CUDA Graph、持续NIXL KV patch、CPU权重预打包和候选GPU权重offload不变，不减少缓存容量、不借助GPU权重复用。此项只优化CPU复制，不改变KV池伸缩或整卡退出的范围。

**验证。** CPU检查覆盖行顺序／稀疏映射／非连续stride／快照隔离、源别名拒绝、共享目标回滚及私有目标失败清理。GPU对照同时记录服务端fence和实际客户端逐token间隔，拆开HiCache初始化、保留行复制和新接收行复制。初轮漏开客户端计时，只保留正确性及服务端观测，不与正式对照混算；正式新／旧190.686／179.790秒内完成监督清理并正常release，各136项冻结输入一致。仍限35B／B1／native TP1／EP1 Triton，更多模型／batch和整卡归还未扩展，无独立reviewer。后续prepare自动冻结HiCache实现源文件，本轮在启动前已将同一文件追加到输入清单。状态统一见[TODO](TODO.md)。

**后续（2026-11-13）。** 目标CPU池及稳定行已支持后台准备；参见[后台机制与新对照](HICACHE_BACKGROUND.md)。此页历史测量保持原样。
