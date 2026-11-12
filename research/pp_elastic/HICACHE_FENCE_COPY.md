# 减少HiCache最终交接复制

2026-11-12。已有35B／A100／B1的CPU预打包往返，最终暂停约264 ms，Graph选择仅约2 ms。HiCache快照、传输和目标准备占大头；各rank存在同步等待，不能把每个rank的最大分项相加。

**本轮改动。** 显式开启`--elastic-pp-hicache-fast-copy`（smoke为`--hicache-fast-copy`）。连续CPU行直接复制，任意行仍按原顺序gather／scatter；快照只保留一次独立副本。目标池由现有工厂重新创建，核对其storage与源不重叠后，写入新接收行时不再备份目标中尚未启用的旧内容。通用适配器默认仍保留undo；只有明确私有、未发布的目标可省略。源缓存一直保留至发布，准备或导入失败时丢弃目标。

**保持的条件。** HiCache及所有CPU前缀继续保留，仍检查完整暖缓存输出和最终CPU前缀恢复。CUDA Graph、持续NIXL KV patch、CPU权重预打包和候选GPU权重offload不变，不减少缓存容量、不借助GPU权重复用。此项只优化CPU复制，不改变KV池伸缩或整卡退出的范围。

**验证。** 增加CPU行顺序／稀疏映射／非连续stride／快照隔离、源别名拒绝、共享目标回滚及私有目标失败清理检查。先跑完整CPU清单，再用同源、同节点、相同负载的开关对照，分别报告最终fence、完整切换、跨已记录fence的客户端token间隔，并拆开HiCache初始化、保留行复制和新接收行复制。新路径未通过GPU前不宣称暂停收益。状态统一见[TODO](TODO.md)。
