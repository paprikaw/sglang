# 122B P/D：KVCacheD 容量随 PP 扩缩

固定 P=PP2，先检查 D 的 PP2→PP3→PP2。虚拟地址按最高容量预留；每个 PP 档位有独立的实际 KV 容量和物理页预算，切换时保持存活 token 的槽位。HiCache、GDN、Graph、CPU 权重 offload 和 NIXL 继续保留。P→D 注册只覆盖已映射页面，单独传输 packed K/V 的真实行步长。

先用 8192→12288→8192 token 做最小验证：请求输出连续且一致、真实 CPU HiCache 恢复、物理映射页回收和独立进程复用退役卡。高槽位仍被请求或缓存使用时拒绝缩容，不能截断数据。映射在后台用独立流完成新页清零；回收在排空后的切换边界完成。保留完整的准备、暂停和残留 worker 显存。

通过后才接短 AgentX＋BurstGPT：相同到达、reactive 策略、HiCache CPU 预算和完整收尾，对比原地加一张卡与新增 PP2 实例。记录实际 KV 容量、命中、端到端延迟、GPU 时间；不能用预留虚拟地址或逻辑布局卡数代替实际资源。

当前状态：r5最小GPU验证通过，容量8192→12288→8192；9216-token请求用到槽位10240，物理KV240→336→240 MiB。缩回中间rank KV／权重0、仍reserved1.72 GiB，独立64 GiB复用成功。两个方向各一个512-token持续请求逐token一致，CPU HiCache恢复576 token；控制1.951／1.073秒。一次有界验证，端到端配对与多请求资格分别检查。[结果与原件](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_reactive_burst_20261113/KVCACHED_CAPACITY.md)。
