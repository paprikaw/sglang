# P/D 高负载：HiCache 恢复失败

2026-10-31。E6完成C8三轮，C128第一轮失败；无完整C128性能样本。后续控制队列自动停止，两节点清理、拥有进程退出和GPU idle检查通过。HiCache保持开启。

06:58:12，Decode DP3报告：`device_indices=0, new_indices=0, expected decode_prefix_len=27712 (l1=0, l2=27712, l3=0)`；随后将请求作为`Decode transfer failed`终止。原来的可选GDN检查点断言没有重现，四个rank已实际走过E6的延期路径。

此错误由Decode本地HiCache恢复状态机触发，尚不能归因于RDMA。旧日志中的l1/l2/l3是准入时的匹配值，没有记录失败时重新匹配到的host覆盖长度。因此需要区分：

- 已承诺的前缀仍完整，只是KV或GDN设备分配暂时失败。原生hybrid controller会回滚本次分配、返回空结果；应保留PENDING，等待资源并重试。
- 前缀已经不完整，或返回了不完整的恢复结果。必须明确失败，禁止提前允许Decode读取这些页。

E7在独立worktree实现这两个分支。源commit `797f6488cf5fea2f88e38ccaeb1703443deec537`；旧行为回归为3失败/3通过，修复后连同GDN与PP检查共22项CPU测试通过。重试有有限超时，成功提交仍受H2D完成事件和原生跨rank传输门槛约束；保留Prefill传入的当前GDN状态。

E7首轮真实C128运行中，四个Decode rank均走过可选状态快照延期；10次临时HiCache恢复不足均成功重新排队，最长记录4.963秒。没有出现承诺前缀丢失或本地恢复重试超时。成功排队仍需等待H2D完成，不能等同于整个请求完成。

该轮最终在07:30:45因另一个边界失败：Mooncake `WaitingForInput` 超过继承配置中的120秒。源码计时从Decode发送目的缓冲元数据开始，包含等待Prefill排队、计算及交接，不能将120秒解读为网络传输时间。此时Prefill仍持续处理请求；07:30:44各rank队列有19–23个请求，pending token约189–200万。两节点07:33:27完成清理，GPU idle与拥有进程退出均通过；无完整C128样本。

新独立运行 `eight_gpu_pd_ep4_hicache_retry_e7_t300_r1` 恢复该引擎默认的300秒等待上限；保留原来的80状态、8接收槽、64请求/node、HiCache和全部计时。旧120秒失败记录不导入均值，不丢弃新运行中的慢请求，也不减少其等待时间。仍需三轮完整结果和Guard验收。

08:01:48，300秒r1被资源租约失效中止，未形成完整episode；此前18次临时恢复不足均已重新排队，未记录新的Decode传输／本地恢复错误。共享登记中，同一时刻另有三个独立后台租约一起过期，底层主机或GPFS停顿原因尚未定位。两节点的拥有进程退出、GPU空闲检查分别在08:01:57/58通过，但没有最终Guard后检查，不补造验收。登记心跳恢复后，在新目录 `eight_gpu_pd_ep4_hicache_retry_e7_t300_r2` 按相同配置重启；不放宽租约TTL，不绕过登记，不导入未完成请求的计时。

[租约中断与逐节点清理](runs/eight_gpu_pd_ep4_hicache_retry_e7_t300_r1/interruption.json) · [中断记录](e7_t300_interrupted_r1_analysis/REPORT.md) · [当前运行配置](runs/eight_gpu_pd_ep4_hicache_retry_e7_t300_r2/config.json)

缓存检查：这次预热在P和D所有四个Attention-DP rank都命中8192、65536、208320 token前缀；正式运行日志也有P端缓存命中。单个chunk的cached-token=0不能用来断言关闭了缓存。该检查证明路径启用，不证明正式负载具有同等命中率。

[E7失败审计](e7_failure_audit.json) · [E7原始Decode日志](runs/eight_gpu_pd_ep4_hicache_retry_e7_r1/cache/spartan-gpgpu171/pd_ep4.r0.log) · [新运行配置](runs/eight_gpu_pd_ep4_hicache_retry_e7_t300_r1/config.json)

[失败运行](runs/eight_gpu_pd_ep4_checkpoint_e6_r2) · [客户端异常](runs/eight_gpu_pd_ep4_checkpoint_e6_r2/coordinator.pd_ep4.log) · [Decode原始日志](runs/eight_gpu_pd_ep4_checkpoint_e6_r2/cache/spartan-gpgpu171/pd_ep4.r0.log) · [清理](runs/eight_gpu_pd_ep4_checkpoint_e6_r2/task/cleanup.json)
