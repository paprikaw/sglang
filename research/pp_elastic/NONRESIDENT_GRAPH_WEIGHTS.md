# Graph 预热与权重驻留分开

用户于 2026-11-11 明确：候选布局不能靠一直保留 GPU 权重来换取快速切换。历史约 0.3 秒是权重常驻的诊断结果，不能用它证明节省 GPU 资源。

**当前实现。** `--elastic-pp-prepared-weight-policy offload` 是预热 Graph 的默认权重策略。启动前将每层权重放到稳定的 CUDA 虚拟地址，保留参数及其视图的对应关系；预热候选 Graph 后，退出布局的层解除物理页映射。重新接班时分配空页，经过正常权重传输、状态交接和全组确认，再选择已捕获的 Graph。Graph 的地址记录不保存另一份 GPU 权重。有限候选固定最后输出 rank，因此同时删除其他 rank 为通用弹性预留的 LM head 和 final norm；空阶段也不能暗中留下这部分模型权重。`reuse`、`recopy` 两种旧策略仍保留为显式诊断对照，均不符合这次的权重释放要求。

**验收。** 验证 PP2→PP3→PP2 的完整输出、HiCache 恢复、目标 Graph 实际执行，且运行时捕获次数为零。每次清理后要求非活跃层物理权重页为零，切换实际传输字节大于零，记录驱动可用显存变化。分别报告后台准备、搬权重、最后暂停和完整切换；不把并行 rank 的时间相加。

**边界。** 首轮只接非量化模型。KV／GDN、Graph 工作区和 worker 上下文仍需单独计费；本次释放权重不等于已归还整个 GPU。CUDA VMM 直接分配不计入 PyTorch `memory_allocated`，因此必须结合物理映射账本与驱动显存数据判断。任何失败都保持失败记录，不将旧 resident 测试转记为新路径通过。

实现和真实 GPU 验证进度见 [TODO](TODO.md)。

**实测结果。** 冻结源码 `26bdee39146f`，35B BF16、A100、B1、HiCache／NIXL 的两次独立 PP2→PP3→PP2 往返均通过。完整扩／缩平均 **19.98／20.65秒**，其中后台搬权重 **19.68／20.30秒**，最后交接暂停 **239／285 ms**。每方向实际传输21.87 GB，权重复用字节为0；运行时capture=0，Graph选择约2.34／2.57 ms。当前权重走GPU→CPU→GPU／Gloo，NIXL只用于KV补丁，因此不能把最后暂停当成全部切换开销。

**释放证据与资格。** 缩回PP2后，中间GPU的13层物理权重页和注册模型参数均为0；每轮解除映射 **20.37890625 GiB**，驱动可用显存增加相同字节数。所有rank的非活跃层物理权重页均为0。两轮暖缓存输出连续性、目标Graph实际replay、在线KV和最终CPU前缀loadback通过；冷暖生成等价未验证。最终1024项完整CPU回归通过（114.57秒），各轮监督、清理、正常lease release及129个冻结输入核对通过。KV／GDN、Graph工作区和worker仍常驻，**没有在转换期间归还整个GPU**。目前只验证35B的初始化峰值，更大模型需另外检查。两轮为描述性重复，不支持稳定尾延迟结论。见[专题报告及原始指标](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_nonresident_graph_weights_20261111/summary.md)。
