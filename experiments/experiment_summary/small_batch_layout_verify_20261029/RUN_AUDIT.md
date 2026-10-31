# 运行与资格审计

最终性能表只收录 `h100011_r9` 与 `h100171_r8`：15 个完成点、75 个正式 cohort、1400 条请求。所有正式轮次保留，没有选择最快轮次。两个节点各有自己的 EP4 对照，未跨节点平均。

| 尝试 | 状态与处理 | 是否进入最终性能表 |
|---|---|---|
| 011 r1 | 仅准备 | 否 |
| 011 r2 | 当前 Guard 未支持 Qwen cache profile，子进程未启动 | 否 |
| 011 r3 | CLI／CPU daemon source 声明检查未通过 | 否 |
| 011 r4 | 旧 bundle 与当前共享 registry 锁协议不兼容，未完成交接 | 否 |
| 011 r5 | 为修正锁协议主动取消；拥有进程清理通过 | 否 |
| 011 r6 | 原生 GDN 池实际只允许每 rank 3 个请求，不满足 N32；取消并修正容量 | 否 |
| 011 r7 | EP4 完成；PP2 新的 B2 静态消息未在旧 allowlist 中，预热失败 | 否 |
| 011 r8 | B2 消息通过；B2 lookahead guard 未覆盖，执行计数为零，资格检查拒绝 | 否 |
| 011 r9 | EP4 与 2×PP2 共 6 点全部完成，原生固定分组输出、四个 stage 执行与退休计数、Guard／cache／cleanup 均通过 | 是 |
| 171 r8 | EP4、4×单卡、原生 PP4 共 9 点全部完成，Guard／cache／cleanup 均通过 | 是 |

此前尝试保留在 `/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_small_batch_layout_verify_20261029/runs`。失败状态保持原样。

使用私有兼容 Guard bundle：保留已有 Qwen admission 和 cache/source 检查，仅将共享 `gpu_registry.py` 换成当前 canonical 的原文件，以匹配 POSIX 锁协议。双向 CPU 互斥测试通过，bundle 文件与两套实际 engine source 身份均冻结。此前自动审核拒绝了继承整个 source 列表的方案；最终保留真实 inference commit，并另行明确声明实际 CPU weight daemon commit，不沿用被拒方案。

PP2 在私有实验适配器中将静态消息和 lookahead 的形状集合从 B4/8/16 扩至 B2/4/8/16；其余边界、结束、请求所有权和消息 schema 检查保留。CPU 检查确认 GDN 边界与结束条件仍拒绝提前执行；每个正式 PP2 点还逐请求比较原生固定分组与 lookahead 的全部 128 输出 token，并检查实际提前执行与退休计数，不能仅凭 policy=9 判定优化生效。

两个最终运行的 GPU 资源已释放；共享 manager 已在两个节点恢复 `verified_live_worker` 的后台 benchmark。CPU 权重缓存保留，无新 Slurm 预约，无缓存替换。
