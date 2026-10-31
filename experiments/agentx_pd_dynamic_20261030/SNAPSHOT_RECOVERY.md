# 计时后快照超时与结果保存修复

2026-10-31。`eight_gpu_col_ep4_balanced_r2`完成C8三轮和C128第一轮后，第二轮C128在`replay()`返回之后、读取结束计数快照时失败。旧实现先采集快照再保存HTTP结果，因此这一轮没有可审计的延迟数据，不计入统计。

证据：共享控制字节为10；011四个rank的快照均停在epoch9，171四个rank均更新到10。没有看到对应的模型CUDA错误；两节点所有权清理和GPU空闲检查通过。该证据提示共享文件mmap通知／可见性存在问题，尚未证明具体文件系统机制。自动队列按失败门槛停止，没有继续启动其他配置。

## 修正

- 控制字节移到各节点的私有tmpfs文件；协调器在计时外写入并取得确认。计数器本身仍为原CPU计数，不增加CUDA同步或新的forward热路径。
- 回放一结束，先原子保存`replay_completed.rN.json`，再采集诊断快照；成功后改名为正式结果并写完成标记。快照异常时保留原始回放数据，仍不伪造诊断成功。
- 三项故障注入／完成标记／路径保护检查通过。两节点各4进程、32轮，共256次本地通知确认通过；011→171的通知命令也通过。

## 继续实验的口径

新目录`eight_gpu_col_ep4_durable_r3`逐文件哈希导入四个已经完整保存的样本，只执行缺少的C128第2、3轮。GPU源码仍为E5，模型、输入、随机命名空间、容量与预热相同；V12在2/4/8位置的路由与原V11逐项相同。`measurement_origin`保留每轮实际来源，不能把导入样本称为新测量。原目录的失败和清理证据完整保留。

实际长回放确认：补跑C128第2轮的2736个请求已完整保存，耗时1214.445秒；结束快照8/8均发布epoch2，数据已转为正式结果并写出完成标记。第3轮已开始，8/8开始快照为epoch3。该检查验证了真实长回放之后的采集路径，完整run的后检查和清理仍待第3轮完成。[第2轮快照](runs/eight_gpu_col_ep4_durable_r3/results/col_ep4/agentx_c128/sticky_hash/forwards.r2.json)

主矩阵后继更新为`eight_gpu_pd_ep4_durable_r3`、`eight_gpu_col_pp2_durable_m16_r3`、`eight_gpu_col_single_durable_kv40_r3`。全部使用V13。原自动队列已停止；新的队列继续要求完整完成、后检查、清理、GPU空闲和冻结哈希通过。

[旧失败日志](runs/eight_gpu_col_ep4_balanced_r2/coordinator.col_ep4.log) · [旧清理](runs/eight_gpu_col_ep4_balanced_r2/task/cleanup.json) · [导入明细](runs/eight_gpu_col_ep4_durable_r3/imported_episodes.json) · [回归检查](snapshot_recovery_tests.log) · [011通知检查](snapshot_smoke_011.json) · [171通知检查](snapshot_smoke_171.json)
