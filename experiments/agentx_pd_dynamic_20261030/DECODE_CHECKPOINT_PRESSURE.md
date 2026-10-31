# Decode 高负载：可选 GDN 检查点在状态池满时失败

2026-10-31。E5 失败已定位，E6 的 16 项 CPU 检查通过；GPU 两节点启动、串行参考、全 rank 预热和C8三轮已通过。C128触发延期路径后，在另一处HiCache恢复环节失败，整组未验收。所有配置继续开启 HiCache；没有通过关闭缓存或降低 C128 负载绕过问题。

## 失败证据

`runs/eight_gpu_pd_ep4_durable_slots8_r4` 完成 C8 三轮后，在 C128 第一轮失败；没有完整的 C128 性能样本。Guard 后检查未通过，但两节点清理通过、拥有的模型进程全部退出、GPU idle 为 true。[清理](runs/eight_gpu_pd_ep4_durable_slots8_r4/task/cleanup.json)

Decode DP3 的调用链为 `process_prebuilt → prepare_transferred_mamba_state → maybe_cache_unfinished_req → UnifiedRadixCache.cache_unfinished_req → MambaComponent.prepare_for_caching_req → _alloc_mamba_slot`。04:46:04 日志显示：GDN 79/80、运行请求 2、等待传输 22、预分配请求 7；KV 占用约 51%，无回退。随后申请额外缓存状态失败，触发 `AssertionError: Can not alloc mamba cache`。[原始日志](runs/eight_gpu_pd_ep4_durable_slots8_r4/cache/spartan-gpgpu171/pd_ep4.r0.log)

这与之前的启动 OOM 不同：8 个接收预留已让实际状态池保持 80。现在的问题是运行期间，其他活跃或预分配请求消耗了可用状态，未完成请求的可选缓存检查点仍强制申请新槽位。准入时检查可用数不等于为所有未来快照预留了独占槽位。

## E6 修复及边界

隔离源码：`.engine-worktrees/agentx-gdn-pd-hicache-r6-20261031`；commit `fc014e3553c4d234f38f3088a3158c0333e64830`；tag `serving/sglang/qwen35b/agentx-gdn-pd-candidate-r6-20261031`。

- 非 INT8 的未完成请求快照：先尝试分配和正常驱逐；若仍无槽位，延期这次可选插入，保留活跃 GDN、跟踪状态、已有 KV 索引与 prefix 锁。
- 仅在实际捐赠或复制检查点后清除跟踪深度。延期后，后续重试或请求完成仍能使用有效检查点；完成可捐赠已有状态，不需要额外槽位。
- 必需状态分配仍严格检查。没有声称解决所有状态分配路径；INT8 路径不在本次修复验证范围内。
- 不改变 HiCache、状态池大小、KV 自动容量、接收队列、活跃请求上限、模型内核和 V3 回放时序。

每个组件首次延期记录 `MAMBA_CACHE_CHECKPOINT_DEFERRED`。这表示保留请求状态并略过可选缓存插入，**不表示清空 KV**；长期命中率影响仍需实际结果确认。

## 验证

原生类的 CPU 测试在修复前复现 5 个行为失败，修复后 16 项通过。测试执行真实 `UnifiedRadixCache`、`MambaComponent` 和状态池方法，覆盖活跃状态不被覆盖或释放、KV 与锁保留、之后重试、无额外槽位时完成捐赠、所有权释放、可驱逐成功、必需分配仍失败、无双缓冲路径、传输 GDN 与 PP 映射回归。[失败记录](unit_r6_pressure_red_v2.log) · [16 项通过记录](unit_r6_pressure_green_v2.log)

CPU 测试进程单独预加载环境中的新版 libstdc++，解决测试导入依赖的 `GLIBCXX_3.4.30` 问题；未修改正在运行的 E5 GPU 进程环境。最初的导入失败日志保留。

新 GPU 目录为 `runs/eight_gpu_pd_ep4_checkpoint_e6_r2`：相同 V3 回放、C8/C128 各三轮，全部重新测量，未导入 E5 样本；260 项冻结哈希与六条完整串行参考序列已核对。单卡六轮及后检查、清理通过后，队列已启动该目录。`r1` 因缺少参考文件仅保留准备记录，明确禁止启动。

06:33左右，两个节点的实际GDN均为每rank80；P侧原生KV每rank2,225,472 token，D侧2,195,008。八个rank均确认HiCache开启、32 decimal GB、静态显存比例0.95。六条串行输出与参考一致，全rank冷／缓存复用预热通过，进入C8第一轮。[输出检查](runs/eight_gpu_pd_ep4_checkpoint_e6_r2/evidence/pd_ep4/serial_qualification.json) · [预热](runs/eight_gpu_pd_ep4_checkpoint_e6_r2/evidence/pd_ep4/untimed_warmup.json)

06:56，C8三轮全部完成，开始C128第一轮。06:56:34–39，四个Decode rank都记录了首次`MAMBA_CACHE_CHECKPOINT_DEFERRED`；之后继续执行Decode，日志出现GDN79/80或80/80、KV占用约52%–61%的状态。这证明真实GPU已进入原失败条件下的延期路径，尚不等于整轮通过。[带后续Decode记录的现场快照](e6_pressure_live_20261031_0657.json)

06:58:12，C128在另一处HiCache恢复环节失败：已承诺的27,712个host前缀token未恢复，被上层报成传输失败。此次没有重现原可选检查点断言，但全组仍未通过；两节点清理通过。[下一处问题与修复边界](DECODE_HICACHE_RESTORE_PRESSURE.md)。E6与E5性能表分别标注版本，失败运行不能计作P/D性能差。

## 保留的低负载样本

失败 E5 目录的 C8 三轮均完整：TTFT 572.0373 ms、请求平均 TPOT 6.922716 ms/token、间隔加权 TPOT 6.944185 ms/token、请求 E2E 3590.8083 ms、根会话 282.5784 s、压缩整轮 451.8011 s；host 命中与回退均为 0。它们是失败整组中已完成的原始样本，主验收矩阵暂不把该整组标为通过。
