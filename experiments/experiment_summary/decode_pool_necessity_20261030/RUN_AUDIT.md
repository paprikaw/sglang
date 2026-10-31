# 运行与资格审计

原始目录统一位于 `/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_decode_pool_necessity_20261029/runs/`。失败记录保持原样，没有改写 postcheck 或把失败赋予一个性能值。

| 运行 | 实际状态 | 纳入的性能记录 | 解释 |
|---|---|---:|---|
| 011_screen_r1 | 失败，拥有进程清理、租约释放 | 0 | auto KV 占用过多，PP2 首次创建 NCCL channel 时无法分配 10 MiB；此前 EP4 记录也未混入正式汇总 |
| 011_screen_r2 | 失败，无模型启动 | 0 | 向远端传了相对 run 路径，worker 未读取到配置；Guard 先 quarantine 后按原有 grace 策略释放，未手动删除租约 |
| 011_screen_r3 | 部分完成，整次 postcheck 失败；拥有进程清理、租约释放 | 44 | 最后 four_ep1_dense 启动失败，前三个 dense 单卡点无性能值；之前八个 arm 的完整性能记录独立校验通过。另有一个 128K/N32 单卡容量记录，未计入性能数 |
| 011_ep2_r4 | 完成，postcheck／权重使用／清理通过 | 16 | 同节点补齐 2×EP2，覆盖全部 16 个 workload |
| 011_single_dense_r5 | 未计时预热失败，拥有进程清理、租约释放 | 0 | static fraction0.98、按需 KV 可以启动；native GDN CoW 临时申请 1.82 GiB 时 OOM。没有正式性能样本 |
| 171_screen_r1 | 完成，postcheck／权重使用／清理通过 | 42 | EP4、2×EP2、PP4；另外六个 N128 EP4／EP2 条件因初版 Mamba 池将本地并发限制为25而跳过，不能解释为 EP 固有容量极限 |
| 171_dense_r2 | 完成，postcheck／权重使用／清理通过 | 9 | 修正 Mamba 状态预算后，EP4、2×EP2、PP4 的三个 N128 条件均成功计时 |
| 171_reverse_r3 | 完成，postcheck／权重使用／清理通过 | 9 | 顺序为四单卡→优化 PP2→EP4；P4096/N8、N64，P131072/N8，每点五轮 |
| 171_long32_r4 | 完成，postcheck／权重使用／清理通过 | 2 | P131072/N32，PP2→EP4，各五轮；专门复核小幅 E2E 优势 |

011_screen_r3 第一次 Guard 申请因上一失败 run 尚在 quarantine grace 中被拒绝，未启动 worker；等待注册表按原规则释放后，使用同一未修改的冻结 spec 成功申请。没有抢占其他前台实验或绕过锁。

## 如何验收部分完成的主扫描

`audit_arms.py` 使用原 Guard 的 `_json_sha256`、`validate_contract` 和 `_verify_usage_log`。检查原始 spec、contract、lease receipt 和绑定哈希；对每个实际完成 arm 检查新增 canonical CPU loader 记录及精确 rank 覆盖、BF16 完整层覆盖、64／128／256 专家分片、拥有进程组清理。它不要求旧租约重新活跃，不修改原绑定，不把未启动的 arm 伪装成通过。

`analyze.py` 再从全部原始正式请求复算 TPOT／E2E／TTFT，与保存的均值一致；检查 cached_tokens=P、输入 P+1、输出128、无 CPU KV 命中、四 rank 的 Decode／Graph 资格计数。所有 PP2 正式输出与对应原生固定分组输出逐 token 一致，ahead／consumed 正数且平衡。

主扫描的44条记录均在最后启动失败前完成；失败发生在新的 four_ep1_dense arm，未发生在这些计时窗口。原整体 `passed=false` 仍保留在 [已完成 arm 审计](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/accepted_arm_audit.json) 和原始 Guard 文件中。本报告只验收列表中的完整记录，**不声称整次主扫描成功**。

总计：122条性能记录，388个完整正式cohort、16376个请求。初版六个受 Mamba 池影响的容量行和主扫描一个长前缀单卡容量行另存，不计入上述122条。相同输入的独立复测分别报告，没有跨节点合并轮次。

## 复现

从实验根目录运行：

```bash
python3 collect.py --output /ABSOLUTE/OUTPUT
```

它逐个处理明确列出的六次验收运行，生成主扫描、五轮复测、补充对照、完整轮次和来源校验。`prepare.py`、`verify_workload.py` 保存新运行的冻结与执行逻辑；各原始 run/code 才是当次实际执行的版本。启动必须通过已资格验证的 Guard bundle，使用仍有效的预约；本报告不是跳过 Guard 的启动许可。

完整 engine commit、tag、DeepEP binary hash、缓存 profile、节点／Slurm job、实际 rank 映射及代码哈希见各 run 的 launch_manifest.json。所有旧日志保留在原始目录；快照保存汇总、CSV、审计和代码，按路径及哈希引用日志。
