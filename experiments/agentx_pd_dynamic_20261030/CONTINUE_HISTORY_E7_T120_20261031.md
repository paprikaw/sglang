# 当前继续执行入口：2026-10-31 07:22 Melbourne

目标仍 active：修复技术问题，用两个既有H100节点共8卡验证真正P/D和EP/PP（含单卡）的动态分配是否有收益，交付索引Markdown报告。不要因基线完成或排队等待而结束目标。全部角色HiCache开启，GPU KV原生.95自动容量，host每rank32 decimal GB；会话/DP亲和，同episode不flush。用户已授权两个节点，不新预约、不重复问权限、不启动子agent。

## 当前只运行 E7 高负载验证

- 运行 `runs/eight_gpu_pd_ep4_hicache_retry_e7_r1`，pd_ep4，C128×3，预热及串行参考仍保留。当前加载/预热中；实时读task/progress.json。
- Guard exec **96634**，PID **235662** / start_ticks **383294605**；两个节点011与171已acquired。API25221，dashboard29501。
- CPU只读watcher exec **84595**，PID **245851**，`e7_direct_watcher_r1.json`；源码 `watch_direct_run_v2.py`。它拥有STATUS.json，只分析当前run，不混入E5/E6。
- E7 source `.engine-worktrees/agentx-hicache-restore-retry-r7-20261031`，commit **797f6488cf5fea2f88e38ccaeb1703443deec537**，tag `serving/sglang/qwen35b/agentx-hicache-retry-candidate-r7-20261031`；已冻结、不可编辑。
- 新准备脚本 `prepare_cluster_v17.py`，260项冻结哈希核对，C128配置/图/预热/副本协议与E6完全一致，GDN80、receive8、64请求/node，源E5 EP4六条32-token参考。无E5/E6性能样本导入。
- 7:14左右验证Slurm：011 job33758541仍RUNNING，余4d10h；171 job33646189余3d22h。canonical backfill_trigger已执行，supervisor live。
- 计划入口 `E7_RUN_PLAN.json`。先观察此压力路径，不要同时重启旧队列。

## E6 已失败并清理，所有旧队列和watcher均已退出

`eight_gpu_pd_ep4_checkpoint_e6_r2` 完成C8三轮，C128 r1失败。06:56四个D rank都实际触发MAMBA_CACHE_CHECKPOINT_DEFERRED并继续生成；未重现原可选快照断言。06:58:12 DP3本地HiCache恢复失败：原承诺27712个host前缀token，device_indices=0/new_indices=0，被上层报成Decode transfer failed。不是已证明的RDMA错误；旧日志没有记录rematch host覆盖。

06:59:20清理：passed/GPU idle/owned model processes exited全true，整组Guard未验收。后续控制全部门槛停止。

**不要再poll这些关闭session**：E6主5006 exit1；E6 watcher60426 exit0；控制等待72248 exit1、默认队列等待6332 exit1；对应CPU watcher等待77844/86406 exit0。更早single75146/direct56674/PP6927等也关闭。

E6控制目录仍是准备记录，不能自动启动：`eight_gpu_pp2_pd_controls_checkpoint_e6_r1`、`eight_gpu_ep2_split26_checkpoint_e6_r1`、`eight_gpu_ep4_native_queue32_checkpoint_e6_r1`。E6_RUN_PLAN已标stopped，后续需新E7目录。

## E7 修复与当前证据边界

只改DecodeRequest重试时间字段、decode_hicache_mixin状态机、一个回归测试文件；继承E6可选快照延期。

- 若重新匹配仍覆盖全部承诺前缀，但init_load_back返回0新设备页，则保持PENDING并重试，使用原生disaggregation waiting timeout（默认300秒）作有限界限。
- 缺失承诺覆盖或非零但不完整的恢复仍FAILED；H2D事件完成及跨rank传输gate仍控制进入Decode，不覆盖Prefill传入的当前GDN。
- 首次重试marker `HICACHE_LOAD_BACK_DEFERRED`，成功排队恢复marker `HICACHE_LOAD_BACK_RETRY_SUCCEEDED`；后者仍需H2D完成，不能当作已生成输出。
- `unit_r7_restore_pressure_red.log`：旧行为3失败3通过。`unit_r7_restore_pressure_green.log`：22通过（新6+前16）。测试期间只对CPU进程LD_PRELOAD新版libstdc++，未改GPU环境。
- 旧日志无法区分临时分配失败与host前缀丢失，E7会明确记录。**尚无E7 GPU压力完成证据**。若记录promised prefix lost，需检查准入到恢复之间host锁生命周期；不能无限重试或关闭HiCache。
- 细节 `DECODE_HICACHE_RESTORE_PRESSURE.md`；既有可选快照 `DECODE_CHECKPOINT_PRESSURE.md`。

## 已验收 E5 基线：18轮，三轮算术平均

| 配置/负载 | TTFT ms | 请求TPOT ms/token | 请求E2E ms | root s | 压缩wall s |
|---|---:|---:|---:|---:|---:|
| 2×EP4 C8 | 431.710 | 9.640 | 3542.973 | 277.341 | 443.421 |
| 4×PP2 C8 | 277.152 | 7.137 | 3266.097 | 275.731 | 441.653 |
| 8×单卡 C8 | 348.458 | 4.990 | 2558.706 | 244.118 | 397.420 |
| 2×EP4 C128 | 2271.319 | 73.435 | 22624.015 | 790.365 | 1168.206 |
| 4×PP2 C128 | 1188.733 | 39.419 | 13128.898 | 554.648 | 835.796 |
| 8×单卡 C128 | 27305.923 | 18.662 | 33312.386 | 1012.758 | 1449.092 |

C8/C128是根会话上限，不是GPU batch。单卡40状态，EP4/PP2每GPU完整模型等效80；PP2微批16/policy0。实际ready确认EP4、单卡原生overlap开启，PP2原生流水；DeepEP auto，不据Python hook推通信后端比例。

单卡C128 host命中平均1257，回退8。完整模型去重GPU KV：single2712832、PP2 15083904、EP4 17806080；EP4更多总容量不保证更多命中。新E6分析器增加逐实例、分P/D角色容量和8rank原生HiCache attached日志检查；三E5已验收总容量复核 `e6_cache_capacity_reconciliation.json`。

严格1000ms/10ms SLO，C128 ready-goodput EP4 .3445、single .2175、PP2 .1369；1000/15变PP2 .5946、EP4 .4214、single .2278。完整16目标网格保留，不挑目标证明动态。8个源家族请求E2E在C8/C128都PP2低于EP4，各3/3方向一致。事后标签不能作为线上可知特征。

E5commit37e0828c642ce4f3795c4c4394f4ce913bcd1a35；PP恢复层号修复27CPU、48/48真实GPU host恢复通过。E5/E6/E7不混合均值。

## 仍需完成的 bounded controls

E7高负载验收后，按已有设计准备新同源目录：
1. col_PP2、PD_PP2、P_PP2→D_EP4，C128各三轮、micro16。
2. col_EP2、PD_EP2 2P6D，C128各三轮；六D缓存owner分配已有8CPU检查，一个D tile与P同节点、另两个跨节点，保留拓扑边界。
3. col_EP4、PD_EP4，GDN160/receive32，C128各三轮。显式160预算覆盖(16+32)*3=144，避免用小接收队列否定P/D；它同时改状态预算和队列，不作单因子因果归因。

不继续无边界扩展配置搜索。固定池、同一SLO若仍互补，再实现在线请求路由并给同质池相同cache/queue策略。`cache_queue`现有常数未校准，不直接运行当成动态证明。固定权重池分流不是动态增减实例或in-place重分片；计真实迁移和缓存成本。

## 报告与科学限制

主简明Markdown `../experiment_summary/agentx_pd_dynamic_20261030/REPORT.md`，索引README；旧详细进度归档。E5 `matrix_analysis/REPORT.md` + COMPARISONS/FAMILY_TRADEOFFS/STRICT_SLO_OVERLAP等。E6 `e6_analysis/REPORT.md`完整保留失败状态。E7当前per-run分析在完整episode后生成，由watcher更新。

8个源家族171调用，C12816隔离副本2736调用；不增加多样性，输出均值435 vs全源860，嵌套比26% vs58%。输入为hash恢复前缀关系，非语义文本，不能保证专家路由相同。

V3全局空闲压缩超10秒，原trace含约40h间隔，分支返回触发与官方不同；wall/root/goodput是派生压缩回放，不是生产时间或官方分数。V4独立分支时间修正已CPU准备9通过/24图，无GPU测量；匹配控制后根据关键结论决定独立敏感性，不能静默混入V3。

输出一致程度 `output_agreement_e5/OUTPUT_AGREEMENT.md` 已15paired episodes/17955calls；受控串行/host恢复一致不等于动态batch逐token全一致，冷请求也有首token差异，无logits因果归因。

最后还需完整P/D、强固定和条件动态收益验证、证据审计、canonical experiment-manager归档和索引报告。当前是新问题已定位与修复验证的progress，不是blocked，不要update_goal complete。
