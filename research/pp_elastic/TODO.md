# PP 原地伸缩＋HiCache：实现与实验清单

更新：2026-11-10。本页是本项目的统一进度入口；任务完成后更新这里，不另外维护相互冲突的清单。勾选表示该条写明的范围已有证据，并不表示全部功能已验证。

**当前决定。** 保留已完成有限正确性及单次 overhead 观测的同步基线；下一阶段联合研究稳定地址的 KV 池扩缩、有限 CUDA Graph 预捕获与持续 KV patch，目标是减少推理干扰。2026-11-09 已完成独立调研；用户随后要求本轮由主 agent 自己实施，不启动 sub-agent。实施前先固定共同源码、兼容依赖、层分组和状态地址约束；主协调器维护项目分支与本页。当前实现与本次验收汇总见[基线状态](BASELINE_ACCEPTANCE_20261109.md)。

## 本轮实施：主 agent 顺序完成

- [x] 接入真实 PP 前向与标准 token／paged allocator，保护复制中的槽位复用与重写。仅限 native／eager／background＋overlap 的在线 KV CPU shadow；最终 fence 补齐余量后复用跨 rank 传输，GDN／HiCache仍在fence交接。512项CPU检查全过；冻结 `e7c9cb1264a9` 的A100 64-token页PP2→PP3→PP2通过同暖512边界连续性和CPU前缀loadback。扩容两个源rank各在线完成662行，缩容源rank完成659行；它们是各rank的槽位行计数，不是全局输入token数。整轮含清理171.795秒，正常release。r1／r2失败保留于[验证记录](VALIDATION.md)。不据此宣称网络 overlap、Graph 接入或性能收益。

从冻结基线 `ce563696401c` 建立 `codex/pp-kv-live-foundation-20261109`，本轮不使用子 agent。[Draft PR #4](https://github.com/paprikaw/sglang/pull/4) 已推送，目标为项目分支，尚未合入。底层原语GPU冻结源码为 `8d7c21241c`，本轮native在线shadow为 `e7c9cb1264a9`；两条执行路径分开验收。旧分工仅表示模块边界，不是正在运行的任务。

- [x] 实现 KVCacheD 按层映射协议、映射账本、部分失败重试及页面 pin；保存可重放依赖补丁、Python／原生版本检查和 stacking 边界检查。27 项依赖 CPU 检查通过；补丁在干净上游索引重放得到完全相同的源码树。见[实现与边界](KV_LAYER_MAPPING.md)及依赖锁。真实 GPU 页面验证不包含在此勾选中。
- [x] 修复 SGLang import／abort／cleanup 在校验 generation 前触碰物理页面的问题，并在 map 前检查槽位和 payload。旧基线可复现 3 个故障，新代码 769 项引擎 CPU 回归通过；保留层地址／数值仅在 CPU 适配器验证，不能据此推定真实 GPU 验收。
- [ ] 接通 HiCache、GDN 和 Attention 的完整布局约束，并验证实际 OOM／迁移失败回退。底层 GPU 的 map／unmap、保留地址和数据及页面 pin 拒绝已通过；这不替代整模型新后端兼容性或归还工作 GPU，现有 hybrid HiCache 的 native-only 保护不撤除。
- [x] 建立有限 Graph 预捕获库与持续 KV 行追补的实验模块，本轮75项CPU检查通过（25项新模块＋50项生命周期／参考资格回归）。Graph按完整tensor地址／shape／stride匹配，eager回退同样保护页面；整模型Graph尚未接入。Patch的native CPU shadow已在本轮接入scheduler，在线跨rank传输和GDN增量仍待实现。见[接口与接入顺序](GRAPH_KV_PATCH.md)。
- [x] A100 `gpu-r1` 完成底层 GPU 检查：退役组驱动可见显存增加8 MiB，保留地址和数据；2次 Graph capture／2次 replay、1次地址变化 fallback及1次未映射拒绝；patch为2轮／4行，包含新增行及槽位复用。它是小规模单 GPU 检查，尚未证明完整模型、跨 rank 或在线复制的性能。
- [x] 同一租约内，开启 HiCache 的 native KV 整模型 PP2→PP3→PP2 往返通过同暖512边界的生成连续性和 CPU 前缀 loadback；整轮含清理186.509秒，Guard postcheck、清理及正常release通过。Graph在整模型中仍关闭，cold/warm等价和转换期间GPU归还未通过资格。见[验证说明](VALIDATION.md)。

## 当前：先把基线做正确

- [x] 整理 EP／PP／P/D／HiCache 共同代码基线，完成有限范围的静态 CPU 与 A100 回归。证据见 [VALIDATION.md](VALIDATION.md)；这些结果不替代动态转换验证。
- [x] 实现 resident PP 层布局往返、活跃 KV／GDN／请求进度及 CPU 前缀的转移。已有 PP2→PP3→PP2 同一暖缓存边界的连续性检查通过（r11，`c0a1a1ecb0de`）。这不证明冷／暖缓存生成等价，也不证明空阶段进程退出、GPU 已归还。
- [x] 实现三种基础策略：同步暂停、后台预搬不可变权重、暂停期间分块复制。默认基线的KV／GDN仍在最终 fence 同步；新opt-in CPU shadow的接入与验收单列在本轮条目，分块复制本身不等于在线增量同步。
- [x] 实现 CPU spill／目标池重建路径，以及 GDN 引用与清理流程修复；相关 CPU 检查及独立审查已有记录。CPU spill 验收版本为 `5fa14eae1df0`，见 [Runtime PR #2](https://github.com/paprikaw/sglang/pull/2)。
- [x] 验收同步 CPU spill 在真实 GPU 上的成功往返：r14（`5fa14eae1df0`）观察到源 KV／GDN backing 释放、目标恢复、同暖缓存边界的生成连续性及切换后的 CPU 前缀实际 loadback，清理和租约正常结束；完整监督时长 163.294 秒。独立审计 `independent_r14_audit.json`（SHA `6afd282252d8`）通过；原始结果 SHA `ff753f536a37e`。这不包含冷／暖生成等价、严格峰值比较、性能收益或转换期间归还工作 GPU。
- [ ] **B 运行、独立审查：**补齐受影响故障路径的实际 GPU 回退验证；已有 CPU 回退检查不替代真实 OOM／迁移失败场景。background、batched 的 GPU 资格也不能由 sync 的成功推定。
- [x] **主 agent：**Runtime PR #2 已以 `3a72d5781835` 合入项目分支，Cache PR #3 已在此前合入；候选组合742项CPU检查通过，独立审查未发现阻止有界 naive 基线合入的问题。引擎、测试及适配 tree 与已测候选一致。见[本次验收摘要](BASELINE_ACCEPTANCE_20261109.md)。
- [x] **主 agent：**总 PR #1 已以 `5542d19d90d8` 合入共同基线 `codex/pp-elastic-integration`；主线合并后29项定向CPU检查再次全过。冻结标签为 `baseline/sglang/pp-hicache-naive-20261109`，最终提交与远端核对保存在本次验收产物；引擎、测试及适配与742项检查的候选一致。未完成的低干扰机制与硬件资格继续保留未勾选。

## 当前：测清楚转换 overhead

- [x] 完成一次最小配对 overhead 观测：r15（`8374b0276d19`）在同暖512边界、输入及硬件下，每方向各1条无转换参考与1次 sync/cpu_spill 转换；完整监督186.775秒，正常清理／release。扩／缩 E2E 差分别 +4570.757／+3215.168 ms，各 rank 平均 fence 3389.554／3356.288 ms（非重复实验平均）。差值含目标拓扑速度及控制影响，TTFT在触发前；不作为孤立切换成本或统计性能结论。报告 `gpu-hicache-r15/REPORT.md`，原始结果 SHA `25dea5a4d093`。
- [ ] **B 运行、独立审查：**分别记录总转换时间、推理暂停／token 间隔、权重准备、KV／GDN 备份与恢复、传输字节、峰值 GPU／CPU 内存及请求 TTFT／TPOT。报告绝对 ms 和可比重复的平均值，保留原始样本；未分离的阶段标明未测，不由总时长倒推出原因。 r15已有E2E／TTFT／平均TPOT、各rank coordinator prepare／fence／total及host字节；隔离的权重／传输／恢复耗时、最大token gap、严格峰值和可比重复平均仍未测，因此本条保留未完成。
- [ ] **B 运行、独立审查：**在共同正确性条件下比较 sync、background、batched，说明后台准备隐藏了哪些开销。启动／预热和转换期间的开销分开报告，完整运行时长另列；正确性 smoke 的单次耗时不直接当作稳定性能收益。

## 已完成调研：下一阶段的接口与机制选择

- [x] 主 agent 核对 workbench vLLM/PipeLive、SGLang runtime `8374b0276d19` 及 KVCacheD 工作区，形成研究库 `LOW_INTERRUPTION_PP_REVIEW_20261109.md` 和问题簿 Q25。建议优先 KVCacheD 稳定虚拟地址，保留旧块指针内核作备选；发现按层物理映射 API 不匹配、GDN backing 重建和 stacking 边界限制。仅完成调研，无新增引擎改动或 GPU 验收。
- [ ] **A：**固定兼容的 KVCacheD 依赖与能力检查，先验证物理页映射／回收和存活状态；只改 local view 不算释放显存。
- [ ] **B：**将预捕获库接入整模型执行器，补齐权重、KV、GDN、输入和 workspace 的完整 binding 及稳定地址；底层库的实际 replay／fallback 已在 A100 检查，不等于模型 Graph 已可切换。不在正常推理期间无保护 capture。
- [ ] **B：**预初始化并预热候选布局的真实通信路径，验证路由版本、通信顺序和缓冲生命周期。
- [ ] **A：**选择固定且满足 attention 池层序号边界的 stacking 粒度。显式 preclaim 须与层所有权一致，并计入物理显存；不默认打开全局自动预分配。

## 下一步一：持续 KV patch，缩短推理中断

- [ ] **主 agent：**以旧GPU→目标GPU的后台直传作为活跃KV主路线，CPU shadow保留为对照，CPU HiCache保留历史前缀及恢复能力。先准备目标状态空间、预热实际通信边，再实现有界基础复制和持续增量；记录实际P2P／网络后端，跨节点RDMA需运行证据。见[实施路线](GRAPH_KV_PATCH.md)。
- [ ] **主 agent：**目标在最终暂停前实际应用非空数据并确认；覆盖请求新增／结束、KV槽位复用和改写、迟到包、取消、缓冲回收及正常PP通信顺序。追不平时限速／推迟；提交前失败保留源，不承诺提交后任意回滚。
- [ ] **主 agent：**最后只排空在途批次、补齐剩余KV和最新GDN／请求进度，再共同发布布局；HiCache保持开启，CPU-only前缀不全部恢复GPU。把完整模型Graph选择及稳定地址容量机制接入同一交接流程，代码、GPU正确性和性能分别验收。
- [ ] **主 agent：**用相同输入、缓存、源／目标及资源条件，对照同步基线和CPU shadow，记录逐token间隔、平均TPOT、尾部停顿、整请求耗时、搬运字节及峰值。必须证明准备期间继续生成、切换后连续输出和实际暂停缩短；单次正确性或脚本总时长不作为性能结论。

## 下一步二：GPU KV 池扩缩，降低转换显存需求

- [ ] **A：**核查并选取可行实现：复用 KVCacheD／VMM，或移植此前支持非连续 GPU KV 块寻址的内核／分配器。先检查与 SGLang Attention、HiCache 和 GDN 的兼容性，不要求同时实现两套。
- [ ] **A：**实现池容量与层归属变化时的按需增减和状态保留，明确 active KV、可回收缓存及 CPU 备份的处理方式。当前 native tensor pool 的重建／搬迁是对照；paged token 索引分配不等于物理显存在线扩缩。
- [ ] **B 运行、独立审查；A 提供物理账本：**测峰值显存、实际释放字节、复制量、转换时间和稳态 Attention 性能，检查非连续寻址或 VMM 的额外成本。先分别评估内存机制与 KV patch，再验证组合收益。

## 后续应用

- [ ] 真正退出不再需要的 worker 并归还 GPU；当前 resident 空阶段不计为释放资源。
- [ ] 接入 P／D 独立伸缩和状态交接，再以短动态负载验证 GPU 时间与延迟收益。复杂控制、更多模型和更多布局按前述结果追加。

**更新约定。** 每个 agent 交付前更新自己的任务条目，写明结果、版本／证据和剩余限制。子分支修改本文件后随 PR 提交；尚未含本文件或由单一负责人写入时，交付具体条目补丁，由主协调器顺序落盘。失败也要更新，未通过验收不打勾。不覆盖他人的条目，不为本页更新而重复跑已有效的检查。新增任务先落在本页，再派发；详细日志留在实验归档中。
