# Graph 预捕获与持续 KV patch

更新2026-11-10。由主 agent 顺序实现，属于 [TODO](TODO.md) 的低干扰阶段，接在[按层物理映射](KV_LAYER_MAPPING.md)之后。代码在 Draft PR #4；新接入范围与验证状态见下文。

**下一阶段的主路线：活跃缓存从旧 GPU 后台直传目标 GPU，CPU 保留历史缓存与恢复能力。** 用户随后提出把迁移与推理的NCCL分开；本轮调研建议改用独立NIXL＋UCX迁移通道，推理保留NCCL，先通过固定GPU双缓冲传输，详见[NIXL方案](NIXL_KV_TRANSFER_DESIGN.md)。它取代前一建议中“先接NCCL迁移”的实施顺序，本轮已接入可选运行路径，native／eager的有限GPU资格见下文。同节点核实实际P2P链路，跨节点核实GPUDirect RDMA；换库不消除显存／链路争用。当前已通过的CPU shadow保留为正确性对照，不作为活跃KV的必经中转。

**先准备，再边生成边搬。** 服务前预初始化并预热允许切换的通信边和有限缓冲；转换时在后台准备新增层和目标状态空间，保留层尽量不动。空间不足就推迟或有选择地使用CPU，不能靠重建整个源池完成低干扰路径。源端在每次真实写完成后，把需要迁移的已有KV和后续新增／改写的行分块发送；目标写入自己的待启用状态空间后确认。NIXL数据面与正常PP的NCCL独立，但仍使用有界缓冲、发送预算、CUDA完成事件及应用确认，处理进度、版本和内存生命周期。请求结束、槽位复用、旧包及取消均受保护。

**最后只补差额，再一起接班。** 暂停接收新批次、排完在途流水后，补齐最后 KV、最新 GDN／请求进度，确认目标已应用，再共同发布新布局。CPU-only 前缀继续在 CPU 保留／异步交接，不为转换全部恢复到 GPU；未完成的 HiCache I/O 纳入交接。提交前失败保留源，提交后按新代处理。缩短这段暂停是目标，不承诺零中断。完整模型 Graph 采用服务前有限预捕获、切换时检查并选择的路线；动态容量优先使用已规划的稳定地址机制，二者未接通前明确保留资格限制。

**完成标准看回答是否被打扰。** 同输入、缓存、源／目标布局和资源条件下，与同步基线及当前 CPU shadow 比较；先验证连续输出、缓存恢复、失败处理，再测后台搬运时及最终交接时的逐 token 间隔、平均 TPOT、尾部停顿、整请求耗时、传输量及过渡峰值。源布局准备期间必须继续真实生成，目标必须在最后暂停前已有实际应用的非空数据；把基础复制后的新增／改写和槽位复用覆盖进去。通信初始化、Graph 准备和模型加载单独计时，不能用整轮脚本耗时充当推理暂停。

**真实请求接入：先做在线 CPU shadow。** `--enable-elastic-pp-kv-patch` 仅允许 native KV、标准 token／paged allocator、eager、无 speculative 的 background／overlap 路径。旧布局继续前向时，以该次实际 request→token 映射和完成事件发布已使用的 KV 行，独立 stream 每轮最多256行备份到 CPU。分配器记录新的槽位 epoch；分页路径追踪完整页的分配／释放，但不把未写的页尾当作已完成的前缀。物理 free 或重写已发布行前等待正在复制的行，free group 仍按原协议收尾。HiCache 已分配但尚未被前向使用的行不提前发布；最后排空、退休 HiCache I/O 后补齐完整 live set，再通过已有跨 rank 协议交接。GDN 和 CPU 前缀索引也仍在 fence 同步。CPU shadow 额外内存单独准入并记录字节；复制失败先 drain，再拒绝本次转换，旧布局保留。

**这一步验证什么。** 它连接了真实前向、allocator、增量复制和最终网络交接，但网络本身尚未提前发送，也没有启用整模型 Graph 或 VMM 后端。计数区分服务期间已完成的行与 fence 内追补的总行；真实 GPU 必须观察非空在线轮次并通过生成连续性，才能勾选该范围。当前不能据此声称缩短了暂停；后续再把在线影子复制改为直接向目标发送，避免最终 H2D 回传。

**2026-11-10接入验收。** `e7c9cb1264a9` 的512项CPU检查全过，包含后台构建线程隔离、真实token／paged allocator生命周期和Dynamo backend切换检查。A100 native／HiCache／64-token页PP2→PP3→PP2通过同暖512边界连续性与CPU前缀loadback；扩容源rank各在线完成662行，缩容源rank完成659行，整轮含清理171.795秒，正常release。仅此CPU shadow路径取得有界整模型资格；它不扩大下面Graph／VMM原语的资格。失败尝试、source tag与结果哈希见[VALIDATION](VALIDATION.md)。

**Graph：先准备少量布局，再选择。** `graph_bank.py` 在启动或明确排空的边界捕获，激活后禁止追加 capture。布局、batch、tensor 地址、shape、stride、dtype 和 device 全部匹配才 replay；其余情况统计 eager fallback。调用方必须登记权重、KV／GDN、输入、输出及 workspace，不能只检查 KV。页面未映射时拒绝执行；replay 和 eager 都把页面 pin 保留到完成事件。捕获失败或执行失败不能直接再跑一次，以免重复写入状态。当前模块没有替换 SGLang 的 `CudaGraphRunner`，真实通信预热、完整 binding 清单和 GDN 稳定地址仍待接入。

**Patch 的底层协议。** `kv_patch.py` 维护迁移 generation 和每个 token 槽位的 incarnation，源写入记录完成事件后发布不可变 KV 行；独立 CUDA stream 等待事件并分批复制。复制期间允许新增其他行，被复制的槽位必须延后释放／复用，VMM copier 的源和目标物理页也保持 pin。复用的槽位形成新 incarnation，必须重新复制；最终 fence 停止源写入并有界追平。失败时先排空已提交复制，再解除保护。初始 Tensor copier 是单 GPU、按 stride 复制的实现；native CPU shadow 接入见上文；本轮新增NIXL跨rank在线copier，GDN仍在最后边界交接。

**真实 GPU 验证分两部分。** `hicache_transition_smoke.py --kv-foundation-root … --supervised-lifecycle` 在同一受监督 A100 租约内，先用第四张卡运行独立 VMM／Graph／Patch 检查，再用前三张卡运行 HiCache 整模型 PP2→PP3→PP2 基线。前者检查实际显存释放、保留地址／数值、真实 replay、fallback 和非空增量；后者继续使用 native KV 后端。二者成功不等于 KVCacheD 已接通 hybrid HiCache，也不等于整模型持续 patch 已通过。失败和清理记录均保留，整轮包括清理最多五分钟。

**接入顺序。** 先完成上述 GPU 检查，再把行生命周期钩子接到实际 allocator 与 KV 写入，随后接跨 rank transport；Graph 侧先准备真实执行布局和完整 tensor bindings。只有真实请求能在基础复制期间继续生成、最终交接后 token 与状态一致，才比较暂停和端到端性能。保留同步基线用于回退，不预设新机制一定更快。GPU 结果及精确版本写入 [VALIDATION](VALIDATION.md)。

**本轮结果。** 75项CPU检查通过。冻结 `8d7c21241c` 的 A100 `gpu-r1` 检查通过：物理页回收8 MiB，保留地址／数值；2次capture／2次replay，实际触发fallback和未映射拒绝；patch为2轮／4行，新增行和槽位复用检查通过。随后 native HiCache 整模型往返也通过同暖缓存连续性；全程186.509秒并正常清理／release。以上仍不包括完整模型的 Graph 复用、持续 patch 或性能收益。

**NIXL 在线路径的有限验收（2026-11-10）。** 候选GPU启动时建立并预热agent，native／eager／HiCache的35B PP2→PP3→PP2在`152de4332f`通过同暖512缓存边界的连续输出及CPU前缀loadback。目标在最后暂停前确已应用KV；每卡固定注册64 MiB，同节点UCX日志选择CUDA IPC。整轮含清理176.008秒。487项CPU回归及后续14项NIXL定向检查分别记录，不累加；首轮类型不匹配失败也保留。见[验证说明](VALIDATION.md)。GDN／HiCache仍在最后边界交接，Graph／VMM组合及暂停收益尚未验收。
