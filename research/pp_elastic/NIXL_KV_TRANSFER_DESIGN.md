# 用 NIXL 搬缓存，让 PP 伸缩少打扰推理

2026-11-10。调研基于本任务源码 `0860fb7576`、NIXL 官方文档及上游源码快照 `82b356ce0a29`。用户已确认候选 GPU 启动时预初始化；本轮已接入并取得native／eager／HiCache的有限A100正确性资格；进度统一写入 [TODO](TODO.md)，对应 [Graph／持续补丁设计](GRAPH_KV_PATCH.md)和论文实验 E3。

**结论：建议推理继续使用 NCCL，缓存迁移改用独立的 NIXL＋UCX 通道。** NIXL 提供内存注册、异步 READ／WRITE、完成状态与通知，UCX 支持 GPU／CPU 内存及 RDMA 网络。因此可把 KV 数据搬运从正常 PP 的 NCCL 调用顺序中分离。这是基于接口和当前代码的机制判断，还不是本项目已验证的并发收益；两条通道仍共享 GPU、内存及链路资源。[NIXL 架构](https://docs.nvidia.com/nixl/getting-started/architecture/)、[UCX 后端](https://docs.nvidia.com/nixl/user-guide/backend-selection/ucx/)

**第一版只选 NIXL＋UCX。** 同节点核实 CUDA IPC／实际 GPU P2P 路径，跨节点核实 GPUDirect RDMA；UCX 可能使用 host staging 或 TCP，因此“能传”不代表“GPU 直传／RDMA 已通过”。保留现有同步／CPU shadow 对照。Mooncake Transfer Engine 也是候选，现有 SGLang 同时有 Mooncake 与 NIXL 实现；只有 UCX 在目标节点出现具体兼容问题时才评估替换，不同时实现多套。NIXL 也有 Mooncake 和 UCCL-P2P 插件，但并不是第一版的前置依赖。[UCX GPU 与协议核查](https://openucx.readthedocs.io/en/master/faq.html#working-with-gpu)、[SGLang P/D 支持](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/pd_disaggregation.mdx)、[NIXL Mooncake](https://docs.nvidia.com/nixl/user-guide/backend-selection/mooncake/)

**先复用接口经验，不照搬整个 P/D 管理器。** 本地 [NIXL P/D 实现](../../python/sglang/srt/disaggregation/nixl/conn.py)已有 VRAM／DRAM 注册、严格线程同步、WRITE、完成轮询、通知、PP 层映射、GDN 状态和 GPU staging 的代码。弹性 PP 则是移动一部分层的持续状态，需单独实现 migration transport，复用 [现有 KV 生命周期](../../python/sglang/srt/elastic_pp/live_kv_patch.py)、[增量版本协议](../../python/sglang/srt/elastic_pp/kv_patch.py)和 [coordinator](../../python/sglang/srt/elastic_pp/coordinator.py)；不能把静态 P→D 一次交接直接当作在线 PP 重配置。调研时锁定 Python 环境未安装 NIXL；本轮另建依赖覆盖目录，不改共享 serving 环境，实际版本和哈希见依赖锁。

**用户确认的启动范围。** 2026-11-10 确认：首版在所有候选 GPU 的 worker 启动时建立 NIXL agent、注册固定双缓冲、交换元数据并预热可迁移通道；切换时复用。只预留小缓冲，不提前建立每种布局的完整 KV 池。候选 rank 常驻，不能将其算作已经归还 GPU；按需启动／退出端点留到后续实际资源归还阶段。

**首版数据路线：KV 池→固定 GPU 发送缓冲→NIXL／UCX→固定 GPU 接收缓冲→目标 KV 池。** 启动时注册少量有界 GPU 双缓冲、交换端点 metadata 并预热实际通道。CPU 线程处理通知和提交，不把大块活跃 KV 先备份到 CPU 再上传；CPU HiCache 继续保存历史前缀和恢复状态。选择 GPU 缓冲的原因是 KVCacheD 将来会 map／unmap 物理页，稳定虚拟地址不等于网络注册永远有效。先用与 KV 池解耦的稳定缓冲把链路跑通，代价是额外 GPU 内部 gather／scatter 和有限缓冲显存；连续且生命周期稳定的区域后续再尝试直接地址列表传输，并测是否值得。真实 GPU 直传仍以链路资格为前提。[NIXL 注册与异步操作](https://docs.nvidia.com/nixl/getting-started/quick-start/)

**在线复制怎样工作。** 每个 worker 使用独立迁移 agent，由单一迁移任务循环管理，显式配置线程安全和 progress；零层 rank 也必须推进接收，不能等它参与 forward。源端沿用真实 forward 完成事件，只打包已写好的存活 KV 行；gather 完成前固定所读槽位／页，网络未完成前固定发送缓冲。目标提前授予空缓冲，收到通知后校验迁移代、层号、槽位 incarnation、补丁序号和缓冲轮次，再 scatter 到尚未启用的目标空间，等 CUDA 写入完成后回应用确认。缓冲只有确认后才能复用；接收旧包不得覆盖新状态。初版使用 `initialize_xfer`／`transfer`／`check_xfer_state`，再按需要用 prepared descriptors 降低描述符开销，不为每个 token 重新注册。[已核对的上游 Python API](https://github.com/ai-dynamo/nixl/blob/82b356ce0a29b5b749d8b8720caceec6c844cd03/src/api/python/_api.py)

**NIXL 不代替 CUDA 的读写依赖。** 源端必须先确认 GPU 打包完成再发起外部传输，目标必须确认数据到达并经过正确的 CUDA 提交／可见性边界后才能读；避免每轮全设备同步，使用所需缓冲的完成事件。网络 DONE、收到通知、目标 scatter 完成和全实例可切换是不同条件。NVIDIA 明确说明 GPUDirect RDMA 的内存顺序需要 CUDA 同步／提交 API 保证，因此不能让正在运行的 kernel 直接消费尚在写入的目标区。[GPUDirect 内存顺序](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html#synchronization-and-memory-ordering)

**切换时只收尾，仍允许短暂停顿。** 源布局在基础复制和追补期间继续生成；剩余数据及状态已可在预算内交接时，再停止新批次、排完在途流水、补齐最后 KV 和最新 GDN／请求进度。GDN 每步覆写，首版在最后边界传最新快照；保留层状态尽量不重建。HiCache 开启，CPU-only 前缀按缓存版本保留／异步转移，不为迁移全量恢复 GPU；在途 backup／loadback 必须纳入完成条件。目标完成确认后共同发布新布局，提交前失败保留源。限速、积压上限和超时均需明确，追不平就推迟／有界取消，而不是无限追补。

**与 Graph 和 KV resize 的关系。** Graph 仍采用服务前为有限布局准备、切换时检查并选择的路线；NIXL 不负责捕获或更新模型 Graph。固定 GPU 网络缓冲先独立于物理 KV 页管理；采用直接池注册后，必须按物理映射／allocation 版本维护注册，禁止注册未映射范围或在传输中 unmap，回收前先 drain、撤销远端资格并 deregister。本地 KVCacheD 的 vLLM NIXL 兼容补丁只处理 layout／block 数等初始化问题，不能作为 SGLang 动态映射及注册生命周期已兼容的证据。

**实施顺序。** 先在隔离依赖环境固定 NIXL、UCX、CUDA 和 Python ABI，验证可用插件、GPU 内存注册及目标链路。再做小规模并发检查：正常 NCCL／GPU 计算继续运行时，NIXL 连续传基础数据与更新，检查源新写、目标应用、双缓冲复用、迟到通知及取消。随后接真实模型 HiCache PP2→PP3→PP2，先沿用 native KV／eager 验证持续生成与状态连续性；再接入已规划的稳定地址池和有限 Graph，不把 native 通过扩大为完整后端通过。

**端到端验收。** 同输入、缓存、源／目标布局和资源条件，对照同步版、当前 CPU shadow 版和 NIXL 在线版；加入无转换参考，分别观察后台传输期间与最终交接期间的逐 token 间隔、平均 TPOT、尾部停顿、请求总时长、真实传输字节、额外显存和积压。必须在最后暂停前观察到目标实际应用非空数据，同时源仍在生成；覆盖基础复制后的新写、槽位复用、CPU 前缀恢复与失败清理。若要归因于 NIXL 本身，再固定同一补丁协议做 NCCL 传输对照，不能把协议变化的收益都归于换库。GPU 短检查继续使用 Experiment Guard 和已有预约，每轮含清理不超过五分钟。

**本轮实现。** 已隔离安装并锁定 NIXL／nixl-cu12 1.5.2、原生绑定和 UCX 插件。`nixl_transport.py` 在所有候选 rank 启动时建立 agent、注册固定缓冲并执行双向写入／应用确认；`nixl_kv_patch.py` 接入真实 allocator、目标待启用层和最终完整性检查。三候选 rank 每卡注册64 MiB，单个数据片最多8 MiB。由 `--enable-elastic-pp-kv-patch --elastic-pp-kv-patch-transport nixl` 显式选择；CPU shadow仍是默认对照，native／eager／background＋overlap限制保留。依赖与完整模型GPU资格分开记录；测试进度见[TODO](TODO.md)。

**尚未证明。** 上游源码快照用于调研，实际运行依赖以锁文件为准。源码`152de4332f`在A100通过注册预热、目标在线应用及PP2→PP3→PP2同暖512缓存边界的生成连续性；扩／缩目标在线行计数为1320／1318，完整监督含清理176.008秒。见[验收报告](/data/gpfs/projects/punim2715/vllm_workbench/artifacts/source_integration/sglang-pp-elastic-20261105/kv-live-foundation-20261109/nixl-gpu-r2/REPORT.md)。这只覆盖native／eager，暂停缩短、跨节点RDMA、完整模型Graph／VMM组合和转换期间GPU归还仍待验证。
