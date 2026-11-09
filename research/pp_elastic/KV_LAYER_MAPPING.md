# KV 按层映射：低干扰伸缩的第一步

2026-11-09。对应[统一清单](TODO.md)的 KV 池与依赖项。本轮由主 agent 直接实现，不使用子 agent；起点为 naive 基线 `ce563696401c`。

**这一步解决什么。** 原适配器检查的 KVCacheD 按层 API 在当前依赖里不存在，因而可能退回全层物理映射。本轮补齐 ownership filter、目标页映射、整组回收和映射账本，并把 Python API 与原生 ABI 一起检查。不匹配时在创建缓存池之前报错，不把全模型地址预留当作按阶段节省了显存。

**怎样保持数据。** allocator 只为当前拥有的层组映射页面；目标迁移可以显式映射新增层组，但 prepare 不改变旧布局。重复 map 跳过已有页面，避免清零正在使用的数据；部分失败后保留已经成功的操作，重试只处理剩余部分。显式迁移映射在返回前等待新页初始化完成；普通 allocator／listener 的 map 不增加全设备同步，避免阻塞正在等待它的通信。回收检查页面 pin，并保守同步 CUDA；generation 校验在物理 map／unmap 之前。当前仍在 scheduler 的安全边界操作，不宣称已经能在任意在途 GPU 工作旁在线回收。

**页面 pin 的边界。** 新 API 固定的是物理页，尚未接入 HiCache／持续 patch 的完整生命周期。它不固定请求槽位，也不代表 GPU 写入完成；后续调用者还须保护槽位复用、等待写完成与复制完成事件。GDN 仍使用现有同步迁移，权重和 Graph 地址合同也没有随这一步自动完成。

**支持哪些层组。** 首先支持一层一个物理组，并保留固定 stacking 的几何检查。35B 的 40 个模型层只有 10 个 full-attention 层；PP2 的 attention 层数为 5／5，PP3 为 3／3／4。不能按 40 个模型层检查对齐后就启用两层或四层 stacking。非法起始范围在预留显存前拒绝，目标范围在 prepare 前检查。第一版不在线改变 stacking 粒度。

**依赖怎样复现。** 原第三方脏工作区保持不动。兼容依赖在独立 worktree 中整理，从上游 `3cc92bec02fc77b72b86ed6f8d91dc454bbca154` 开始；新增协议和必要的原生层组实现以本目录 `dependencies/kvcached-layer-mapping-v1.patch` 保存，路径、内容哈希及 ABI 记录在 [dependencies.lock.json](dependencies.lock.json)。在干净的相应上游 checkout 中执行 `git apply`，再在预约节点构建 `python setup.py build_ext --inplace`；用 `PYTHONPATH` 显式选择它，不覆盖其他实验的安装。构建和加载的二进制哈希另存实验产物。

**验收怎样判断。** CPU 检查覆盖部分失败、重复包、pin、非法布局及过期控制消息，另用旧基线复现回收顺序问题。原生扩展构建、真实加载、GPU 页面数变化和模型端到端连续性是不同检查。新后端的 HiCache、packed stride 与 Attention 路径、GDN 稳定地址、Graph、持续 patch 和物理 GPU 释放仍要分别验收；现有 hybrid HiCache 的 native-only 限制继续保留。最新完成情况只在 [TODO](TODO.md) 勾选，未通过的项保持未完成。

**这轮已完成。** 769 项引擎 CPU 回归及 27 项依赖协议／几何检查通过；3 个新增故障用例在旧基线均失败。依赖补丁重放得到源码树 `14c85a5f0701`，SHA256 为 `dbd6d41eaf8c603bc6a807a05b55482007b212a0a3cdbe53e6c59f2a2e8f2d75`。原生扩展已经构建并在隐藏所有 CUDA 设备的条件下验证 ABI 加载，尚无本轮 GPU 分配或模型运行验收。详细检查口径见 [VALIDATION](VALIDATION.md)。

**下一步先检查什么。** 先以最小原生 GPU 探针核对物理页变化与保留值，再接缓存和 Attention。packed K／V view 的行 stride 与部分旧 pointer table／拷贝路径的紧密排列假设需要逐一核对。当前 worker 映射账本以一个新进程的一次池生命周期为界，不支持在同进程随意替换过滤器或重建原生池。
