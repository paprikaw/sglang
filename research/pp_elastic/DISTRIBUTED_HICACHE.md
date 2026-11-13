# 同节点 PP 伸缩：CPU 缓存原位复用与异步交接

2026-11-13。原实现核对基于 `05d823897a`；已有 PR #10 随后合入项目。本文保留设计背景，首版实现已进入独立任务分支，见[共享池接入说明](SHARED_HOST_CACHE.md)；尚无新 GPU 验收，进度见[统一清单](TODO.md)。

**用户澄清。** 历史缓存仍按新的层分配完成归属交接，只是 CPU 整理／同步可以独立于实例恢复推理；不以长期向旧实例按需拉取为主要机制。同节点若复用同一份 CPU 底层存储，则连历史数据复制也可以避免，主要更新访问映射和控制权。跨节点或确需重新排布时，才后台按新布局搬运。

**哪些位置变化。** GPU 上正在执行的层及活跃 KV／GDN 随 PP 分层迁移；CPU 中对应层的历史数据可以原位保留。例如第14层由 GPU0 改在 GPU1 执行，CPU 的第14层缓存仍在原共享存储，由 GPU1 恢复和备份。层的执行归属明确，并不表示物理副本只能一份：HiCache 本来允许 CPU 与 GPU 同时留存缓存。必须明确当前版本和写入权限。

**当前怎样做。** 现在不是把整个 CPU HiCache 恢复到新 GPU，而是按新的全局层归属，把有效 CPU 缓存行复制到新布局的 CPU 池。已有后台快照／预搬，但最终仍排空 HiCache I/O，核验当前内容、同步变化并挂接完整目标池；发布后退休旧池。新请求随后从自己的 CPU 池恢复前缀，没有向旧池按需取数的路径。见 `hicache_transition.py` 的 `snapshot_at_fence`、`commit`、`release_retired`，及[当前实现](HICACHE_BACKGROUND.md)。

**为什么当前要复制。** `MHATokenToKVPoolHost` 从 `device_pool.layer_num` 决定 CPU 池的层数、shape 和 stride；`_prepare_target_host_cache` 投影目标 GPU 层集合后重新调用 `init_hicache`。`host_layer_views` 已能无复制地表示某些按层视图，但尚未把它接成跨 worker 的共享池读写和回收协议。因此复制来自当前池与本地执行层的绑定，不是同节点伸缩的必然要求。

**为什么值得考虑。** 最新有界缩容实验最后暂停约87 ms，其中源rank的 CPU 快照／内容核验约39.8 ms；后台虽然提前搬完稳定 CPU 数据，仍要检查整个受影响历史集合。缓存变大时，这种依赖可能更重，但目前只有小规模单轮证据，不能直接预测改后暂停或按容量线性外推。

**已有后端能用多少。** SGLang 的 L2 是实例私有 CPU 内存，跨实例共享走配置好的 L3。NIXL 支持 DRAM／VRAM 传输，但本仓库的 `storage/nixl` connector 主要连接 CPU 池与文件／对象存储，并不自动提供旧 worker CPU 池的远程查找。当前弹性 PP 还在 `assert_host_quiescent` 中明确拒绝 `enable_storage`。因此不能仅加 `--hicache-storage-backend nixl` 就完成本方案，仍需处理动态层归属和旧缓存生命周期。[SGLang 官方设计](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hicache_design.mdx)，[NIXL API](https://github.com/ai-dynamo/nixl/blob/main/src/api/python/_api.py)。

**最小接入范围。** 先沿用同节点 PP2↔PP3、HiCache、已有 GPU→GPU KV patch 和候选权重 offload。CPU 存储按全局层号管理，尽量不随 PP 度数重新分配；新 worker 接入同一份底层共享内存。切换时更新全局层到 CPU 存储句柄／偏移、前缀块到槽位，以及 CPU↔GPU 的目标层映射，连同 GDN 配套状态一起交接。只在访问历史前缀时恢复所需内容到 GPU，这属于通常的 HiCache loadback，不意味着每次都向旧实例发远程请求。

**数据不动，控制权要交接。** 需要共享存储句柄、按层视图、槽位分配／回收权限及读写版本。被读取的块必须有引用或租约，防止旧 worker 逐出、覆盖或复用；切换后新写入遵循新层归属，不能双写。共享存储的持有者须独立于退休 GPU worker；各进程虚拟地址可以不同，不能把传递旧进程裸指针当作共享内存。原池若是整块分配，按层保留视图可能仍保留整块内存，需要另验 CPU 容量和回收。

**同节点先简化。** 当前往返实验的多个 worker 在同一物理节点，旧池和新池都位于该主机 RAM，但当前属于不同进程的私有分配，不会自动互相可见。先评估共享 CPU 存储与受控视图、页锁定／GPU 访问注册及现有 H2D／D2H kernel 的 stride 兼容性；不要求先引入远端缓存服务。跨节点再使用启动时准备好的 NIXL 通道按层迁移 DRAM 数据，实际是否 RDMA 按链路验证。

**后台做什么。** 共享内存方案没有必须完成的全量 CPU 搬运，可以只整理元数据和容量；若目标仍要求新物理池，后台按新层归属同步全部需保留的历史缓存，过渡期间保留旧存储访问映射，完成后再回收。此时按需读取只是同步期间的兼容路径。总 CPU 预算包含新旧池和临时副本，后台搬运限速，避免争抢推理带宽。

**仍须同步什么。** 当前活跃请求的最后 GPU KV patch、最新 GDN、请求进度和小规模映射／控制权提交不能省略。历史 CPU 字节不变时，目标是避免因 PP 换层而重拷或逐字节扫描，不是跳过一致性。CPU→GPU 的正常 loadback 仍有成本，共享内存后的 TTFT 和暂停待实测，不能承诺约87 ms直接减去39.8 ms。

**怎样验证。** 对照现在的 CPU 新池复制与同节点原位复用，先做短往返并在切换后实际恢复已缓存前缀。核对底层共享存储对象／偏移保持、CPU→CPU 历史复制字节、GPU 目标层正确、输出一致性、暂停、准备期 token 间隔、TTFT 和 CPU 峰值；补旧 worker 退出、槽位复用和在途读写检查。之后增加历史容量，检查切换是否仍扫描整个历史集合。实现及验证状态见[共享池接入说明](SHARED_HOST_CACHE.md)。
