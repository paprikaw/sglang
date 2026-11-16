# 122B P/D：接入最新原地伸缩优化

2026-11-14。本轮只验证固定P2、D侧PP2→PP3→PP2，保留HiCache、单个活跃Decode请求及转换期间暂停新P→D交接的边界，不运行完整端到端负载。

**接入内容。** 共享CPU HiCache沿全局层保留历史数据，后台准备目标层和NIXL增量KV，预热有限布局的Graph并在交接时选择。候选GPU权重继续offload。P/D在最终交接前撤销旧地址注册，提交后注册新地址；提交前失败恢复源注册，准备期间不提前发布目标地址。

**FP8权重的区别。** 旧BF16方案在CPU上调用原生loader预打包；Marlin的量化后处理包含GPU重排。本路径在服务启动前，利用私有布局预热取得处理后的权重并存入有上限的pinned CPU池。正式转换只做CPU→GPU恢复，不再重排、不导出源GPU权重，也不通过Gloo搬权重。启动期导出、CPU内存和Graph准备成本单独记录，不能称为纯CPU预打包。

**验收要求。** 同暖前缀的完整输出逐token一致，同一RID跨两次转换且无重试／重新Prefill；转换后新的P→D请求及真实CPU恢复正确。核对共享底层身份不变、历史CPU载荷不复制、运行时Graph不捕获且有实际replay、非活跃候选权重物理页为0。缩回后独立进程实际分配／写入64 GiB并计算，报告仍驻留的worker、Graph／KV及剩余显存。分别报告控制面完整时间、最终暂停及准备期推理间隔，每方向一次，不作为重复均值。

**当前状态。** r23（`5856c9ea8528`）最小往返通过：启动前预加载CUDA模块及数据，首次NIXL gather不再等待整段权重H2D。38.04 GiB／1.671秒恢复期间同一请求12→101，新增89个token；最长客户端间隔由r20的1684.2降到177.8 ms，准备期最大约34.0 ms。扩／缩控制1.917／1.279秒、最终fence124／103 ms；512-token一致、CPU恢复576 token和独立64 GiB复用通过，Graph仍开启、运行时capture=0。130项相关CPU检查复用于相同引擎代码；退役rank权重／KV为0，worker及约1.78 GiB reserved仍驻留。每条件一次，不是完整E2E、P侧、多请求、容量扩展或整卡退出证明。见[重叠验证报告](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_reactive_burst_20261113/WEIGHT_DECODE_OVERLAP.md)。默认`pd_elastic_smoke.py --prepare`选择预加载；`--lazy-cuda-modules`保留旧加载行为用于诊断，`--sync-baseline`保留同步参考。r16／r18原件不改。

首轮r17在启动期将已加载权重转换为VMM backing时OOM，未开始请求；失败与清理记录保留。旧PyTorch空闲块只在全部层转换后归还，改为逐层归还，避免累计保留第二份权重；同配置r18通过。旧后端释放夹具未初始化P/D钩子的CPU失败也保留。

**后续开销审计。** [专题报告](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_reactive_burst_20261113/MIGRATION_OVERHEAD_AUDIT.md)区分准备期1.683秒间隔与最终fence。opt-in `--diagnostic-prepare`只用于CPU栈／心跳采样，默认关闭；r19扩容采样显示空阶段rank1仍在GPU输出irecv路径，具体CUDA／NCCL同步未隔离。r19随后缩容准备失败，不能作新的成功往返或无停顿证明，原r18执行资格不扩大。
