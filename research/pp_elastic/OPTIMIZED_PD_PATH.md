# 122B P/D：接入最新原地伸缩优化

2026-11-14。本轮只验证固定P2、D侧PP2→PP3→PP2，保留HiCache、单个活跃Decode请求及转换期间暂停新P→D交接的边界，不运行完整端到端负载。

**接入内容。** 共享CPU HiCache沿全局层保留历史数据，后台准备目标层和NIXL增量KV，预热有限布局的Graph并在交接时选择。候选GPU权重继续offload。P/D在最终交接前撤销旧地址注册，提交后注册新地址；提交前失败恢复源注册，准备期间不提前发布目标地址。

**FP8权重的区别。** 旧BF16方案在CPU上调用原生loader预打包；Marlin的量化后处理包含GPU重排。本路径在服务启动前，利用私有布局预热取得处理后的权重并存入有上限的pinned CPU池。正式转换只做CPU→GPU恢复，不再重排、不导出源GPU权重，也不通过Gloo搬权重。启动期导出、CPU内存和Graph准备成本单独记录，不能称为纯CPU预打包。

**验收要求。** 同暖前缀的完整输出逐token一致，同一RID跨两次转换且无重试／重新Prefill；转换后新的P→D请求及真实CPU恢复正确。核对共享底层身份不变、历史CPU载荷不复制、运行时Graph不捕获且有实际replay、非活跃候选权重物理页为0。缩回后独立进程实际分配／写入64 GiB并计算，报告仍驻留的worker、Graph／KV及剩余显存。分别报告控制面完整时间、最终暂停及准备期推理间隔，每方向一次，不作为重复均值。

**当前状态。** r18（`dccbefa3c9c7`）GPU最小往返通过：512-token同暖参考、真实CPU恢复及独立64 GiB复用正确；1275项组合CPU、随后74项相关检查及73项保存结果复算通过。扩／缩控制1.940／1.262秒，最终暂停129／103毫秒；准备期仍有1.683秒出token间隔。退役rank权重／KV为0，worker及约1.78 GiB reserved仍驻留。启动预热约28.2秒单列；不是完整E2E、P侧、多请求、容量扩展或整卡归还证明。见[本轮报告](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_reactive_burst_20261113/OPTIMIZED_PP_VALIDATION.md)。默认入口为`pd_elastic_smoke.py --prepare`；同步参考必须显式使用`--sync-baseline`。旧r16保持不变。

首轮r17在启动期将已加载权重转换为VMM backing时OOM，未开始请求；失败与清理记录保留。旧PyTorch空闲块只在全部层转换后归还，改为逐层归还，避免累计保留第二份权重；同配置r18通过。旧后端释放夹具未初始化P/D钩子的CPU失败也保留。
