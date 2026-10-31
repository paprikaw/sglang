# A100 旧 DeepEP：节点内与跨节点 EP 的纯通信对照

**结论：本轮不能把 PP2×EP4 的完整模型优势归为“节点内 dispatch/combine 必然更快”。** 固定四卡、相同逻辑 token 和专家路由时，双节点 cross4 的 Prefill 与 Decode 均比两个单节点 EP4 基线更快。实际 cross8 的 Prefill 也更快，但 Decode 比 EP4 慢。

| 布局 | GPUs/节点 | Decode4 Graph mean / median ms | Prefill1024 eager mean / median ms |
|---|---|---:|---:|
| local168 | 4/1 | 0.153291 / 0.150240 | 8.860064 / 8.894688 |
| local169 | 4/1 | 0.144128 / 0.141984 | 8.715243 / 8.705024 |
| cross4 | 4/2 | 0.127435 / 0.130048 | 5.405067 / 5.304256 |
| cross8 | 8/2 | 0.221888 / 0.225920 | 6.565184 / 6.552864 |

## 配置与计时边界

- 168、169 两个已预约节点，A100 80GB PCIe；NVLink GPU0↔1、GPU2↔3，跨pair SYS。两节点均使用逻辑 HCA `mlx5_bond_0`；GPU0/1→NIC SYS、GPU2/3→NIC NODE。不能由逻辑 HCA 数量推断物理网卡数量。
- 所有布局同旧 native `9dc34887`、host source `7d1ebddc`、pair-only=1、PCIe threshold=0，保留pair内NVLink，所有非pair payload走RDMA；NIC-handler cpu_host_memory、QP depth4096。
- BF16 hidden5120、160 experts、top-k8、相同逐token expert IDs。Prefill每方向80 MiB逻辑expert payload，Decode每方向320 KiB；不代表物理线上字节或测量带宽。
- local/cross4：40 experts与40 QPs/rank、capacity256、Prefill256 source tokens/rank；cross8：20 experts/QPs/rank、capacity128、Prefill128 source tokens/rank。Decode4在cross8有8个source slots，其中4个padding，padding专家ID=-1、不发专家payload。
- 三次warmup、三次正式采样。初始化、pre-call barrier、exact correctness和退出清理不计时；native packing/reduction/control、recv-hook等待、CUDA event span中的host launch gaps计时。Graph每sample一次完整dispatch/combine replay，不含GEMM或Attention。
- 每次迭代先取所有rank最大值，再算mean/median。Dispatch/Combine独立rank最大值不能相加；Decode Graph未插入阶段事件，不臆造分项。三个样本不支持P99、显著性或在线SLO结论。

## 对 PP 结论的约束

固定world4对照提供放置位置的直接证据，但同时把一节点的一套逻辑HCA/bond及CPU代理资源换成两节点的两套资源，不能单独隔离NIC、CQ或CPU代理根因。聚合资源增加是合理解释，未通过独立干预证明。
本轮cross8改变GPU数、expert ownership、QPs、source容量及Decode padding，属于实用scale-out比较。既有同旧native的1024-token纯通信EP4/EP8实验也出现EP8更快（pair median8.698880/6.491648 ms）；不能将“节点内”当通用通信加速定律。
完整GLM旧库重测Prefill吞吐PP2×EP4为1946.36 input tok/s、EP8为1324.78，PP领先约46.9%；这与单次EP4通信较慢不矛盾。PP每节点承担43/46个MoE层，EP8每节点承担89个；流水重叠、GEMM形状、批次/调度、stage交接频率同时改变。本profile没有计算或重叠，不能从组件延迟估算46.9%收益的归因比例。
真实节点内PCIe payload消融与本轮不同；历史CE原型固定4卡sweep不能当成当前旧RDMA路径的证据。

原始runtime：`/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/runtime/a100_ep_locality_causal_20261010_r1`。Guard scientific post、owned-process/GPU cleanup与全局lease释放通过，模型和CPU缓存未加载或删除。
