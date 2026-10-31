# 固定配置对比：EP4与单卡／PP在不同P、D工况是否互有胜负

2026-10-30 · [统一索引](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/README.md)

**有静态互补。** 35B、同一节点四张H100预算下，冷Prefill偏单卡／PP；固定Decode小并发偏单卡／PP，大并发偏EP4。Decode的同阶段反转已经在另一节点用五轮、反向布局顺序复核。它支持继续评估动态分配，但不是已测出的动态收益。

本次把已有实测中的EP4完整列出，并从全部正式轮次重新核对均值，没有重复运行已覆盖的GPU实验。共引用143条配置×工况×运行记录；各节点、缓存协议、PP实现与指标分别保留。单卡与纯PP统一作为EP=1类别，PP不需要独自胜过单卡。

## 1. Prefill：EP4已经有直接对照

以下固定011节点，旧缓存P=0，每请求输出O=1，各点三轮请求算术平均。N为全组请求数，Q为每请求新输入。这里的PP2采用已通过当前协议资格的原生policy0，不冒用固定Decode的policy9。

| N×Q＝全组新输入token | EP4平均TTFT ms | 2×EP2 | 4×单卡 | 2×PP2 |
|---|---:|---:|---:|---:|
| 32×512＝16384 | 219.11 | 205.88 | 163.39 | 146.06 |
| 16×8192＝131072 | 720.38 | 687.48 | 501.32 | 424.94 |
| 8×16384＝131072 | 834.77 | 786.91 | 567.40 | 483.94 |

三档平均首token均为2×PP2最低，但整批完成均为4×单卡最低；二者都属于EP=1类别。以N16/Q8192为例，EP4、2×EP2、4×单卡、2×PP2的整批完成分别为1030.50、1034.23、706.53、721.86 ms。171独立主扫描方向一致。TTFT是请求体验指标，整批时间是完成全部工作的指标，不能混为一个排名。

## 2. Decode：同一缓存长度，改变并发就能反转

固定011节点。P为每请求已有GPU缓存，Q1/O128；没有CPU KV恢复。全部为三轮逐请求平均TPOT，单位ms/token；PP2为已核验固定批次输出一致性的policy9。

| P | N | EP4 | 2×EP2 | 4×单卡 | 2×优化PP2 |
|---:|---:|---:|---:|---:|---:|
| 4096 | 8 | 6.122 | 6.534 | 4.752 | 5.495 |
| 4096 | 16 | 6.740 | 7.435 | 5.996 | 6.700 |
| 4096 | 32 | 7.781 | 8.735 | 7.890 | 8.625 |
| 4096 | 64 | 9.220 | 10.683 | 10.310 | 11.181 |
| 256 | 128 | 10.630 | 12.919 | 未通过运行资格 | 15.402 |
| 131072 | 8 | 7.970 | 8.354 | 6.550 | 7.306 |
| 131072 | 32 | 14.936 | 15.598 | 容量不足 | 15.293 |

**最清楚的一对是P4096、N8与N64。** N8时单卡／PP均快于EP4；N64时EP4均快于单卡／PP。这不是跨缓存协议或不同模型拼出的反转。包含首token后也保留方向：N8请求E2E，EP4 964.66、单卡745.08、PP2 792.57 ms；N64则为EP4 1365.02、单卡1450.14、PP2 1534.63 ms。

独立171节点五轮反向复测：

| P4096，全组N | EP4平均TPOT ms | 4×单卡 | 2×优化PP2 |
|---:|---:|---:|---:|
| 8 | 6.331 | 4.878 | 5.658 |
| 64 | 9.438 | 10.479 | 11.389 |

N128单卡的GDN临时显存申请失败，不能用不存在的值夸大EP收益；P131072/N32单卡有独立容量约束。后一个工况PP2请求E2E2181.57 ms、EP4 2284.38 ms，PP虽有小幅整请求优势，TPOT仍略慢。缺测、容量和指标取舍均保留，详见原运行审计。

## 3. 固定布局下的P/D混合流量：把EP4也并列展示

以下为011主扫描，三轮平均请求E2E，单位ms。每轮64请求；P类为Q8192/O16、D类为Q256/O256，旧缓存P=0；同质布局按未完成请求数路由。每请求在一个实例完成P和D，没有真实P→D交接。

| P类／D类请求数 | 到达速率 请求/秒 | EP4 | 2×EP2 | 4×单卡 | 2×PP2 |
|---|---:|---:|---:|---:|---:|
| 48／16 | 16 | 2753.32 | 2538.76 | 875.40 | 1492.53 |
| 48／16 | 32 | 2484.50 | 2272.95 | 1719.50 | 1735.80 |
| 32／32 | 16 | 3272.80 | 2797.60 | 1567.96 | 2486.88 |
| 32／32 | 32 | 2579.28 | 2444.17 | 2048.80 | 2543.61 |
| 16／48 | 16 | 3854.36 | 3574.56 | 2244.42 | 3742.36 |
| 16／48 | 32 | 2987.43 | 3127.61 | 2639.24 | 3756.68 |

这六档混合流量中，EP4尚未赢得平均请求E2E；171方向一致。D类较多、32请求/秒时，EP4整轮完成4633.24 ms略低于单卡4736.90 ms，但平均E2E仍更高。因此应表述为“固定Decode有已复测的反转，当前混合到达网格尚无明显E2E反转”。

表中PP为主扫描微批8、policy0；后续微批16＋提前发送在三档改善TTFT，但没有改变四单卡最低平均E2E的结论。不能把主表当作全部PP实现的性能上限。[后续调优与SLO对照](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/pd_mix_pool_20261030/REPORT.md)。

## 4. 对动态分配的含义

静态互有胜负这一必要线索已经满足，无需先实现完整P/D分离，也无需要求PP胜过单卡。此前固定Decode零切换成本的选型代理从8.052降到7.451 ms，来自EP4与单卡之间的选择，属于本方向；它不是在线混合池收益的上界。

应先以固定P4096、低／高Decode并发两个端点为基准，记录实际活跃batch与队列。接下来如果增加新的固定流量工况，重点是让实际Decode并发进入EP4有利区间，而不只是提高名义请求率或增加P类占比。若之后测动态分配，必须把分流导致的batch变化、缓存位置和调整开销计入，并与同预算最佳固定布局比较。

这批Prefill与Decode分别使用冷输入和GPU热缓存，能定位部署工作区间；不能把二者差值全部归因于“阶段”这一单一因素。大模型、A100、跨节点与真实P/D迁移均不在本表结论范围。

## 数据与复核

- [全部143条均值与来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/ep4_fixed_phase_comparison_20261030/metrics.csv) · [逐轮重算脚本](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/ep4_fixed_phase_comparison_20261030/build_report.py) · [源CSV哈希和协议清单](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/ep4_fixed_phase_comparison_20261030/manifest.json)
- [固定Decode完整实验与失败审计](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/REPORT.md)
- [冷Prefill、混合到达完整实验](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/pd_mix_pool_20261030/REPORT.md)
- [单卡／PP类别与真实P/D分离的范围说明](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/pd_disaggregation_scope_20261030/REPORT.md)

历史测量保留原时间与来源。本次仅重排EP4对照并复核算术均值，没有新增GPU数据、动态配比或P/D分离结果。
