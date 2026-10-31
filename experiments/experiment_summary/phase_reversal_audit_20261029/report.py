"""Render absolute saved observations, without averaging or normalization."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
W=Path('/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling')
rows=list(csv.DictReader((ROOT/'formal_rounds.csv').open()))
groups=defaultdict(list)
for r in rows:
    groups[(r['series'],r['phase'],r['workload'],r['layout'],r['policy'])].append(r)

chosen={
 '35b_prefill_8192':('3','0'),
 '35b_decode_latest':('native','pp'),
 '35b_decode_lookahead_r2':('0','9'),
 '122b_decode_latest':('native','pp'),
 '122b_decode_refinement':('ep_control_r1:p0','pp_feedback_r3:p10'),
 '122b_historical_1025':('historical','historical'),
 '35b_a100_eager':('0','0'),
}
comparisons=[]
for series,(ep,pp) in chosen.items():
    cases=sorted({(k[1],k[2]) for k in groups if k[0]==series})
    for phase,work in cases:
        a=groups[(series,phase,work,'ep2',ep)]
        b=groups[(series,phase,work,'pp2',pp)]
        for metric in ['makespan_ms']+(['longest_decode_tail_ms'] if phase=='decode' else []):
            av=[float(x[metric]) for x in a];bv=[float(x[metric]) for x in b]
            lo,hi=min(av)-max(bv),max(av)-min(bv)
            comparisons.append(dict(series=series,phase=phase,workload=work,metric=metric,
                ep_policy=ep,pp_policy=pp,ep_first=av[0],pp_first=bv[0],
                ep_min=min(av),ep_max=max(av),pp_min=min(bv),pp_max=max(bv),
                ep_minus_pp_min=lo,ep_minus_pp_max=hi,
                observed_order='PP' if lo>0 else 'EP' if hi<0 else 'overlapping_ranges',
                ep_rounds=len(av),pp_rounds=len(bv)))
with (ROOT/'comparisons.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)

colors={'ep2':'#2463ad','pp2':'#d66b22'}
workloads=['b8_repeated','b16_repeated','b32_repeated','b32_diverse']
labels=['B8 repeated','B16 repeated','B32 repeated','B32 diverse']
fig,axes=plt.subplots(1,2,figsize=(11.8,4.7),sharey=True)
for ax,metric,title in zip(axes,['longest_decode_tail_ms','makespan_ms'],
    ['Generation after first token: latency order reverses',
     'Whole 128-token batch: PP still finishes sooner']):
    for i,work in enumerate(workloads):
        for arm,policy,offset in [('ep2','native',-.13),('pp2','pp',.13)]:
            rs=groups[('35b_decode_latest','decode',work,arm,policy)]
            vals=[float(r[metric]) for r in rs]
            y=i+offset
            ax.plot([min(vals),max(vals)],[y,y],color=colors[arm],lw=2)
            ax.scatter(vals,[y+(j-2)*.018 for j in range(len(vals))],s=20,
                color=colors[arm],label=arm.upper() if i==0 else None,zorder=3)
            ax.text(max(vals)+14,y,f'{min(vals):.0f}–{max(vals):.0f}',
                ha='left',va='center',fontsize=8,color=colors[arm])
    ax.set_title(title,fontsize=10,pad=12)
    ax.set_xlabel('Milliseconds (lower is better)')
    ax.set_xlim(800,1980);ax.set_yticks(range(4),labels)
    ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
axes[0].invert_yaxis()
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.93),frameon=False,ncol=2)
fig.suptitle('35B BF16 / 2 H100 / GPU-cached 4096 + new 1 → output 128',fontsize=13)
fig.text(.5,.025,'Each dot is one formal cohort (5 per point). Lines and numbers show observed min–max, not confidence intervals.\nLeft: longest per-request time from output 1 to output 128 (127 intervals); excludes TTFT. Right: full cohort wall time.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.12,1,.93])
for ext in ['png','pdf','svg']:fig.savefig(ROOT/f'35b_decode_absolute.{ext}',dpi=170,bbox_inches='tight')
plt.close(fig)

def table(series,phase,metric):
    out=['| Workload | EP2 第1轮 | PP2 第1轮 | EP2 全部轮次范围 | PP2 全部轮次范围 | 范围关系 |',
         '|---|---:|---:|---:|---:|---|']
    for r in comparisons:
        if (r['series'],r['phase'],r['metric'])!=(series,phase,metric):continue
        out.append(f"| {r['workload']} | {r['ep_first']:.2f} | {r['pp_first']:.2f} | {r['ep_min']:.2f}–{r['ep_max']:.2f} | {r['pp_min']:.2f}–{r['pp_max']:.2f} | {r['observed_order']} |")
    return '\n'.join(out)

text=f'''# 现有数据能否支持 P/D 阶段内 PP/EP 性能反转？

日期：2026-10-29。离线审查；没有启动模型、申请 GPU 或增加性能样本。

结论：**部分支持。最可信的是35B H100的Decode生成延迟随batch反转；尚不足以声称Prefill和Decode都已有稳定、较大的双向吞吐反转。** 跨阶段的配置偏好差异已有证据，但不能替代同阶段内部的反转，也不能直接证明异构实例池优于最佳同构池。

## 口径与审查范围

- 复核 {len(rows)} 条正式cohort记录，以及156条来自四卡实验的两卡组记录。后者是两个同时运行的组，不能当作156次独立实验。
- 展示固定第1正式轮和全部正式样本的最小–最大值，不计算跨轮均值，不归一化。范围不是置信区间，也不是P99。
- Prefill用整批完成时间。Decode同时保留整批128-token完成时间，以及每轮最慢请求从第1个到第128个输出的时间，即127次生成的绝对总耗时；后者排除了TTFT，没有除以token数。
- 当前两卡H100对照：EP2为Attention TP1/DP2 + Expert EP2；PP2为Attention TP1、Expert EP1、两个stage/固定槽。35B为BF16，122B为原生FP8。Prefill使用normal通信、Decode用经过资格检查的Graph/LL路径。
- EP采用该阶段表现更好的已测配置。没有用明显更慢的Decode两分块overlap来人为制造PP优势。Prefill主表比较EP p3 overlap/reserved-SM与PP native p0（PP early-send另留原始记录）；Decode比较native EP与PP policy9。
- 只核查已保存数据，不宣称覆盖未测batch、context或现代后端的全部最优实现。相同输入和输出token数量不代表EP/PP长生成序列完全相同，既有数值验证范围见各源报告；这里不将差额全部归因于专家工作集。

## 1. 35B Decode：延迟反转稳定，但完整cohort吞吐没有同样反转

每请求真实GPU缓存4096，新增1，输出128；同一H100178 GPU0/1 NVLink卡对。每点5轮。

**最慢请求的127次生成总耗时，单位ms，越低越好：**

{table('35b_decode_latest','decode','longest_decode_tail_ms')}

B8的PP优势在两侧全部样本任意组合下为77.84–82.69 ms；B32重复输入的EP优势为74.76–82.45 ms；B32多样输入的EP优势为67.38–70.86 ms。它们超过同一点观察到的轮次波动。B16接近边界，不能将其精确位置当作已知连续阈值。

独立的lookahead_r2接受运行重复了B8偏PP、B32偏EP的方向；B16范围重叠。复核表如下，不与上面的5轮拼成一个统计总体。

{table('35b_decode_lookahead_r2','decode','longest_decode_tail_ms')}

**包括首token的整批完成时间，单位ms：**

{table('35b_decode_latest','decode','makespan_ms')}

这里所有已测点仍是PP整批稍快：B8领先173.45–188.81 ms，B32重复仅4.62–15.82 ms，B32多样仅13.95–20.16 ms。PP较短的首token时间抵消了B32后续生成较慢的一部分成本。当前实验是有限128-token批次，没有测连续到达的饱和吞吐或在线SLO goodput；因此不能把生成延迟反转写成已观察到的大吞吐反转。

![35B Decode绝对时间：全部正式样本](35b_decode_absolute.png)

**幅度判断：** 足够稳定，值得作为延迟敏感选型的动机；但每127次生成节省约80 ms，对实时实例分流的净收益仍要计入合批/排队。它并不自动覆盖一次动态权重迁移的暂停，更不是已经测得的混合池收益。

## 2. 35B Prefill：优势明显，但目前是单向优势

冷输入P=0、每请求Q4096、输出1；两布局每Attention实例预算8192，B8/B32重复与B32多主题。下表为整批完成时间ms。

{table('35b_prefill_8192','prefill','makespan_ms')}

B8的PP优势为90.14–96.33 ms；B32重复为187.32–197.83 ms。多主题PP保留了一个较慢正式样本，仍未跨过EP范围。提前发送/其他EP策略记录在formal_rounds.csv中；没有只挑最快轮次。

因此：已有数据支持Prefill选型能有明显收益，但不支持“Prefill内部PP和EP各有稳定优势区间”这一更强命题。早一版Q128/Q512数据有显著跨布局数值差异，不用于加强等质量部署结论；没有用LL分块的组件大N曲线替代这里的normal后端实测。

## 3. 122B：最新Decode明显偏EP；旧Prefill反转证据偏弱

最新4组Decode、同样GPUcached4096+1→128的最慢请求生成尾段ms：

{table('122b_decode_latest','decode','longest_decode_tail_ms')}

四组尾段全部偏EP。B8完整cohort范围重叠，其余三组EP完整cohort也更快；没有找到最新版本的PP优势工况。

更新的B32 PP调优，24+24层policy10相对新鲜EP控制仍然落后：

{table('122b_decode_refinement','decode','makespan_ms')}

EP完整cohort优势191.69–242.47 ms；25+23层的另一候选没有改变方向。较大的优势是真实的EP单向优势，不是同阶段两侧都明显获益的反转。

历史2026-10-25 Prefill提供了一点反转线索：moderate为B8/P0/Q4096；near_capacity_b8为B8/P26752/Q512；near_capacity_b16为B16/P13376/Q512，均输出1且真实缓存准备不计时。

{table('122b_historical_1025','prefill','makespan_ms')}

前两组稳定偏PP；第三组EP为350.33–365.24 ms，PP为361.40–424.33 ms，范围重叠，PP的一轮较慢使汇总排序偏EP。只有三轮，且是旧PP、EP/PP分别在同节点不同GPU对运行，并非最新同卡对顺序复测。因此它是补测候选，不能作为足够强的反向优势证明。

旧moderate同一历史cohort中，Prefill偏PP、Decode偏EP，属于跨阶段反转。新优化把B8完整Decode从旧版明显EP优势收窄成接近持平，说明不能拼接旧P与新D幅度作为最新系统结论。

## 4. HiCache、其他硬件与副本候选

35B四卡自然重放拆分成两个同时运行的两卡组，每组B32。短前缀P8256：Prefill/Q512的全部组轮次EP2为319.78–335.92 ms、PP2为303.17–314.33 ms；Decode/+1→128整批EP2为1604.08–1684.69 ms、PP2为1896.93–2007.24 ms。这个协议下跨阶段反转明确，但PP调度、HiCache路径与前述最新短Decode组不同，不能把差额直接认作同一版本的Expert/通信作用。

长前缀P130304自然重放的Decode整批EP2为4804.29–5577.52 ms、PP2为5068.54–5397.23 ms，范围重叠。CPU重放也重叠，不能用某个首轮的轻微排序翻转作稳健证据。大Prefill差距同时包含缓存位置、容量和恢复，不是受控的纯专家/通信反转。

此处两卡预算的2×EP1也必须保留：短前缀自然重放Prefill为258.52–266.53 ms，Decode为1471.87–1559.86 ms，均快于同协议EP2/PP2。长期缓存容量不足时副本又明显落后。它支持更广的容量/副本选型问题，也提醒我们PP/EP混合池必须胜过包含副本的强基线。

A10035B的最新eager回归在已测3组Prefill和3组Decode中均偏PP，原始轮次已重算并单列，不能当作H100的同架构反转。已有H100 GLM跨节点历史对照的Prefill和Decode均偏PP2×EP4；这也不提供同阶段双向反转。HY3/GLM的二维Expert/通信派生图继续只作为机制线索，本轮没有把组件预算当整模型实测。GLM/HY3保留最小EP4约束。

另有HY3的历史整模型容量反转，必须与上述H100计算边界分开：8张A100、BF16完整80层、Graph关闭，EP8对PP2×EP4；8个请求、合成CPU HiCache、每点仅1个正式cohort，计时包含H2D和排队。Decode均新增1、输出128。缓存P2304时整批EP8为21.908秒、PP2×EP4为22.742秒，EP领先0.834秒；P3712时EP8为43.599秒、PP2×EP4为22.352秒，PP领先21.247秒。原始EP长缓存记录中有请求首token等到22.097秒，符合容量不足导致排队的解释。EP有两个Attention副本，每个容量12224 token，而4条请求的旧缓存就需要14848 token；PP的逻辑容量36992 token能够容纳8条请求的29696个旧缓存token及生成尾部。

这是一组幅度很大的Decode整批排序反转，但优势不对称，且大差额主要对应容量/排队边界。只有单轮、缓存不是模型生成的真实语义状态，不能将其当成已复现的纯Expert/通信交叉点。同协议Prefill/Q256在P2304和P3712两点均偏PP。逐项原始计时与协议另存hy3_capacity_observations.csv和hy3_sources.json，不并入上面的391条正式轮次统计。

## 对当前研究假设的判定

| 命题 | 目前证据 |
|---|---|
| P和D可以偏好不同布局 | 支持；存在历史同协议跨阶段反转，当前35B冷P与GPU-hot D也显示不同偏好，但不是完整PD交接实验 |
| Decode内部存在随batch变化的布局反转 | 支持35B生成延迟；两个接受运行重复方向 |
| Prefill内部存在稳定且较大的双向反转 | 尚不支持；35B单向偏PP，122B旧反向点范围重叠 |
| 两阶段都有大的吞吐反转，适合异构池 | 尚不支持；有限批次吞吐和生成延迟排序不同 |
| 混合池提高固定预算的SLO goodput | 未测；还需计入分流后的batch、排队、副本基线和P→D状态交接 |

建议保留“分阶段测量与选型”，暂时把机制动机收窄到35B Decode延迟边界。最小证据缺口是：最新同一实现下，找到反向且足够大的完整服务收益，并在连续到达负载、相同GPU预算、包含副本的基线上验证。现有数据不支持预先宣称P/D两个池都需要异构PP/EP实例。

## 来源与复现

- [35B最新Decode]({W}/experiments/h100_ep_decode_overlap_20261027/report/REPORT.md)
- [35B Prefill调优与normal后端]({W}/experiments/h100_ep_prefill_overlap_20261027/report/REPORT.md)
- [122B最新Decode]({W}/experiments/h100_qwen122b_latest_decode_20261027/report/REPORT.md)
- [122B PP再次调优]({W}/experiments/h100_qwen122b_pp_refine_20261027/report/REPORT.md)
- [HiCache三配置组口径]({W}/experiments/h100_five_layouts_hicache_20261028/two_gpu_three_layouts_analysis/REPORT.md)
- [H100 GLM/122B历史通信域对照]({W}/experiments/h100_ep_pp_comm_domain_comparison_20261026/REPORT.md)
- [HY3 A100合成缓存容量对照]({W}/experiments/hy3_stream_cached_measure_v4_20261020/comparison.json)、[对应协议]({W}/experiments/hy3_stream_cached_measure_v4_20261020/PROTOCOL.md)。
- [391条正式轮次](formal_rounds.csv)、[观察范围](ranges.csv)、[同协议差额](comparisons.csv)、[156条缓存组轮次](hicache_group_rounds.csv)、[原始文件哈希](sources.json)。

运行analyze.py仅读取原始JSON/CSV、验证声明哈希及SSE计数；report.py从保存的逐轮表生成此报告与PNG/PDF/SVG。未修改历史实验数据。
'''
(ROOT/'REPORT.md').write_text(text)
assert len(rows)==391
assert all(r['observed_order']=='PP' for r in comparisons if r['series']=='35b_prefill_8192')
latest_tail={r['workload']:r['observed_order'] for r in comparisons if r['series']=='35b_decode_latest' and r['metric']=='longest_decode_tail_ms'}
assert latest_tail['b8_repeated']=='PP' and latest_tail['b32_repeated']=='EP'
assert all(r['observed_order']=='PP' for r in comparisons if r['series']=='35b_decode_latest' and r['metric']=='makespan_ms')
(ROOT/'validation.json').write_text(json.dumps({'formal_cohorts':len(rows),'comparisons':len(comparisons),
 'new_gpu_measurements':0,'source_hashes_verified':len(json.loads((ROOT/'sources.json').read_text())['sources']),
 'all_formal_rounds_retained':True,'no_arithmetic_means':True,'source_count_and_direction_checks_passed':True},indent=2)+'\n')
print('Rendered',ROOT/'REPORT.md')
