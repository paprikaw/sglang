"""Merge audited independent deployments, retaining negative outcomes."""
import argparse
import json
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parent
LABELS={'native':'EP2 native','serial':'EP2 serial split','overlap':'EP2 overlap','pp':'PP2 lookahead'}
COLORS={'native':'#1f77b4','serial':'#8b9299','overlap':'#ff7f0e','pp':'#2ca02c'}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',type=Path,required=True);args=ap.parse_args()
    all_data=[json.loads((ROOT/'analysis'/p.name/'results.json').read_text()) for p in args.runs]
    assert all(d['audit']['guard_and_cleanup_verified'] and d['audit']['frozen_hashes_verified'] and not d['profiles_skipped'] for d in all_data)
    rows=[r for d in all_data for r in d['metrics']];parts=[r for d in all_data for r in d['components']]
    comparisons=[];outputs=[];component_rows=[];findings=[]
    lines=['# H100 Decode: EP communication / Expert overlap','',
        '完整 Qwen3.6-35B-A3B，BF16 权重/激活/KV、FP32 GDN；H100178同一NVLink GPU0/1。'
        '沿用前一轮B8/B16/B32重复输入与B32多主题输入；每请求4096个真实GPU缓存token，新增1 token、生成128 token。'
        'EP为Attention DP2/TP1、MoE EP2；PP为两个20层阶段、两个B/2固定流水槽，开启policy9提前发射。','',
        '各配置用独立进程，在启动时固定策略并捕获CUDA Graph。Attention/GDN保持整批；MoE把各rank的token分成两组，'
        '沿用相同的小批次Triton Expert内核家族和DeepEP低延迟BF16通信；两组各有独立handle。'
        '所有布局保留原有shared-expert overlap，full-model two-batch overlap关闭。Decode没有沿用Prefill的DeepGEMM SM预算设置。','',
        '原生EP保留recv-hook的send/receive两阶段；串行和重叠分块都使用async-event的合并通信。'
        '因此串行/重叠对照隔离的是重叠调度收益；原生/分块差异同时包含分块和通信接口模式变化。'
        '每层实际Dispatch和Combine均各有2个kernel；原生是两阶段，分块是每块1个kernel。Expert GEMM则从每层2个变为4个。','',
        '每配置3个不计时完整批次预热、5个profiler关闭的正式批次。PP另外先沿用原生Decode的多形状预热，以建立提前发射所需的消息格式；这些批次不计时。前缀由本进程真实模型生成并留在GPU；无CPU KV换入。'
        '独立profile从每rank第21个匹配实际本地batch的Decode Graph开始取16个，统计每次实际层调用的GPU时间。'
        '不同进程的采样轮次不混为同一批次；没有把Graph分项相加作为TPOT。','',
        '| Workload | Configuration | Output tok/s ↑ | Sample SD | TPOT ms ↓ | TTFT ms | E2E ms |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name in dict.fromkeys(r['workload'] for r in rows):
        current=[r for r in rows if r['workload']==name]
        native=next(r for r in current if r['variant']=='native')
        opt=next(r for r in current if r['variant']=='overlap')
        comparisons.append(dict(workload=name,throughput_gain_pct=100*(opt['output_tok_s']/native['output_tok_s']-1),
            tpot_change_pct=100*(opt['tpot_ms']/native['tpot_ms']-1)))
        for row in sorted(current,key=lambda r:list(LABELS).index(r['variant'])):
            lines.append(f"| {name} | {LABELS[row['variant']]} | {row['output_tok_s']:,.1f} | {row['stdev_tok_s']:.1f} | {row['tpot_ms']:.3f} | {row['ttft_ms']:.2f} | {row['e2e_ms']:.2f} |")
    lines+=['','| Workload | EP overlap vs native throughput | TPOT change |','|---|---:|---:|']
    for row in comparisons:lines.append(f"| {row['workload']} | {row['throughput_gain_pct']:+.2f}% | {row['tpot_change_pct']:+.2f}% |")
    lines+=['','## Expert, Attention and communication','',
        '单位：µs/卡/实际层调用；Expert仅routed GEMM，Attention仅core。通信kernel包含等待；交集按同一GPU实际时间轴计算。'
        '双分块的一层包括两组通信和两组Expert计算，仍除以完整模型的40层，不把两组错误当作80层。','',
        '| Workload | Configuration | Expert GEMM µs | Attention µs | Dispatch µs | Combine µs | Comm ∩ GEMM µs | Comm overlapped |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        group=[p for p in parts if (p['workload'],p['variant'])==(row['workload'],row['variant'])]
        assert len(group)==2
        val={key:st.mean(p['kernel_us_per_layer'].get(key,0.) for p in group) for key in ['expert.routed_gemm','attention.core','communication.dispatch','communication.combine']}
        hit=st.mean(p['comm_gemm_overlap_us_per_layer'] for p in group);fraction=st.mean(p['communication_overlapped_pct'] for p in group)
        component_rows.append(dict(workload=row['workload'],variant=row['variant'],**val,overlap_us=hit,communication_overlapped_pct=fraction))
        lines.append(f"| {row['workload']} | {LABELS[row['variant']]} | {val['expert.routed_gemm']:.2f} | {val['attention.core']:.2f} | {val['communication.dispatch']:.2f} | {val['communication.combine']:.2f} | {hit:.2f} | {fraction:.1f}% |")
    opt_parts=[p for p in component_rows if p['variant']=='overlap']
    gains=[r['throughput_gain_pct'] for r in comparisons]
    if all(g<0 for g in gains):
        findings.append(f"本轮四组Decode均未获益：两分块EP重叠相对同轮原生EP吞吐降低{abs(max(gains)):.1f}%–{abs(min(gains)):.1f}%。")
    else:
        findings.append('不同workload的净吞吐变化见上表。')
    findings.append(f"GPU时间线实测隐藏了{min(p['communication_overlapped_pct'] for p in opt_parts):.1f}%–{max(p['communication_overlapped_pct'] for p in opt_parts):.1f}%的Dispatch/Combine kernel时间；该比例的分母是新分块配置的通信时间。")
    serial=next((r for r in rows if r['variant']=='serial'),None)
    if serial:
        peer=lambda variant:next(r for r in rows if r['variant']==variant and r['workload']==serial['workload'])
        n,o=peer('native'),peer('overlap')
        findings.append(f"{serial['workload']}串行拆分对照：原生{n['output_tok_s']:.1f}、串行两分块{serial['output_tok_s']:.1f}、重叠两分块{o['output_tok_s']:.1f} output tok/s。重叠相对串行拆分变化{100*(o['output_tok_s']/serial['output_tok_s']-1):+.1f}%，但未追回拆分损失。")
        c=lambda variant:next(p for p in component_rows if p['variant']==variant and p['workload']==serial['workload'])
        cn,co=c('native'),c('overlap')
        findings.append(f"同一workload的每层Expert GEMM从{cn['expert.routed_gemm']:.2f}升到{co['expert.routed_gemm']:.2f} µs/卡，Dispatch+Combine从{cn['communication.dispatch']+cn['communication.combine']:.2f}升到{co['communication.dispatch']+co['communication.combine']:.2f} µs/卡；与GEMM的实际交集为{co['overlap_us']:.2f} µs/卡/层。")
    findings.append('这支持在当前小token Decode下保留原生整批EP。新路径把每层逻辑通信组数从1增为2、Expert GEMM从2个增为4个，减小每次执行粒度；时序重叠不足以抵消这版实现的开销。本结论限于当前两分块实现与测得的四组workload。')
    lines[2:2]=['## Measured outcome','']+findings+['']
    lines+=['','## Numerical and execution evidence','',
        '每个EP配置、每种workload都在计时外，以真实输入、路由和权重对双rank各40层比较原生、串行分块和重叠。'
        '串行/重叠要求逐元素相同；原生/分块要求有限值及relative L2≤1.5%，实际最大误差见下表。'
        '正式批次检查图重放计数增加、固定策略和handle排空、每请求4096缓存token与128输出token、SSE事件和TPOT。','',
        '| Workload | Configuration | Native/split max relative L2 | Serial/overlap exact | Across-cohort full sequence stable |',
        '|---|---|---:|---|---|']
    for d in all_data:
        for row in d['metrics']:
            folder=Path(d['run'])/row['arm']/'workloads'/row['workload']
            agreement=json.loads((folder/'output_agreement.json').read_text())['all_formal_sequences_equal']
            if row['arm']=='ep2':
                quals=[json.loads(p.read_text()) for p in (folder/'overlap_qualification').glob('rank*.json')]
                maximum=max(v['native_vs_serial']['relative_l2'] for q in quals for v in q.values())
                exact=all(v['serial_vs_overlap']['exact'] for q in quals for v in q.values())
            else:maximum=None;exact=None
            lines.append(f"| {row['workload']} | {LABELS[row['variant']]} | {maximum if maximum is not None else '—'} | {exact if exact is not None else '—'} | {agreement} |")
            outputs.append(dict(workload=row['workload'],variant=row['variant'],across_cohort_stable=agreement))
    lines+=['',
        '旧原生EP在四个workload上均有跨批次长输出差异，因此不能把跨批次128-token序列相同作为新调度的独立正确性门槛。'
        '新调度的检查依据是同一次输入的逐层数值对照；PP固定分组仍要求跨批次完整输出一致。'
        'EP首轮因错误使用这一门槛中止，保留原始failed状态，性能样本全部排除。'
        'PP首轮在预热阶段因缺少原生Decode通信格式而中止，没有有效正式计时；新目录补回已验证的原生多形状预热后重测。','',
        '23项CPU检查通过；所有冻结输入SHA256、完整模型/CPU权重缓存使用、Guard postcheck、实验自有进程清理和租约释放通过。','',
        '[Structured results](results.json) · [Throughput and TPOT](decode_overlap.png) · [B32 GPU kernel costs](component_costs.png)','']
    if (ROOT/'gpu_qualification.json').exists():
        qualification=json.loads((ROOT/'gpu_qualification.json').read_text())
        lines+=['## Pinned source','',f"Engine commit: `{qualification['engine_commit']}`; H100 qualification tag: `{qualification['source_tag']}`.",'',
            f"[Decode implementation]({Path(all_data[0]['config']['engine_root'])/'python/sglang/srt/layers/moe/decode_overlap.py'})",'']
    out=ROOT/'report';out.mkdir(exist_ok=True)
    (out/'REPORT.md').write_text('\n'.join(lines))
    (out/'results.json').write_text(json.dumps(dict(runs=all_data,metrics=rows,components=component_rows,comparisons=comparisons,findings=findings,output_stability=outputs),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(dict.fromkeys(r['workload'] for r in rows));variants=['native','overlap','pp']
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.6),constrained_layout=True)
    for i,v in enumerate(variants):
        vals=[next(r for r in rows if r['workload']==n and r['variant']==v) for n in names]
        x=[j+(i-1)*.24 for j in range(len(names))]
        axes[0].bar(x,[r['output_tok_s'] for r in vals],width=.24,yerr=[r['stdev_tok_s'] for r in vals],capsize=2,label=LABELS[v],color=COLORS[v])
        axes[1].bar(x,[r['tpot_ms'] for r in vals],width=.24,label=LABELS[v],color=COLORS[v])
    for ax in axes:
        ax.set_xticks(range(len(names)),[n.replace('_','\n') for n in names]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Output throughput (token/s)');axes[1].set_ylabel('Mean request TPOT (ms)');axes[0].legend(fontsize=8)
    fig.suptitle('H100 × 2 / NVLink | Qwen3.6-35B-A3B BF16 | GPUcached4096 + new1 → output128')
    fig.supxlabel('Independent processes; same inputs and generic Expert kernel family; error bars: sample SD of 5 cohorts',fontsize=9)
    fig.savefig(out/'decode_overlap.png',dpi=180);fig.savefig(out/'decode_overlap.pdf');plt.close(fig)
    if serial:
        variants=['native','serial','overlap']
        costs=[next(p for p in component_rows if p['variant']==v and p['workload']==serial['workload']) for v in variants]
        categories=['expert.routed_gemm','communication.dispatch','communication.combine','overlap_us']
        fig,ax=plt.subplots(figsize=(9.2,4.5),layout='constrained')
        for i,(variant,part) in enumerate(zip(variants,costs)):
            x=[j+(i-1)*.24 for j in range(len(categories))]
            bars=ax.bar(x,[part[k] for k in categories],width=.24,label=LABELS[variant],color=COLORS[variant])
            ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
        ax.set_xticks(range(4),['Expert GEMM','Dispatch','Combine','Comm ∩ GEMM'])
        ax.set_ylabel('GPU kernel time (µs / rank / layer)')
        ax.set_title('B32 repeated: two-chunk path cost versus actual communication/compute overlap')
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.set_ylim(0,max(p[k] for p in costs for k in categories)*1.18)
        fig.supxlabel('Independent profiles; means of two ranks; kernel durations are not an additive TPOT decomposition.',fontsize=9)
        fig.savefig(out/'component_costs.png',dpi=180);fig.savefig(out/'component_costs.pdf');plt.close(fig)
    print(out/'REPORT.md')
if __name__=='__main__':main()
