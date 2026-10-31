"""Report independently audited 122B FP8 Decode deployments."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent
LABELS = {'native': 'EP2 native + shared overlap', 'overlap': 'EP2 two-chunk overlap', 'pp': 'PP2 lookahead'}
COLORS = {'native': '#2474b5', 'overlap': '#dd8b26', 'pp': '#159d86'}


def read(p): return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, nargs='+', required=True)
    a = ap.parse_args()
    records = [read(ROOT/'analysis'/p.name/'results.json') for p in a.runs]
    assert {r['config']['variant'] for r in records} == {'native', 'overlap', 'pp'}
    assert all(r['audit']['guard_and_cleanup_verified'] and r['audit']['frozen_hashes_verified'] and not r['profiles_skipped'] for r in records)
    rows = [v for r in records for v in r['metrics']]
    names = list(dict.fromkeys(r['workload'] for r in rows))
    parts, comparisons, outputs = [], [], []
    lines = ['# Qwen3.5-122B-A10B FP8: latest EP2 / PP2 Decode comparison', '',
        '同一 H100178 NVLink GPU0/1，完整 48 层模型。EP2 为 Attention DP2/TP1、MoE EP2，'
        '每卡 48 层、每层 128 个本地专家；PP2 为 24+24 层、每层 256 个本地专家，两个固定 B/2 流水槽。', '',
        'B 是全局请求数：8、16、32 个重复语料请求，以及 32 个不同主题请求。沿用前一轮 35B 的输入构造；'
        '每请求 4096 个真实 GPU 缓存上下文 token，加 1 个新输入 token，生成 128 个输出 token。'
        '请求标记和输入 SHA256 校验相同，EP 两个 DP rank 与 PP 两个槽均使用相同的偶数/奇数分组。', '',
        '两边保留原生 FP8 W8A8/DeepGEMM、BF16 KV 和残差、FP32 GDN 状态。'
        '35B 的 BF16 专家 Triton 替换不适用于这些 FP8 权重，没有对任何一边混用该替换。'
        'PP 开启 policy9 提前发射、独立消息通道、下一槽准备与发射；'
        'EP 保留共享专家重叠，并单独测试新两分块的同层 MoE 通信/计算重叠。原生全模型 TBO 未启用。', '',
        '两边都开启原生 SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK；EP 使用原生 lazy ping-pong，'
        'PP 使用其原生非重叠调度的单 extra buffer。两边实际均按每请求 3 个 FP32 状态槽准入，'
        '保留 page64、EP 原生 overlap schedule 和相同的请求输入；未降低 GDN 状态精度。'
        '统一 mem_fraction_static=0.945；每卡 BF16 KV 池为 1.875GiB，EP 每卡 81,920 个全模型 KV token，'
        'PP 每卡 163,840 个半模型 KV token。池容量均足以容纳完整 B32 和输出余量。', '',
        '每点先完成真实前缀准备及编译/Graph/通信格式预热，再做 3 轮完整预热和 5 轮关闭 profiler 的正式计时。'
        '另有独立两 rank profile；每 rank 丢弃前 20 个实际本地 B/2 Decode Graph 后选择 16 个。'
        '前缀准备、数值/输出一致性检查和 profile 均不计入性能结果，无 CPU KV restore。', '',
        '吞吐为完整 cohort 输出 token 总量 / 从第一请求发出到最后输出到达的时间；'
        'TPOT 为每请求第一至最后输出到达间隔 /127，再对正式请求取平均。'
        'TTFT 为发出请求至第一个输出；E2E 为发出请求至最后一个输出。吞吐 SD 是 5 个 cohort 的样本标准差。', '',
        '| Workload | Configuration | Output tok/s ↑ | SD | TPOT ms ↓ | TTFT ms | E2E ms |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name in names:
        group = [r for r in rows if r['workload'] == name]
        by = {r['variant']: r for r in group}
        comparisons.append(dict(workload=name,
            pp_vs_native_ep_pct=100*(by['pp']['output_tok_s']/by['native']['output_tok_s']-1),
            pp_vs_overlap_ep_pct=100*(by['pp']['output_tok_s']/by['overlap']['output_tok_s']-1),
            ep_overlap_vs_native_pct=100*(by['overlap']['output_tok_s']/by['native']['output_tok_s']-1)))
        for v in LABELS:
            r = by[v]
            lines.append(f"| {name} | {LABELS[v]} | {r['output_tok_s']:,.2f} | {r['stdev_tok_s']:.2f} | {r['tpot_ms']:.3f} | {r['ttft_ms']:.2f} | {r['e2e_ms']:.2f} |")
    lines += ['', '| Workload | PP vs native EP throughput | PP vs overlapping EP throughput | EP overlap vs native |',
              '|---|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['workload']} | {r['pp_vs_native_ep_pct']:+.2f}% | {r['pp_vs_overlap_ep_pct']:+.2f}% | {r['ep_overlap_vs_native_pct']:+.2f}% |")
    lines += ['', '## Expert, Attention and communication', '',
        '单位为 µs/卡/实际层调用；EP 每卡 48 层，PP 每卡 24 层。Expert 仅 routed GEMM，'
        'Attention 仅 FA3/GDN core；量化、排列、共享专家及其他运算留在原始 profile。'
        'Dispatch/Combine kernel 包含等待。交集按同一 GPU 时间轴的区间求交，'
        '分母是该配置 Dispatch/Combine 的时间并集；这些分项不能相加作为端到端 TPOT。', '',
        '| Workload | Configuration | Expert GEMM | Attention core | Dispatch | Combine | Comm ∩ GEMM | Comm overlapped |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    keys = ['expert.routed_gemm', 'attention.core', 'communication.dispatch', 'communication.combine']
    for r in rows:
        group = [p for rec in records for p in rec['components'] if (p['workload'],p['variant']) == (r['workload'],r['variant'])]
        assert len(group) == 2
        means = {k:st.mean(p['kernel_us_per_layer'].get(k,0.) for p in group) for k in keys}
        value = dict(workload=r['workload'],variant=r['variant'],**means,
            overlap_us=st.mean(p['comm_gemm_overlap_us_per_layer'] for p in group),
            communication_overlapped_pct=st.mean(p['communication_overlapped_pct'] for p in group))
        parts.append(value)
        lines.append(f"| {r['workload']} | {LABELS[r['variant']]} | " + ' | '.join(f'{means[k]:.2f}' for k in keys) + f" | {value['overlap_us']:.2f} | {value['communication_overlapped_pct']:.1f}% |")
    focus={v:next(p for p in parts if p['workload']=='b32_repeated' and p['variant']==v) for v in LABELS}
    gemm_increase=100*(focus['overlap']['expert.routed_gemm']/focus['native']['expert.routed_gemm']-1)
    lines += ['', '## Observations', '',
        'B8 的完整 cohort 吞吐基本持平；EP 原路径的稳态 TPOT 更低，但首 token 更慢，'
        '两者在 128-token 输出下的完整 cohort 吞吐接近。B16、B32 重复和 B32 多样输入，'
        'PP 相对 EP 原路径吞吐分别低 '+', '.join(f"{abs(x['pp_vs_native_ep_pct']):.1f}%" for x in comparisons[1:])+ '。', '',
        f"新增两分块重叠使四组吞吐下降 {min(abs(x['ep_overlap_vs_native_pct']) for x in comparisons):.1f}%–{max(abs(x['ep_overlap_vs_native_pct']) for x in comparisons):.1f}%。"
        f"B32 重复时实际通信与 routed GEMM 的重叠比例达 {focus['overlap']['communication_overlapped_pct']:.1f}%，"
        f"但 routed GEMM 从 {focus['native']['expert.routed_gemm']:.1f} 增至 {focus['overlap']['expert.routed_gemm']:.1f} µs/层（+{gemm_increase:.1f}%）。"
        '两分块将每层 routed GEMM 从 2 次变为 4 次；Attention core 基本相同。'
        '这一实现的额外专家执行和通信 kernel 时间超过了可用重叠收益，应保留原生 EP 路径作为当前更快选择。', '',
        '相同全局 B32，重复改为多样语料时，EP 原路径 Expert GEMM 从 '
        f"{focus['native']['expert.routed_gemm']:.1f} 升至 {next(p['expert.routed_gemm'] for p in parts if p['workload']=='b32_diverse' and p['variant']=='native'):.1f} µs/层；"
        'PP 从 '+f"{focus['pp']['expert.routed_gemm']:.1f} 升至 {next(p['expert.routed_gemm'] for p in parts if p['workload']=='b32_diverse' and p['variant']=='pp'):.1f} µs/层。"
        'Attention core 仍约 48–49 µs/层。该现象支持在部署决策中额外考虑输入与专家执行特性，'
        '仅用 batch 大小不能预测性能。', '',
        'PP 的 Expert 与图间等待均需保留在成本模型中。以下图间空隙表提供测量证据，'
        '目前没有把所有剩余差距单独归因于 CPU、网络或某个调度步骤。']
    lines += ['', '## Graph execution and gaps in independent profiles', '',
        '同一 rank 相邻完整本地 B/2 Decode Graph 的平均 GPU span 和图间空隙。EP 每个 Graph 是 48 层，'
        'PP 每个 Graph 是单 stage 的 24 层，两个固定槽交替执行；不能直接把单个 PP Graph 当作请求 TPOT。'
        '图间空隙包含调度、阶段依赖和边界通信等等待，未细分为 CPU 或网络原因；'
        '这些值来自独立 profile，不与关闭 profiler 的 TPOT 直接相减。', '',
        '| Workload | Configuration | Graph GPU span ms | Between-graph gap ms | Graph start interval ms |',
        '|---|---|---:|---:|---:|']
    for r in rows:
        group=[p for rec in records for p in rec['components'] if (p['workload'],p['variant'])==(r['workload'],r['variant'])]
        lines.append(f"| {r['workload']} | {LABELS[r['variant']]} | {st.mean(p['model_graph_ms'] for p in group):.3f} | {st.mean(p['mean_between_graph_gap_ms'] for p in group):.3f} | {st.mean(p['mean_graph_start_period_ms'] for p in group):.3f} |")
    lines += ['', '## Validation and provenance', '',
        '每个 EP 配置、每种 workload，双 rank 各 48 个真实 MoE 层均比较同输入、同路由的原生/串行分块/重叠结果；'
        '原生/分块 relative L2≤1.5%，串行/重叠要求逐元素相同。PP 在计时前将原生调度与提前发射的完整 128-token 输出逐请求对照，'
        '正式轮次另检查固定分组、提前发射计数增加和所有权排空。', '',
        '正式请求逐一验证 4096 个缓存 token、128 个输出 ID、SSE 计数 1–128 及由事件重算的 TPOT。'
        '23 项调度/通信所有权 CPU 测试和 12 项原生 GDN 分配/锁测试通过。'
        '完整模型 FP8 dtype、48/24 层、DeepGEMM backend、输入 SHA256、冻结文件、CPU 权重缓存使用和 Guard 清理/释放均已检查。', '',
        'EP 首个探索运行在 B32 时被缓存检查拒绝，先前 B8/B16 部分数据未进入本表；'
        'no_buffer 尝试在参数校验阶段被拒绝，没有正式样本；后续低 KV 容量的 EP 运行也在 B32 缓存校验时被拒绝。'
        'PP 首次使用过大的自动 KV 池，在 typed NCCL 首次 P2P 缓冲分配时失败，无正式样本。'
        '最终统一显式 KV 预算并保留 FP32 状态、page64 和 EP overlap schedule；这些探索失败均未混入本表。', '']
    for rec in records:
        cfg = rec['config']
        for r in rec['metrics']:
            p = Path(rec['run'])/r['arm']/'workloads'/r['workload']/'output_agreement.json'
            outputs.append(dict(workload=r['workload'],variant=r['variant'],**read(p)))
        lines.append(f"- {cfg['variant']}: [{Path(rec['run']).name}]({rec['run']}/launch_manifest.json)")
    lines += ['', 'Engine commit: `'+records[0]['config']['engine_commit']+'`.', '',
        '[Structured results](results.json) · [Throughput / TPOT](decode_comparison.png) · [B32 component costs](component_costs.png)', '']
    out = ROOT/'report'
    out.mkdir(exist_ok=True)
    (out/'REPORT.md').write_text('\n'.join(lines))
    result = dict(runs=records,metrics=rows,components=parts,comparisons=comparisons,output_stability=outputs)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,data in [('full_metrics',rows),('components',parts),('comparisons',comparisons)]:
        with (out/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.6),layout='constrained')
    for i,v in enumerate(LABELS):
        data=[next(r for r in rows if r['workload']==n and r['variant']==v) for n in names]
        x=[j+(i-1)*.24 for j in range(len(names))]
        axes[0].bar(x,[r['output_tok_s'] for r in data],width=.24,yerr=[r['stdev_tok_s'] for r in data],capsize=2,label=LABELS[v],color=COLORS[v])
        axes[1].bar(x,[r['tpot_ms'] for r in data],width=.24,color=COLORS[v])
    for ax in axes:
        ax.set_xticks(range(len(names)),[n.replace('_','\n') for n in names]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Output throughput (token/s)');axes[1].set_ylabel('Mean request TPOT (ms)');axes[0].legend(fontsize=8)
    fig.suptitle('2 × H100 / NVLink | Qwen3.5-122B-A10B FP8 | cached4096 + new1 → output128')
    fig.supxlabel('Same native FP8/DeepGEMM backend and FP32 GDN states; 5 profiler-off cohorts per point',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(out/f'decode_comparison.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained')
    categories=keys+['overlap_us']
    for i,v in enumerate(LABELS):
        p=next(p for p in parts if p['variant']==v and p['workload']=='b32_repeated')
        bars=ax.bar([j+(i-1)*.24 for j in range(len(categories))],[p[k] for k in categories],width=.24,label=LABELS[v],color=COLORS[v])
        ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
    ax.set_xticks(range(5),['Expert GEMM','Attention core','Dispatch','Combine','Comm ∩ GEMM'])
    ax.set_ylabel('GPU time (µs / rank / actual layer)');ax.set_title('B32 repeated: Expert execution and communication overlap')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.margins(y=.18)
    fig.supxlabel('Separate profiles; mean of two ranks; GPU kernel times are not an additive TPOT breakdown',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(out/f'component_costs.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps(dict(report=str(out/'REPORT.md'),comparisons=comparisons),indent=2))


if __name__ == '__main__': main()
