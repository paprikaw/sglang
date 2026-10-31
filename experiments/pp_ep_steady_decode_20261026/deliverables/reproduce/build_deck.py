"""Create a source-backed Chinese presentation from accepted experiment CSVs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib import pyplot as plt
import numpy as np

from deck_engine import Deck, BLUE, TEAL, ORANGE, INK, MUTED, FONT_ROOT

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
A = ROOT / 'runs/a100_r2'
H = ROOT / 'runs/h100_r2'
OLD_A = EXPERIMENTS / 'ep2_pp2_qwen35b_a100_20261025/runs/a100_qwen35b_ep2_pp2_r2/analysis'
OLD_H = EXPERIMENTS / 'ep2_pp2_qwen35b_h100_20261025/runs/h100_qwen35b_ep2_pp2_r3/analysis'
BIG_H = EXPERIMENTS / 'ep2_pp2_qwen_h100_20261025/runs/h100_prefill_components_r1/analysis'
SUMMARY = EXPERIMENTS / 'experiment_summary'


def read_csv(path):
    return list(csv.DictReader(Path(path).open()))


def f(row, key):
    return float(row.get(key) or 0)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def setup_plotting():
    for weight in ['Regular', 'Bold']:
        font_manager.fontManager.addfont(str(FONT_ROOT / f'NotoSansCJKsc-{weight}.otf'))
    plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'font.size': 11,
        'axes.unicode_minus': False, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.edgecolor': '#CBD5E4',
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})


def finish_chart(fig, path):
    fig.savefig(path, dpi=240, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def grouped(path, labels, series, ylabel, errors=None, names=('EP2', 'PP2'), colors=(BLUE, TEAL), decimals=3, size=(8.2, 3.8)):
    fig, ax = plt.subplots(figsize=size)
    x = np.arange(len(labels))
    width = .72 / len(series)
    upper = 0
    for index, values in enumerate(series):
        error = errors[index] if errors else None
        pos = x - .36 + width / 2 + index * width
        ax.bar(pos, values, width * .9, color=colors[index], label=names[index],
            yerr=error, capsize=3, error_kw={'elinewidth': .8, 'ecolor': '#344360'})
        for k, (p, value) in enumerate(zip(pos, values)):
            extra = (error[k] if error is not None else 0)
            upper = max(upper, value + extra)
            ax.annotate(f'{value:.{decimals}f}', (p, value + extra), xytext=(0, 5),
                textcoords='offset points', ha='center', fontsize=10)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, upper * 1.24 if upper else 1)
    ax.grid(axis='y', color='#E7ECF4', linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, ncol=len(series), loc='upper right')
    fig.tight_layout()
    finish_chart(fig, path)


def history_chart(path, pairs):
    fig, ax = plt.subplots(figsize=(9, 4.45))
    labels, values = [], []
    for row in pairs:
        labels.append(f"{row['model']} · {row['gpu']} · {row['phase'].capitalize()}")
        values.append(row['pp_gain_pct'])
    y = np.arange(len(values))
    ax.barh(y, values, color=[TEAL if value > 0 else BLUE for value in values], height=.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0, color='#7E8CA4', lw=.8)
    ax.set_xlim(-35, 48)
    ax.set_xlabel('PP2 相对 EP2 的吞吐变化（%）；正值偏向 PP，负值偏向 EP')
    for k, value in enumerate(values):
        ax.text(value + (1 if value >= 0 else -1), k, f'{value:+.1f}%',
                va='center', ha='left' if value >= 0 else 'right', fontsize=10)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.grid(axis='x', color='#E7ECF4', lw=.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    finish_chart(fig, path)


def collect():
    sources = [A/'analysis/native_metrics.csv', A/'analysis/components.csv',
        H/'analysis/native_metrics.csv', H/'analysis/components.csv',
        OLD_A/'moderate_comparison.csv', OLD_H/'prefill_components.csv',
        OLD_H/'decode_components.csv', OLD_H/'full_metrics.csv', BIG_H/'comparison.csv',
        SUMMARY/'qwen_ep2_pp2_prefill_profiles.csv', SUMMARY/'qwen_prefill_diagnosis/REPORT.md',
        SUMMARY/'component_reports/Q02.md']
    for run in [A, H]:
        validation = json.loads((run/'analysis/validation.json').read_text())
        assert validation['passed'] and not validation['partial']
        postchecks = list((run/'dashboard_guard/checks').glob('*-post.json'))
        assert postchecks and json.loads(postchecks[-1].read_text())['passed']
    metrics, components = {}, {}
    for run in [A, H]:
        for row in read_csv(run/'analysis/native_metrics.csv'):
            metrics[(row['hardware'], row['arm'])] = row
        for row in read_csv(run/'analysis/components.csv'):
            components[(row['hardware'], row['arm'], row['execution'])] = row
    history = read_csv(OLD_A/'moderate_comparison.csv')
    pairs = []
    for model in ['35B BF16', '122B FP8']:
        for gpu in ['A100', 'H100']:
            for phase in ['prefill', 'decode']:
                subset = [r for r in history if (r['model'], r['gpu'], r['phase']) == (model, gpu, phase)]
                e = next(r for r in subset if r['arm'] == 'ep2')
                p = next(r for r in subset if r['arm'] == 'pp2')
                pairs.append(dict(model=model, gpu=gpu, phase=phase,
                    ep_tokens_per_s=f(e, 'tokens_per_s'), pp_tokens_per_s=f(p, 'tokens_per_s'),
                    pp_gain_pct=100 * (f(p, 'tokens_per_s') / f(e, 'tokens_per_s') - 1)))
    pref = []
    for model, path in [('35B BF16', OLD_H/'prefill_components.csv'), ('122B FP8', BIG_H/'comparison.csv')]:
        selected = [r for r in read_csv(path) if r['case'] == 'moderate']
        ep = next(r for r in selected if r['arm'] == 'ep2')
        pp = [r for r in selected if r['arm'] == 'pp2']
        def expert(row):
            return f(row, 'routed_expert_ms') + f(row, 'shared_expert_ms')
        pref.append(dict(model=model,
            attention_ep_ms=f(ep, 'attention_ms'),
            attention_pp_ms=statistics.mean(f(r, 'attention_ms') for r in pp),
            expert_ep_ms=expert(ep), expert_pp_ms=statistics.mean(expert(r) for r in pp),
            expert_increase_ms=statistics.mean(expert(r) for r in pp) - expert(ep),
            ep_dispatch_combine_ms=f(ep, 'dispatch_ms') + f(ep, 'combine_ms')))
    workload = []
    rows = read_csv(OLD_H/'full_metrics.csv')
    for case in ['moderate', 'near_capacity_b16', 'near_capacity_b32']:
        record = dict(case=case)
        for phase in ['prefill', 'decode']:
            selected = [r for r in rows if (r['case'], r['phase']) == (case, phase)]
            ep = next(r for r in selected if r['arm'] == 'ep2')
            pp = next(r for r in selected if r['arm'] == 'pp2')
            record[phase + '_pp_gain_pct'] = 100 * (f(pp, 'tokens_per_s') / f(ep, 'tokens_per_s') - 1)
            record['requests'] = int(ep['requests'])
            record[phase + '_cached_each'] = int(ep['cached_context_per_request'])
            record[phase + '_new_each'] = int(ep['fresh_input_per_request'])
        workload.append(record)
    long_source = next(r for r in read_csv(OLD_H/'decode_components.csv')
        if r['case'] == 'near_capacity_b16' and r['arm'] == 'ep2')
    long_context = [dict(metric=label, ms=value) for label,value in [
        ('Attention 核心',f(long_source,'attention_ms')),
        ('Routed GEMM',f(long_source,'routed_gemm_ms')),
        ('Dispatch + Combine',f(long_source,'dispatch_ms')+f(long_source,'combine_ms'))]]
    interference = [dict(metric='同层 Marlin 投影', low_overlap_ms=.7718, high_overlap_ms=1.2679),
        dict(metric='Full Attention 核心',low_overlap_ms=5.0535, high_overlap_ms=5.0619)]
    return dict(metrics=metrics, components=components, history=pairs, prefill=pref,
        workload=workload, long_context=long_context, attention_interference=interference, sources=sources)


def main(out):
    setup_plotting()
    data = collect()
    out.mkdir(parents=True, exist_ok=True)
    charts = out/'figures'
    charts.mkdir(exist_ok=True)
    metrics, components = data['metrics'], data['components']
    def c(hw, arm, execution='native'):
        return components[(hw, arm, execution)]
    def m(hw, arm):
        return metrics[(hw, arm)]
    def delta(hw, field, execution='native'):
        return 100*(f(c(hw, 'pp2', execution), field)/f(c(hw, 'ep2', execution), field)-1)
    history_chart(charts/'history_matrix.png', data['history'])
    grouped(charts/'native_tpot.png', ['A100 · 原生 eager', 'H100 · 原生 Graph'],
        [[f(m(hw, arm), 'tpot_ms') for hw in ['A100', 'H100']] for arm in ['ep2', 'pp2']],
        '平均 TPOT（毫秒，越低越好）', decimals=2)
    grouped(charts/'native_throughput.png', ['A100', 'H100'],
        [[f(m(hw, arm), 'output_tokens_per_s') for hw in ['A100', 'H100']] for arm in ['ep2', 'pp2']],
        '输出吞吐（token/s，越高越好）',
        errors=[[f(m(hw, arm), 'throughput_stdev') for hw in ['A100', 'H100']] for arm in ['ep2', 'pp2']], decimals=1)
    for hw, execution, fields, labels, name in [
        ('A100', 'native', ['attention.total', 'expert.routed', 'communication.dispatch', 'communication.combine'],
         ['Attention\n含投影', 'Routed Expert', 'Dispatch', 'Combine'], 'a100_modules'),
        ('H100', 'native', ['attention.core', 'expert.routed_gemm', 'communication.dispatch', 'communication.combine', 'other'],
         ['Attention\n核心', 'Routed\nGEMM', 'Dispatch', 'Combine', '其他\n含投影等'], 'h100_graph'),
        ('H100', 'eager_diagnostic', ['attention.total', 'expert.routed', 'communication.dispatch', 'communication.combine'],
         ['Attention\n含投影', 'Routed Expert', 'Dispatch', 'Combine'], 'h100_eager_modules')]:
        grouped(charts/(name+'.png'), labels,
            [[f(c(hw, arm, execution), field) for field in fields] for arm in ['ep2', 'pp2']],
            'GPU 活动（毫秒／每卡／每次实际层调用）',
            errors=[[f(c(hw, arm, execution), field+'.profile_stdev') for field in fields] for arm in ['ep2', 'pp2']], decimals=3)
    grouped(charts/'model_expert_tradeoff.png', [r['model'] for r in data['prefill']],
        [[r['expert_increase_ms'] for r in data['prefill']],
         [r['ep_dispatch_combine_ms'] for r in data['prefill']]],
        'GPU 活动（毫秒／每卡／整批 Prefill）',
        names=('PP 增加的 Routed + Shared Expert', 'EP 的 Dispatch + Combine'),
        colors=(ORANGE, BLUE), decimals=1)
    grouped(charts/'attention_interference.png', ['同层 Marlin 投影', 'Full Attention 核心'],
        [[1., 1.], [r['high_overlap_ms']/r['low_overlap_ms'] for r in data['attention_interference']]], '相对内核时间（各自无重叠 = 1）',
        names=('未与 NCCL 重叠', '与 NCCL 高度重叠'), colors=(BLUE, ORANGE), decimals=2)
    grouped(charts/'long_context.png', ['Attention 核心', 'Routed GEMM', 'Dispatch + Combine'],
        [[r['ms'] for r in data['long_context']]],
        'GPU 内核活动（毫秒／每卡／一次实际 Graph）',
        names=('H100 EP2 · 16 请求 × 179,968 缓存',), colors=(BLUE,), decimals=2)

    deck = Deck(out)
    def note(text, x=.82, y=6.37, width=11.65, height=.49, size=12, ink=MUTED):
        deck.text(text, x, y, width, height, size, ink)
    def card(x, y, width, height, title, body, accent=BLUE, title_size=20, body_size=16):
        deck.rect(x, y, width, height)
        deck.text(title, x+.23, y+.20, width-.46, .48, title_size, accent, True)
        deck.text(body, x+.23, y+.86, width-.46, height-1.04, body_size)
    sources = data['sources']
    ep_tpot = {hw:f(m(hw,'ep2'),'tpot_ms') for hw in ['A100','H100']}
    pp_tpot = {hw:f(m(hw,'pp2'),'tpot_ms') for hw in ['A100','H100']}
    a_reduce = 100*(1-pp_tpot['A100']/ep_tpot['A100'])
    h_reduce = 100*(1-ep_tpot['H100']/pp_tpot['H100'])

    deck.slide('什么时候用 EP，什么时候用 PP？',
        '固定 GPU 预算 · 模型 × 平台配置 × workload · 2026-10-26', sources=sources,
        notes='本报告综合历史端到端测量与本轮35B Decode补测。目标是形成有证据的部署判断，而非宣布通用最优布局。平台含原生后端、Graph和overlap差异。')
    deck.rect(.78, 1.8, 11.78, 2.05)
    deck.text('跨域扩大 EP 是否值得，\n取决于专家执行效率与通信代价的共同变化。', 1.10, 2.13, 11.08, 1.3, 30, bold=True)
    card(.78, 4.2, 3.76, 1.75, '模型与容量', '先判断权重、KV 与运行缓冲能否放下。', BLUE, 19, 15)
    card(4.77, 4.2, 3.76, 1.75, '通信域与平台', '比较逐层专家交换与流水阶段开销。', TEAL, 19, 15)
    card(8.76, 4.2, 3.80, 1.75, '请求与阶段', 'Prefill、Decode、上下文和微批次分别考察。', ORANGE, 19, 15)
    note('Attention 作为对照项；最终选择用原生端到端性能确认。')
    deck.finish_slide()

    deck.slide('固定两卡：总专家权重相近，每层工作集不同',
        '35B：40 层、每层 256 个路由专家、每 token 选 8 个；Attention 均为 TP1',
        sources=[A/'launch_manifest.json', H/'launch_manifest.json'],
        notes='均衡层、专家同尺寸的示意。EP每卡40层×128专家=5120专家块；PP每卡20层×256专家=5120专家块。Attention复制、KV和通信/Graph缓冲不同，不能据此声称总显存完全相等。EP的Attention为TP1×DP2，PP为分层TP1，两槽每槽最多4请求。')
    for x, title, accent in [(.78,'EP2：同一层分专家',BLUE),(6.82,'PP2：按层分阶段',TEAL)]:
        deck.rect(x,1.85,5.73,3.55)
        deck.text(title,x+.25,2.08,5.15,.5,22,accent,True)
    for x in [1.10,3.73]:
        deck.rect(x,2.93,2.15,1.32,'#EAF0FF')
        deck.text('GPU\n40 层 × 128 专家',x+.12,3.13,1.92,.89,16,BLUE,align='center')
    for x, text in [(7.12,'GPU 0\n前 20 层 × 256 专家'),(9.79,'GPU 1\n后 20 层 × 256 专家')]:
        deck.rect(x,2.93,2.15,1.32,'#E0F4F0')
        deck.text(text,x+.10,3.13,1.95,.89,15,TEAL,align='center')
    deck.text('逐层 Dispatch / Combine',1.1,4.62,5.0,.42,17,BLUE)
    deck.text('阶段边界传输 + 两槽流水',7.12,4.62,4.9,.42,17,TEAL)
    note('相同 8 请求：EP 每卡 4×40 = 160 请求层；PP 每卡 8×20 = 160 请求层。', y=5.75, height=.42, size=16, ink=INK)
    note('关键变化是每层专家工作集、单专家 token 复用和通信位置；Attention 并没有在 EP 中变成 TP2。',y=6.35)
    deck.finish_slide()

    deck.slide('历史端到端结果：同样两卡，没有固定赢家',
        '8 请求；Prefill：无旧缓存、每条新输入 4096；Decode：GPU 已缓存 4096、再生成 128',
        sources=[OLD_A/'moderate_comparison.csv'], notes='三轮吞吐均值。PP均为两槽。35B为BF16、122B为FP8检查点；A100与H100的Expert/Attention后端及Graph/overlap不同。不能将平台对照解读为纯硬件因果。35B H100 Prefill EP方波动较大，122B A100的Prefill小差距也只作线索。')
    deck.rect(.78,1.75,11.78,4.62)
    deck.image(charts/'history_matrix.png',.95,1.85,11.45,4.32)
    note('模型规模、平台和阶段都要纳入候选评估；小幅 Prefill 差距与 BF16／FP8差异不支持“越大必选 EP”。',y=6.48,height=.43,size=11)
    deck.finish_slide()

    deck.slide('本轮补测：稳态 Decode 的选择发生反转',
        '35B BF16 · 同一对 GPU0/1 顺序运行 · 各 3 轮无 profiler 测量；三段分项 profile 独立采集',
        sources=[A/'analysis/native_metrics.csv',H/'analysis/native_metrics.csv'],
        notes='各轮8请求，真实模型GPU前缀4096，新输入1，输出128。A100保留Triton/eager/无overlap；H100保留DeepGEMM/FA3/Decode Graph及原生EP overlap。H100 r1首轮TTFT异常及未生效诊断保留，r2补齐本地1/2/3/4形状预热后重新测全部三轮，不从同轮挑选样本。')
    deck.rect(.78,1.8,6.0,3.6)
    deck.image(charts/'native_tpot.png',.93,1.95,5.7,3.15)
    card(7.02,1.8,5.54,1.64,'A100：PP2 更优',f'TPOT {ep_tpot["A100"]:.2f} → {pp_tpot["A100"]:.2f} ms，降低 {a_reduce:.1f}%。',TEAL,20,16)
    card(7.02,3.7,5.54,1.64,'H100：EP2 更优',f'TPOT {pp_tpot["H100"]:.2f} → {ep_tpot["H100"]:.2f} ms，降低 {h_reduce:.1f}%。',BLUE,20,16)
    note('这是各平台原生软件栈下的部署结果；不是只更换 GPU 型号的单变量实验。',y=5.85,height=.47,size=15,ink=INK)
    note('端到端吞吐与三轮范围见附录；TPOT、TTFT 和整批吞吐分别保留。',y=6.43)
    deck.finish_slide()

    deck.slide('A100：Attention 相近，专家通信代价明显',
        '原生 eager；Attention 含投影；每卡每次实际层调用、本地 Decode batch=4',
        sources=[A/'analysis/components.csv', A/'analysis/validation.json'],
        notes='每配置3个独立profile、每profile2 ranks；每rank丢弃最初4个Decode forward后选择24个完整B4 forward，EP每rank960层调用、PP每rank480。误差条是3份profile卡均值的样本标准差，非同一trace内核重复的独立样本。PP边界收发单列，不记为零通信。')
    deck.rect(.78,1.75,11.78,3.92)
    deck.image(charts/'a100_modules.png',.96,1.85,11.4,3.67)
    note(f'EP → PP：Attention {delta("A100","attention.total"):+.1f}%；Routed Expert {delta("A100","expert.routed"):+.1f}%（结合误差条判断）。',y=5.88,height=.42,size=16,ink=INK)
    note(f'PP 阶段 NCCL 驻留约 {f(c("A100","pp2"),"pp_boundary_residency_ms_per_selected_forward"):.3f} ms／卡／forward，包含等待；不能与每层分项相加。',y=6.43)
    deck.finish_slide()

    deck.slide('H100：保留原生 Graph，检查专家与通信的变化',
        '实际 Graph replay；Attention 核心、Routed GEMM、DeepEP 与其他内核分开统计',
        sources=[H/'analysis/components.csv', H/'analysis/validation.json'],
        notes='只使用实际cudaGraphLaunch correlation匹配的Decode GPU kernel；每次核对Full/Linear Attention层数。PP以每层归一化，避免将20层stage与40层EP直接比总时间。其他项保留投影、共享专家、路由等，不能把它们视作不重要。PP边界在Graph外，另行统计驻留。')
    deck.rect(.78,1.75,11.78,3.92)
    deck.image(charts/'h100_graph.png',.96,1.85,11.4,3.67)
    note(f'EP → PP：Attention 核心 {delta("H100","attention.core"):+.1f}%；Routed GEMM {delta("H100","expert.routed_gemm"):+.1f}%。',y=5.88,height=.42,size=16,ink=INK)
    note(f'PP 阶段 NCCL 驻留约 {f(c("H100","pp2"),"pp_boundary_residency_ms_per_selected_forward"):.3f} ms／卡／forward，含等待、可与计算重叠；不与图中每层值相加。',y=6.43,size=11)
    deck.finish_slide()

    deck.slide('补充完整模块范围：H100 的 eager 诊断',
        '同一批已加载真实权重；单独切换原生 eager fallback；这张图不代表 Graph 模块时间',
        sources=[H/'analysis/components.csv',H/'analysis/components_per_rank.csv'],
        notes='三次独立eager诊断，每rank选24个稳态B4 forward；必须没有整模型cudaGraphLaunch，并有40/20层完整CPU组件注解。正式TPOT只在原生Graph路径测量。该诊断用于查看含投影的Attention与完整Routed Expert范围，不将eager绝对耗时套到Graph端到端。')
    deck.rect(.78,1.75,11.78,3.92)
    deck.image(charts/'h100_eager_modules.png',.96,1.85,11.4,3.67)
    note(f'此诊断中 EP → PP：Attention {delta("H100","attention.total","eager_diagnostic"):+.1f}%；Routed Expert {delta("H100","expert.routed","eager_diagnostic"):+.1f}%。',y=5.88,height=.42,size=16,ink=INK)
    note('绝对耗时与上一页 Graph 统计范围不同；只在本页内部比较 EP／PP。',y=6.43)
    deck.finish_slide()

    deck.slide('模型维度：更大的 Expert 成本会吃掉通信收益',
        'H100 历史冷 Prefill：8×4096 新 token；每卡整批 GPU 活动；两阶段 PP 取卡均值',
        sources=[OLD_H/'prefill_components.csv', BIG_H/'comparison.csv'],
        notes='Expert包含Routed+Shared，Attention包含投影。35B与122B分别为BF16与FP8、不同架构，不能隔离参数规模因果。图中的Expert增量和D/C活动来自重叠GPU组件，不能相减后称作净端到端收益。35B EP Prefill吞吐三轮波动大，14%与6%只作方向性线索。')
    deck.rect(.78,1.75,11.78,3.92)
    deck.image(charts/'model_expert_tradeoff.png',.96,1.85,11.4,3.67)
    note('两个模型的 Attention 在 EP／PP 间都接近；PP 增加的 Expert 活动约从 66 ms/卡升到 118 ms/卡。',y=5.88,height=.49,size=14,ink=INK)
    note('可支持“看 Expert 成本变化”；尚不能推出“模型越大越应该用 EP”。',y=6.43)
    deck.finish_slide()

    deck.slide('workload 维度：Attention 仍然必须保留',
        'H100 35B 历史数据：上下文长度、请求数与新增输入分别标注',
        sources=[OLD_H/'full_metrics.csv',OLD_H/'decode_components.csv'],
        notes='两条长缓存点总旧缓存均为2879488 tokens，但16请求与32请求的单条上下文、总新token数不同，不是固定上下文的纯batch sweep。右图为16请求、每条缓存179968的EP2，实际Graph本地batch8，包含40层；仅Attention核心、Routed GEMM、D/C累计内核活动，不能相加为TPOT。')
    deck.rect(.78,1.8,5.65,4.45)
    deck.text('PP 相对 EP 的吞吐变化',1.02,2.02,5.12,.45,20,bold=True)
    labels=['8 请求：Prefill 无旧缓存，\nDecode 每条缓存 4,096',
            '16 请求：每条缓存 179,968\nPrefill 每条新增 512',
            '32 请求：每条缓存 89,984\nPrefill 每条新增 512']
    for index,(row,text) in enumerate(zip(data['workload'],labels)):
        y=2.76+index*1.0
        deck.text(text,1.02,y,4.9,.55,13)
        deck.text(f'Prefill {row["prefill_pp_gain_pct"]:+.1f}%  /  Decode {row["decode_pp_gain_pct"]:+.1f}%',1.02,y+.58,4.9,.34,14,TEAL if row['prefill_pp_gain_pct']>0 else BLUE,True)
    deck.rect(6.67,1.8,5.89,4.45)
    deck.image(charts/'long_context.png',6.83,2.02,5.55,3.21)
    deck.text('长上下文时，Attention 核心本身\n可以成为很大的执行项。',6.99,5.27,5.1,.75,16,INK,True)
    note('Attention 在布局之间变化小，与 Attention 的绝对成本很小，是两个不同判断。',y=6.49)
    deck.finish_slide()

    deck.slide('一个必要的修正：通信也会间接拖慢 Attention',
        'A100 122B Prefill 的已有 trace：同层、同形状的投影与核心操作对照',
        sources=[SUMMARY/'qwen_prefill_diagnosis/REPORT.md',SUMMARY/'component_reports/Q02.md'],
        notes='数据来自第35层同一kernel形状、同一调用顺序的观测。投影无NCCL重叠0.7718ms、高重叠1.2679ms；FullAttention核心5.0535与5.0619ms。相关性支持通信驻留与SM资源干扰解释，未测精确SM占用和频率，也不是移除通信因果A/B。已有就绪后发送Prefill优化15.08%另有正确性验收；不外推Decode。')
    deck.rect(.78,1.75,7.53,4.55)
    deck.image(charts/'attention_interference.png',.95,2.02,7.18,3.91)
    card(8.55,1.75,4.01,2.05,'Attention 模块也会变', '两槽 PP 的 Attention\n比 EP 约高 11.2%。', ORANGE,19,16)
    card(8.55,4.03,4.01,2.27,'决策中保留调度项', '阶段等待与通信驻留\n可能影响计算效率；\n不能只比较传输字节数。', BLUE,19,15)
    note('NCCL 驻留包含等待；其时长不是纯 NVLink 载荷传输时间。',y=6.49)
    deck.finish_slide()

    deck.slide('把 insight 落成一个简洁的部署决策流程',
        '按目标 workload 和当前软件栈评估候选；避免仅用模型总大小或总 GPU 数查表',
        sources=[A/'analysis/native_metrics.csv',H/'analysis/native_metrics.csv'],
        notes='这是由测量支持的候选筛选流程，并非已训练或验证的自动预测器。每个候选都固定Attention TP/DP、微批次、Graph、后端及精度。通信曲线要按实际域、实际payload采样；当前补测仅两卡NVLink连接，不给出跨节点EP阈值。最终目标可选Prefill TTFT/吞吐，或Decode TPOT/吞吐与SLO。')
    card(.78,1.8,3.77,3.66,'1 先筛容量与通信域', '权重 + KV + 运行缓冲\n是否放得下？\n\nEP 组是否跨域？\nPP 边界位于哪里？', BLUE,20,17)
    card(4.78,1.8,3.77,3.66,'2 比较执行代价', 'Expert 工作集与复用\nDispatch / Combine\nPP 边界、等待与调度\n\nAttention 作为对照。', TEAL,20,17)
    card(8.78,1.8,3.78,3.66,'3 用端到端目标确认', 'Prefill：TTFT / 吞吐\nDecode：TPOT / 吞吐\n\n在可行候选中短测，\n选择满足目标的一组。', ORANGE,20,17)
    deck.rect(.78,5.77,11.78,.67,'#E6EDF9')
    deck.text('通信节省大、Expert 惩罚小 → 倾向 PP；Expert 惩罚或流水依赖更大 → 倾向 EP。',1.0,5.96,11.3,.36,16,INK,True)
    note('判断针对关键路径上未被隐藏的成本；不能把重叠的 GPU 分项直接求和当作预测值。',y=6.63,height=.32,size=10.5)
    deck.finish_slide()

    deck.slide('附录：端到端复测、证据范围与数据入口',
        '可复核的数据、原始失败和适用边界一起保留', sources=sources,
        notes='A100 a100_r2与H100 h100_r2均要求Guard postcheck和清理通过。H100 r1正式记录仍保留，首轮TTFT异常与错误标记的eager诊断未纳入新结果。另有准备阶段a100_r1仅CPU失败，无GPU测量。PPT与PDF、全部图表源数据、逐卡统计、验证报告及生成脚本一同交付。不同模型、阶段、缓存和Graph范围不合并。')
    deck.rect(.78,1.75,6.0,3.7)
    deck.image(charts/'native_throughput.png',.94,1.96,5.68,3.18)
    card(7.02,1.75,5.54,2.08,'已支持', '同样两卡的选择随平台和 workload 改变；\nExpert 与通信的变化值得单独测量。', TEAL,20,15)
    card(7.02,4.02,5.54,2.48,'仍有边界', '不是硬件单变量或模型大小单变量；\nAttention、调度和重叠不能省略；\n跨节点 EP／PP 阈值尚未测出。', ORANGE,20,15)
    note('补测：每配置 3 轮无 profiler；分项各 3 段×2 ranks×24 个稳态 forward。误差条为三轮／三段标准差。',y=5.87,width=6.0,height=.83,size=12)
    deck.finish_slide()

    result = deck.save('PP_EP_deployment_decision_20261026')
    provenance = []
    for path in sources:
        raw = path.read_bytes()
        provenance.append(dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    save_json(out/'sources.json', provenance)
    serializable = dict(data, metrics=[dict(key=list(key), **row) for key,row in metrics.items()],
        components=[dict(key=list(key), **row) for key,row in components.items()], sources=list(map(str,sources)))
    save_json(out/'deck_data.json', serializable)
    save_json(out/'build_result.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'deliverables')
    args=parser.parse_args()
    main(args.out.absolute())
