"""Plot saved component profiles; do not infer end-to-end critical-path shares."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CANONICAL = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')
TWO = ROOT / 'experiments/h100_ep2_pp2_model_weight_analysis_20261026'
EIGHT = CANONICAL / 'glm47_h100_ep8_pp_attribution_20261008'
PREFILL = ROOT / 'experiments/ep2_pp2_qwen_h100_20261025/runs/h100_prefill_components_r1/analysis'
DECODE = ROOT / 'experiments/ep2_pp2_qwen_h100_20261025/runs/h100_ep2_pp2_r7/analysis/component_followup'
E2E = ROOT / 'experiments/h100_ep_pp_comm_domain_comparison_20261026'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sources():
    specs = [
        (TWO / 'absolute_components.csv', 'two_gpu_components.csv'),
        (TWO / 'source_manifest.json', 'two_gpu_parent_sources.json'),
        (TWO / 'validation.json', 'two_gpu_parent_validation.json'),
        (TWO / 'manifest.json', 'two_gpu_parent_manifest.json'),
        (PREFILL / 'comparison.csv', 'two_gpu_prefill_comparison.csv'),
        (PREFILL / 'validation.json', 'two_gpu_prefill_validation.json'),
        (PREFILL / 'REPORT.md', 'two_gpu_prefill_report.md'),
        (DECODE / 'comparison.csv', 'two_gpu_decode_comparison.csv'),
        (DECODE / 'validation.json', 'two_gpu_decode_validation.json'),
        (DECODE / 'REPORT.md', 'two_gpu_decode_report.md'),
        (EIGHT / 'layer_metrics.csv', 'eight_gpu_layer_metrics.csv'),
        (EIGHT / 'manifest.json', 'eight_gpu_manifest.json'),
        (EIGHT / 'report.md', 'eight_gpu_report.md'),
        (E2E / 'full_metrics.csv', 'e2e_full_metrics.csv'),
        (E2E / 'source_manifest.json', 'e2e_parent_sources.json'),
    ]
    records = []
    (HERE / 'sources').mkdir(exist_ok=True)
    for path, name in specs:
        target = HERE / 'sources' / name
        shutil.copy2(path, target)
        assert sha(path) == sha(target)
        records.append(dict(path=str(path), snapshot=str(target.relative_to(HERE)), sha256=sha(path)))
    (HERE / 'source_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    return records


def two_rows():
    rows = []
    keys = ['attention_ms', 'routed_expert_ms', 'shared_expert_ms', 'dispatch_ms',
            'combine_ms', 'nccl_residence_ms', 'other_kernel_ms']
    for source in read_csv(HERE / 'sources/two_gpu_components.csv'):
        if source['model'] != 'Qwen3.5-122B-A10B-FP8':
            continue
        row = {key: (float(source[key]) if source[key] else None) for key in keys}
        row = dict(phase=source['phase'], arm=source['arm'], stage=source['stage'],
                   unit=source['unit'], **row)
        total = sum(row[key] for key in keys if row[key] is not None)
        row['kernel_cumulative_total_ms'] = total
        row['graph_span_ms'] = float(source['graph_span_ms']) if source['graph_span_ms'] else None
        row['deep_ep_dispatch_combine_ms'] = row['dispatch_ms'] + row['combine_ms']
        for key in keys + ['deep_ep_dispatch_combine_ms']:
            row[key.removesuffix('_ms') + '_pct_of_kernel_sum'] = (
                row[key] / total * 100 if row[key] is not None else None)
        row['routed_plus_shared_expert_ms'] = (
            row['routed_expert_ms'] + row['shared_expert_ms']
            if row['shared_expert_ms'] is not None else None)
        assert abs(sum(row[key] or 0 for key in keys) / total - 1) < 1e-12
        if source['phase'] == 'decode':
            assert row['shared_expert_ms'] is None
            assert total > row['graph_span_ms']  # Different streams overlap.
        rows.append(row)
    assert len(rows) == 6
    write_csv(HERE / 'two_gpu_component_shares.csv', rows)
    return rows


def eight_rows():
    samples = read_csv(HERE / 'sources/eight_gpu_layer_metrics.csv')
    keys = ['attention', 'routed_expert', 'deep_ep', 'tp_collective', 'router_shared', 'other']
    assert len(samples) == 24
    for sample in samples:
        assert int(sample['P']) == 2048
        assert int(sample['T']) == (512 if sample['arm'] == 'ep8' else 1024)
        assert abs(sum(float(sample[k]) for k in keys) - float(sample['layer_ms'])) < 1e-4
    rows = []
    for layer in (20, 50):
        for arm in ('ep8', 'pp2ep4'):
            group = [r for r in samples if r['arm'] == arm and int(r['layer']) == layer]
            assert len(group) == (8 if arm == 'ep8' else 4)
            row = dict(arm=arm, layer=layer, rank_layer_samples=len(group),
                       cached_context_tokens=2048, fresh_tokens_per_local_call=512 if arm == 'ep8' else 1024,
                       denominator='mean independently anchored local exclusive layer span; not a global step')
            row['layer_ms'] = statistics.mean(float(r['layer_ms']) for r in group)
            for key in keys:
                row[key + '_ms'] = statistics.mean(float(r[key]) for r in group)
                row[key + '_pct_of_layer'] = row[key + '_ms'] / row['layer_ms'] * 100
            row['expert_and_tp_communication_ms'] = row['deep_ep_ms'] + row['tp_collective_ms']
            row['expert_and_tp_communication_pct_of_layer'] = row['expert_and_tp_communication_ms'] / row['layer_ms'] * 100
            assert abs(sum(row[k + '_pct_of_layer'] for k in keys) - 100) < .005
            rows.append(row)
    write_csv(HERE / 'eight_gpu_prefill_layer_shares.csv', rows)
    return rows


COLORS = dict(attention='#5b8ec1', routed='#d98c37', shared='#efbd77',
              dispatch='#159e89', combine='#79cbb8', nccl='#9571b5', other='#d3dae3',
              deep_ep='#199e89', tp='#9571b5', router_shared='#acb6c3')


def stack(ax, rows, keys, denominator, percent, labels):
    for i, row in enumerate(rows):
        left = 0
        for key, label, color in keys:
            value = row.get(key) or 0
            width = value / row[denominator] * 100 if percent else value
            ax.barh(i, width, left=left, color=color, height=.60, edgecolor='white', linewidth=.6)
            if percent and width >= 7:
                ax.text(left + width / 2, i, f'{width:.1f}%', va='center', ha='center', fontsize=10)
            if not percent and width >= row[denominator] * .075:
                ax.text(left + width / 2, i, f'{value:.2f}' if row[denominator] < 20 else f'{value:.1f}',
                        va='center', ha='center', fontsize=10)
            left += width
        if percent:
            ax.text(102, i, f"{row[denominator]:.3f} ms" if row[denominator] < 20
                    else f"{row[denominator]:.1f} ms", va='center', fontsize=10, color='#405065')
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - .45, -.55)
    ax.set_xlim(0, 126 if percent else max(r[denominator] for r in rows) * 1.04)
    if percent:
        ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Share of summed kernel durations (%)' if percent else 'Cumulative GPU kernel time (ms)')
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=.18)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)


def plot_two(rows):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.6))
    fig.subplots_adjust(top=.78, bottom=.19, left=.13, right=.98, hspace=.65, wspace=.34)
    keys = [('attention_ms', 'Attention', COLORS['attention']),
            ('routed_expert_ms', 'Routed Expert', COLORS['routed']),
            ('shared_expert_ms', 'Shared Expert (Prefill)', COLORS['shared']),
            ('deep_ep_dispatch_combine_ms', 'Dispatch + Combine', COLORS['dispatch']),
            ('nccl_residence_ms', 'NCCL residence / waits', COLORS['nccl']),
            ('other_kernel_ms', 'Other kernels', COLORS['other'])]
    for r, phase in enumerate(('prefill', 'decode')):
        group = [row for row in rows if row['phase'] == phase]
        labels = ['EP2\n2-card mean', 'PP2\nstage 0', 'PP2\nstage 1']
        for col in (0, 1):
            stack(axes[r, col], group, keys, 'kernel_cumulative_total_ms', col == 0, labels)
        if phase == 'prefill':
            axes[r, 0].set_title('Prefill: full modules, per card / whole 8-request batch', fontsize=12, pad=13)
            axes[r, 1].set_title('Prefill: absolute cumulative kernel durations', fontsize=12, pad=13)
        else:
            axes[r, 0].set_title('Decode: per card / one actual Decode Graph', fontsize=12, pad=13)
            axes[r, 1].set_title('Decode: Attention core + routed GEMM only', fontsize=12, pad=13)
    fig.suptitle('H100 / 2 GPUs / Qwen3.5-122B FP8: component profiles', fontsize=19, y=.98)
    fig.text(.5, .91, 'Prefill: 8 requests, no cached context, 4096 new tokens each, output 1.\n'
             'Decode: 8 requests, GPU-cached 4096 each, 1 new token, output 128; two PP slots of 4.',
             ha='center', fontsize=11)
    fig.legend(handles=[Patch(color=color, label=label) for _, label, color in keys],
               loc='upper center', bbox_to_anchor=(.5, .865), ncol=3, frameon=False, fontsize=10)
    fig.text(.04, .112, 'Percent denominator = sum of classified GPU kernel durations, including overlapping kernels; NOT E2E / TTFT / TPOT.\n'
             'Prefill: full Attention and Expert modules. Decode: only Attention cores and routed GEMMs; shared experts and projections remain in Other.\n'
             'PP has no MoE Dispatch/Combine here. Its Graph-external stage transfers, feedback and scheduling are NOT included in Decode bars.\n'
             'NCCL residence includes dependency/control waits; it is not pure link transfer. PP stages are separate, not added as a latency.',
             va='top', fontsize=10, color='#49576b', linespacing=1.5)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(HERE / f'two_gpu_component_shares.{ext}', dpi=175, metadata={'Creator': 'Offline saved-profile analysis'})
    plt.close(fig)


def plot_eight(rows):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.8))
    fig.subplots_adjust(top=.70, bottom=.30, left=.15, right=.98, wspace=.36)
    keys = [('attention_ms', 'Attention', COLORS['attention']),
            ('routed_expert_ms', 'Routed Expert', COLORS['routed']),
            ('deep_ep_ms', 'DeepEP Dispatch + Combine', COLORS['deep_ep']),
            ('tp_collective_ms', 'Attention TP communication', COLORS['tp']),
            ('router_shared_ms', 'Router + shared / mixed', COLORS['router_shared']),
            ('other_ms', 'Other / gaps', COLORS['other'])]
    labels = ['Layer 20 / EP8\n8-rank mean', 'Layer 20 / PP stage 0\n4-rank mean',
              'Layer 50 / EP8\n8-rank mean', 'Layer 50 / PP stage 1\n4-rank mean']
    for i in (0, 1):
        stack(axes[i], rows, keys, 'layer_ms', i == 0, labels)
        axes[i].set_xlabel('Share of local exclusive layer span (%)' if i == 0 else 'Mean local exclusive layer time (ms)')
    axes[0].set_title('Prefill: communication and Expert shares', pad=13)
    axes[1].set_title('Prefill: absolute local layer spans', pad=13)
    fig.suptitle('H100 / 8 GPUs / GLM-4.7 INT4: saved Prefill layer profiles', fontsize=18, y=.97)
    fig.text(.5, .88, 'Related to the 4-request E2E cohort (no cache, 8192 new input tokens each, output 1).\n'
             'Profile at 2048 already processed tokens: EP8 local chunk 512; PP local chunk 1024. Eager; sampled layers 20 / 50.',
             ha='center', fontsize=10.5)
    fig.legend(handles=[Patch(color=color, label=label) for _, label, color in keys],
               loc='upper center', bbox_to_anchor=(.5, .82), ncol=3, frameon=False, fontsize=10)
    fig.text(.04, .213, 'Denominator = independently anchored local layer span, including exposed waits and submission gaps; NOT full-model E2E.\n'
             'Each rank partition is exclusive; means do not create a synchronized global step. Different local token chunks and native libraries remain.\n'
             'DeepEP and TP spans can include waiting. Router/shared is mixed, so it is not counted as routed Expert compute.\n'
             'PP stage-boundary transfers are outside these layer samples and are not zero. Independent communication microbenchmarks are not stacked.',
             va='top', fontsize=10, color='#49576b', linespacing=1.5)
    fig.text(.5, .055, 'Missing: matching 8-GPU Decode compute profiles (4 / 16 requests), and the 16-request Prefill layer breakdown.',
             ha='center', fontsize=11, color='#a14326', weight='bold')
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(HERE / f'eight_gpu_prefill_layer_shares.{ext}', dpi=175, metadata={'Creator': 'Offline saved-profile analysis'})
    plt.close(fig)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows])


def report(two, eight):
    chunks = ['# H100：截图中 EP／PP 实验的通信和 Expert 占比',
        '**这是已保存 profile 的离线补图，没有新增 GPU 实验。现有资料不能把截图全部端到端时间准确拆成 100%：两卡有 Prefill／Decode 内核记录；八卡只有关联的 4 请求 Prefill 单层诊断。**',
        '## 先说明图里到底算了什么',
        '- **Dispatch**：把本轮 token 送给专家；**Combine**：取回并合并专家结果。两者合称专家通信。等待其他 rank 的时间可能包含在内，不能视为纯链路传输。\n'
        '- **路由 Expert**：被路由选中的专家。两卡 Prefill 是整个专家模块，Decode 只识别到 grouped GEMM；两种范围不混算。共享专家仅在 Prefill 单独分类，Decode 的共享专家、投影等保留在“其他”。\n'
        '- **两卡占比分母**：同一行所有已分类 GPU kernel 的累计时长之和。不同 stream 可能重叠，所以它是内核耗时构成，**不是 TTFT、TPOT、端到端延迟的独占占比或 GPU 利用率**。\n'
        '- **八卡占比分母**：同一个 rank 的独立 MoE 层跨度；每行分区互斥，再按层和布局求 rank 均值。跨节点采样未对齐成同一个全局时刻，不能据此拼出完整模型关键路径。\n'
        '- **PP 阶段独立列出**。两卡 PP 的专家本地计算，没有层内跨卡 Dispatch／Combine，但仍有阶段激活传输、元数据和反馈；Decode Graph 外的这些成本不在表中。不能将 0 写成 PP 总通信为零。',
        '## 1. 两张 H100：Qwen3.5-122B-A10B FP8',
        '历史节点 178：EP2 使用 GPU0/1，PP2 使用 GPU2/3，两对都是 NVLink NV6。Attention 均不做张量并行；EP2 是两个 Attention 数据并行副本，专家跨两卡；PP2 是每阶段 24 层、每层所有专家留在本卡，两个 4 请求槽位。FP8 W8A8 DeepGEMM，FA3 全 Attention／Triton 线性 Attention；没有 CPU HiCache、合成 KV 或 MTP。',
        '**Prefill：8 请求，每条没有已缓存前缀，新增输入 4096 token，只生成 1 token。下表单位为毫秒／每卡／整批 8 请求。Decode：同样 8 请求，每条已有 4096 个真实前缀 token 驻留 GPU，新增输入 1 token、生成 128 token。Decode 下表单位为毫秒／每卡／一次实际 Decode Graph；它不是每个最终输出 token 的请求级耗时。**',
        '![两卡通信与专家占比及绝对时间](two_gpu_component_shares.png)']
    for phase in ('prefill', 'decode'):
        chunks.append('### ' + ('Prefill 完整模块' if phase == 'prefill' else 'Decode 核心内核'))
        values = []
        for r in [r for r in two if r['phase'] == phase]:
            label = 'EP2，两卡均值' if r['arm'] == 'ep2' else 'PP2，第' + str(int(r['stage']) + 1) + '阶段'
            values.append([label] + [f'{r[k]:.3f}' if r[k] is not None else '未单列，含于其他' for k in
                          ('attention_ms', 'routed_expert_ms', 'shared_expert_ms', 'dispatch_ms', 'combine_ms',
                           'nccl_residence_ms', 'other_kernel_ms', 'kernel_cumulative_total_ms')] +
                          [f"{r['routed_expert_pct_of_kernel_sum']:.1f}%", f"{r['deep_ep_dispatch_combine_pct_of_kernel_sum']:.1f}%"])
        chunks.append(table(['配置', 'Attention', '路由 Expert', '共享 Expert', 'Dispatch', 'Combine', 'NCCL 驻留', '其他内核',
                             '累计分母 ms', '路由 Expert 占比', 'D+C 占比'], values))
    chunks += ['两卡 Prefill 的 Expert 包含模块内部量化、重排、激活和 GEMM，Attention 包含模块内投影。若合计路由与共享专家，EP2 的 Expert 为 **222.519 ms**，PP 两阶段分别为 **337.401／343.202 ms**。Decode 的 Expert 范围更窄，因此不跨 Prefill／Decode 比较其占比。',
        'NCCL 是 kernel 驻留，包含依赖／控制等待。两卡 Decode 的累计内核总和分别为 EP **12.169 ms**、PP **5.955／6.290 ms**，而 Graph 墙钟跨度为 **9.990／5.053／5.365 ms**；累计值更大，正说明不能直接相加成端到端延迟。EP2 运行原生 overlap，当前 PP 路径不支持；profile 仅 1–2 个 Graph／rank，正式吞吐来自另行进行的三轮请求测量。',
        '## 2. 八张 H100：GLM-4.7-AWQ INT4',
        '节点 174／175，各 4 张 H100。EP8 是节点内 Attention TP4×两个数据并行组，专家跨两节点；PP2×EP4 每节点负责一段层，每段 Attention TP4、专家在节点内 EP4。该诊断关联 4 请求、无旧缓存、每条新增输入 8192 token、输出 1 的 Prefill；在已有 **2048 个已处理 token** 时采样层 20／50。**EP8 每个本地 Attention 调用处理 512 个新 token，PP 本地调用处理 1024 个。** 不是控制了本地 chunk 的对照。',
        '![八卡 Prefill 单层通信与专家占比](eight_gpu_prefill_layer_shares.png)']
    values = []
    for r in eight:
        values.append([r['layer'], 'EP8' if r['arm'] == 'ep8' else 'PP2×EP4', r['rank_layer_samples'],
                       r['fresh_tokens_per_local_call']] + [f'{r[k]:.3f}' for k in
                       ('layer_ms', 'attention_ms', 'routed_expert_ms', 'deep_ep_ms', 'tp_collective_ms', 'router_shared_ms', 'other_ms')] +
                      [f"{r['routed_expert_pct_of_layer']:.1f}%", f"{r['deep_ep_pct_of_layer']:.1f}%", f"{r['expert_and_tp_communication_pct_of_layer']:.1f}%"])
    chunks.append(table(['层', '配置', '独立 rank 样本数', '本地新 token', '层跨度 ms', 'Attention ms', '路由 Expert ms',
                         'D+C ms', 'TP 通信 ms', '路由／共享混合项 ms', '其他 ms', '路由 Expert 占比', 'D+C 占比', 'D+C 与 TP 合计占比'], values))
    chunks += ['这里展示完整 24 个已接受 rank-layer 样本的分层均值，不只挑单个 rank。DeepEP 包含暴露等待与 API／提交间隙；路由／共享混合项不能可靠拆成纯共享 Expert，故没有放进路由 Expert。PP 边界通信在层之外，没有把独立微基准的中位数加进层图。',
        '该历史真实模型对照的 EP 用 custom DeepEP，PP 用 installed DeepEP，Expert 组织也有不同；源 manifest 保留具体路径／哈希。EP 样本来自已接受的独立 EP arm，但其外层 r3 Guard 原失败状态保留；PP 来自后续 r4。不能声称两个 arm 来自同一次共同通过的测试，不能把各层样本乘 89 得到完整模型差额。',
        '## 3. 截图各行的 profile 覆盖范围',
        table(['截图条目', 'Prefill 分项', 'Decode 分项'], [
            ['两卡 Qwen，8 请求', '有，独立补采整批所有 Prefill chunk 的逐卡 kernel 归属', '有，独立短采实际 Decode Graph；分类范围较窄'],
            ['八卡 GLM，4 请求', '有关联单层诊断；层 20／50，各卡独立样本，非全模型占比', '未找到匹配的模型计算分项；只有端到端指标和独立纯通信'],
            ['八卡 GLM，16 请求', '未找到匹配的真实模型分项', '未找到匹配的模型计算分项；只有端到端指标和独立纯通信']]),
        '**因此不能画出八卡 Decode 的“Expert 占多少、通信占多少”的可靠 100% 图，也不能用另一个通信微基准填补空白。** 本次没有为缺失项启动新实验。',
        '## 4. 与截图端到端指标的关系',
        '端到端指标和 profile 是分开采的。原图两卡 Decode EP2 吞吐 636.57、PP2 509.77 输出 token/s；八卡 4 请求 EP8 130.60、PP 166.37；16 请求 EP8 429.39、PP 518.09。它们仍是完整配置结果，不能把本报告的局部比例当作这些吞吐差额的排他归因。',
        '[原图和完整 TTFT／TPOT／端到端表](' + str(E2E / 'REPORT.md') + ') · [本报告保留的 12 行指标](full_metrics.csv) · '
        '[两卡全部分项和占比](two_gpu_component_shares.csv) · [八卡单层占比](eight_gpu_prefill_layer_shares.csv) · '
        '[原始文件路径和 SHA256](source_manifest.json) · [校验](validation.json) · [重生成脚本](build_report.py)。',
        '关键观察：两卡 NVLink 条件下，路由 Expert 的额外本地成本与省掉的专家通信同量级；八卡早期 Prefill 的局部层记录里，通信和等待明显更重。这支持关注通信域与 Expert 计算的取舍，但尚未提供八卡 Decode 的完整端到端归因。']
    (HERE / 'REPORT.md').write_text('\n\n'.join(chunks) + '\n')


def main():
    records = sources()
    two, eight = two_rows(), eight_rows()
    plot_two(two)
    plot_eight(eight)
    shutil.copy2(HERE / 'sources/e2e_full_metrics.csv', HERE / 'full_metrics.csv')
    assert len(read_csv(HERE / 'full_metrics.csv')) == 12
    report(two, eight)
    validation = dict(passed=True, new_gpu_measurements=0, sources=len(records),
                      two_gpu_component_rows=len(two), eight_gpu_rank_layer_samples=24,
                      eight_gpu_layer_aggregate_rows=len(eight), formal_metric_rows=12,
                      denominators_checked=True, exclusive_layer_partition_tolerance_ms=.0001,
                      missing_eight_gpu_decode_marked=True, kernel_overlap_not_turned_into_e2e_shares=True,
                      pp_graph_external_communication_not_zero_filled=True,
                      independent_communication_microbenchmark_not_stacked=True,
                      source_hashes_match=all(sha(Path(r['path'])) == r['sha256'] for r in records))
    assert validation['source_hashes_match']
    (HERE / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    manifest = dict(id=HERE.name, title='H100 saved EP/PP component shares, with explicit measurement boundaries',
                    created_at='2026-10-26', status='complete_offline_partial_profile_coverage',
                    new_gpu_measurements=0, source_manifest='source_manifest.json',
                    implementation='Saved original engine/native revisions and rank mapping retained in source manifests; no engine change',
                    workload='Exact two-GPU and eight-GPU cohort definitions in REPORT.md; scopes remain separate',
                    artifacts=['REPORT.md', 'full_metrics.csv', 'two_gpu_component_shares.csv',
                               'eight_gpu_prefill_layer_shares.csv', 'validation.json'],
                    limitations=['Not an end-to-end exclusive decomposition', 'No saved matching 8-GPU Decode compute profile',
                                 '8-GPU Prefill local chunks differ', '2-GPU Decode only core Attention and routed GEMM'])
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (HERE / 'summary.md').write_text('# H100 saved profile shares\n\nOffline plots only; no GPU launch. Two-GPU full-model Prefill and short Decode Graph kernel-duration composition, plus eight-GPU Prefill exclusive local-layer profiles. All denominators and missing Decode coverage explicit.\n\n[Report](REPORT.md) · [Validation](validation.json).\n')
    print(json.dumps(validation))


if __name__ == '__main__':
    main()
