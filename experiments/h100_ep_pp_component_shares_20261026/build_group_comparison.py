"""Combine PP stages while retaining the limits of unmatched historical samples."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent


def read(name):
    with (HERE / name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write(name, rows):
    with (HERE / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def two_gpu():
    source = read('two_gpu_component_shares.csv')
    keys = ['attention_ms', 'routed_expert_ms', 'shared_expert_ms',
            'deep_ep_dispatch_combine_ms', 'nccl_residence_ms', 'other_kernel_ms']
    result = []
    for phase in ('prefill', 'decode'):
        for arm in ('ep2', 'pp2'):
            group = [r for r in source if r['phase'] == phase and r['arm'] == arm]
            # Prefill source: already the whole global-B8 cohort, per card.
            # Decode source: one B4 Graph; PP needs two slots per stage for B8.
            factor = 2 if arm == 'ep2' or phase == 'decode' else 1
            row = dict(phase=phase, configuration='EP2' if arm == 'ep2' else 'PP2 (combined)',
                       scope='two-GPU cumulative kernel work / global B8 Prefill cohort' if phase == 'prefill'
                       else 'derived two-GPU kernel work / B8-equivalent Decode advance; not measured wall time',
                       source_rows=len(group), multiplier_per_source_row=factor)
            for key in keys:
                row[key] = (sum(float(r[key]) for r in group) * factor
                            if all(r[key] != '' for r in group) else None)
            row['total_ms'] = sum(row[key] or 0 for key in keys)
            for key in keys:
                row[key[:-3] + '_pct'] = row[key] / row['total_ms'] * 100 if row[key] is not None else None
            result.append(row)
    assert len(result) == 4
    write('two_gpu_combined_group_work.csv', result)
    return result


def eight_gpu():
    source = read('sources/eight_gpu_layer_metrics.csv')
    keys = ['attention', 'routed_expert', 'deep_ep', 'tp_collective', 'router_shared', 'other']
    result = []
    for arm in ('ep8', 'pp2ep4'):
        group = [r for r in source if r['arm'] == arm]
        assert len(group) == (16 if arm == 'ep8' else 8)
        assert set(r['layer'] for r in group) == {'20', '50'}
        row = dict(configuration='EP8' if arm == 'ep8' else 'PP2 x EP4 (combined)',
                   sampled_layers='20 and 50', independent_rank_layer_samples=len(group),
                   scope='sum of independently anchored local spans from two sampled layers; not full stages or global-step latency',
                   cached_context_tokens=2048,
                   fresh_tokens_per_local_call=512 if arm == 'ep8' else 1024)
        row['total_ms'] = sum(float(r['layer_ms']) for r in group)
        for key in keys:
            row[key + '_ms'] = sum(float(r[key]) for r in group)
            row[key + '_pct'] = row[key + '_ms'] / row['total_ms'] * 100
        row['routed_expert_mean_per_rank_layer_ms'] = row['routed_expert_ms'] / len(group)
        row['local_experts_per_gpu_per_layer'] = 20 if arm == 'ep8' else 40
        assert abs(sum(row[k + '_ms'] for k in keys) - row['total_ms']) < .001
        result.append(row)
    write('eight_gpu_combined_sample_work.csv', result)
    return result


TWO_KEYS = [('attention_ms', 'Attention', '#5b8ec1'),
            ('routed_expert_ms', 'Routed Expert', '#d98c37'),
            ('shared_expert_ms', 'Shared Expert (Prefill)', '#efbd77'),
            ('deep_ep_dispatch_combine_ms', 'Dispatch + Combine', '#159e89'),
            ('nccl_residence_ms', 'NCCL residence / waits', '#9571b5'),
            ('other_kernel_ms', 'Other kernels', '#d3dae3')]
EIGHT_KEYS = [('attention_ms', 'Attention', '#5b8ec1'),
              ('routed_expert_ms', 'Routed Expert', '#d98c37'),
              ('deep_ep_ms', 'Dispatch + Combine', '#159e89'),
              ('tp_collective_ms', 'Attention TP communication', '#9571b5'),
              ('router_shared_ms', 'Router / shared mixed', '#acb6c3'),
              ('other_ms', 'Other / gaps', '#d3dae3')]


def panel(ax, rows, keys, percent, labels):
    for i, row in enumerate(rows):
        left = 0
        for key, _, color in keys:
            value = row[key] or 0
            width = value / row['total_ms'] * 100 if percent else value
            ax.barh(i, width, left=left, height=.52, color=color, edgecolor='white', linewidth=.6)
            if width >= (7 if percent else .075 * max(r['total_ms'] for r in rows)):
                label = f'{width:.1f}%' if percent else f'{value:.2f}'
                ax.text(left + width / 2, i, label, ha='center', va='center', fontsize=11)
            left += width
        if not percent:
            ax.text(row['total_ms'] * 1.012, i, f"{row['total_ms']:.2f} ms", va='center', fontsize=10)
    ax.set_yticks(range(len(rows)), labels)
    ax.set_ylim(1.5, -.5)
    ax.set_xlim(0, 100 if percent else max(r['total_ms'] for r in rows) * 1.21)
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=.18)
    if percent:
        ax.set_xticks([0, 25, 50, 75, 100])
    for name in ('top', 'right'):
        ax.spines[name].set_visible(False)


def save(fig, stem):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(HERE / f'{stem}.{ext}', dpi=175)
    plt.close(fig)


def plots(two, eight):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2))
    fig.subplots_adjust(left=.14, right=.97, top=.67, bottom=.29, wspace=.40)
    fig.suptitle('8 H100s: EP8 vs PP2 x EP4 - PP stages combined', y=.975, fontsize=18)
    fig.text(.5, .88, 'GLM-4.7 INT4 / Prefill / sampled layers 20 and 50 / 2048 already processed context tokens\n'
             'EP8: 16 independent rank-layer samples, 512 new tokens per local call. PP: 8 samples, 1024 new tokens per local call.',
             ha='center', fontsize=10.5)
    fig.legend(handles=[Patch(color=c, label=n) for _, n, c in EIGHT_KEYS], loc='upper center',
               bbox_to_anchor=(.5, .82), ncol=3, frameon=False, fontsize=10)
    labels = ['EP8\n16 rank-layer records', 'PP2 x EP4\n8 records, combined']
    panel(axes[0], eight, EIGHT_KEYS, True, labels)
    panel(axes[1], eight, EIGHT_KEYS, False, labels)
    axes[0].set_title('Composition of the sampled duration sum', pad=15)
    axes[1].set_title('Independent sampled duration sums', pad=15)
    axes[0].set_xlabel('Share of summed local spans (%)')
    axes[1].set_xlabel('Sum of sampled rank-local spans (ms)')
    fig.text(.04, .21, 'Routed Expert per rank-layer: EP8 = 0.256 ms; PP = 0.494 ms (1.93x). The 4.09 vs 3.95 ms sums do NOT show a PP Expert speedup.\n'
             'Both configurations sum ALL recorded ranks of the SAME two model layers; EP has twice as many recorded rank-layer samples.\n'
             'These are sums of independently captured local CUDA/API spans, including waits/gaps. They are NOT full-model or global-step latency.\n'
             'Local chunks, routing loads and native libraries differ. PP stage-boundary transfer is outside the sampled layers.\n'
             'No matching 8-GPU Decode compute profile is available; no missing costs are estimated.',
             va='top', fontsize=10, color='#48576a', linespacing=1.5)
    save(fig, 'eight_gpu_combined_comparison')

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10))
    fig.subplots_adjust(left=.13, right=.97, top=.76, bottom=.19, wspace=.40, hspace=.65)
    fig.suptitle('2 H100s: EP2 vs PP2 - PP stages combined', y=.975, fontsize=18)
    fig.text(.5, .89, 'Qwen3.5-122B FP8 / same global batch of 8 requests / all-GPU kernel work\n'
             'Prefill: no cache, 4096 new tokens each. Decode: GPU-cached 4096 each, 1 new token, output 128.',
             ha='center', fontsize=11)
    fig.legend(handles=[Patch(color=c, label=n) for _, n, c in TWO_KEYS], loc='upper center',
               bbox_to_anchor=(.5, .85), ncol=3, frameon=False, fontsize=10)
    for i, phase in enumerate(('prefill', 'decode')):
        rows = [r for r in two if r['phase'] == phase]
        for j in (0, 1):
            panel(axes[i, j], rows, TWO_KEYS, j == 0, ['EP2\nboth GPUs', 'PP2\ncombined'])
        axes[i, 0].set_title(phase.capitalize() + ': all-GPU kernel-time composition', pad=14)
        axes[i, 1].set_title(phase.capitalize() + ': all-GPU kernel work', pad=14)
        axes[i, 0].set_xlabel('Share of summed kernel durations (%)')
        axes[i, 1].set_xlabel('Cumulative GPU kernel time (ms)')
    fig.text(.04, .12, 'Prefill: EP = 2 x per-card mean; PP = stage 0 + stage 1, covering the whole B8 cohort.\n'
             'Decode B8-equivalent: EP = 2 x B4-Graph mean; PP = 2 slots x (stage 0 + stage 1). Derived from short Graph means.\n'
             'Cumulative kernels can overlap; these totals are NOT E2E / TTFT / TPOT. Decode only identifies Attention cores and routed GEMMs.\n'
             'PP Decode Graph-external stage transfers/feedback are not included. No layer-expert D+C does not mean zero total communication.',
             va='top', fontsize=10, color='#48576a', linespacing=1.5)
    save(fig, 'two_gpu_combined_comparison')


def report(two, eight):
    text = '''# PP 合并后，直接对比 EP

两种配置都汇总全部相关 GPU，而不是只把 PP 相加、EP 却仍用每卡均值。所有数字都从原始已保存的 profile 重算，没有新 GPU 实验。

## 1. 八卡：EP8 对 PP2×EP4

**这组目前只有 Prefill 第 20／50 层的诊断记录，没有完整两阶段计算 profile。** 以下将两层所有采样卡的本地分项相加，PP 合成一条；不是“完整 PP 两阶段一共只用这些时间”。

每请求 Prefill 原负载无旧缓存、8192 新输入、输出 1；采样时已有 2048 个已处理 token。EP8 每个本地 Attention 调用处理 512 个新 token，两个 Attention 数据并行组；PP 每个本地调用处理 1024 个新 token。Attention 都在节点内 TP4。层 20／50：EP 各采 8 卡，共 16 份记录；PP 每层只在其所属阶段采 4 卡，共 8 份记录。分别合并这些记录，不混合两种配置、不把 EP 当成单卡。

![八卡 EP8 与合并 PP 对比](eight_gpu_combined_comparison.png)

单位：**采样层各 rank 本地 CUDA/API 跨度的累计毫秒**，可以含等待和提交空档；括号内为各配置采样跨度之和的占比。

| 分项 | EP8，所有采样 rank 合计 | PP2×EP4，两阶段采样合并 |
|---|---:|---:|
'''
    for key, label in [('attention', 'Attention'), ('routed_expert', '路由 Expert'),
                       ('deep_ep', 'Dispatch＋Combine'), ('tp_collective', 'Attention TP 通信'),
                       ('router_shared', '路由／共享混合项'), ('other', '其他／间隙')]:
        text += '| ' + label + ' | ' + ' | '.join(f"{r[key+'_ms']:.3f}（{r[key+'_pct']:.1f}%）" for r in eight) + ' |\n'
    text += '| 采样跨度合计 | ' + ' | '.join(f"{r['total_ms']:.3f}" for r in eight) + ' |\n'
    text += '''
**这个合计不是两组的请求延迟或同一全局时刻的执行耗时。** 跨节点采样不对齐，本地新 token 数和 native 通信库也不同，不能据此声称 PP 端到端快两倍。PP 阶段边界传输在层外，没有填成零；路由／共享混合项没有冒充纯 Expert 计算。八卡 Decode 以及 16 请求 Prefill 的匹配计算 profile 仍缺失。

### 为什么图里的路由 Expert 合计看起来 PP 更短？

**这里不能得出“PP 的路由 Expert 算得更快”。** 此前只强调合计数，没有把每份记录的成本同时列出来，容易误导。拆开统计口径如下：

| 配置 | 每卡每层存放的路由专家数 | 本地 Attention 每次新增 token | 被加总的单卡、单层记录数 | 每份记录的路由 Expert 平均耗时 | 图中的路由 Expert 合计 |
|---|---:|---:|---:|---:|---:|
'''
    for r in eight:
        text += (f"| {r['configuration']} | {r['local_experts_per_gpu_per_layer']} | "
                 f"{r['fresh_tokens_per_local_call']} | {r['independent_rank_layer_samples']} | "
                 f"{r['routed_expert_mean_per_rank_layer_ms']:.6f} ms | {r['routed_expert_ms']:.6f} ms |\n")
    expert_ratio = eight[1]['routed_expert_mean_per_rank_layer_ms'] / eight[0]['routed_expert_mean_per_rank_layer_ms']
    sum_change = (eight[1]['routed_expert_ms'] / eight[0]['routed_expert_ms'] - 1) * 100
    text += f'''
单看一次被采样的“每卡、每层”，PP 实测是 EP 的 **{expert_ratio:.2f} 倍**，并没有更短。但 EP 对两层各采八卡，共 16 份；PP 每层只归属于一个阶段、各采四卡，共 8 份。所以图中出现 **16×0.255803≈4.093 ms** 对 **8×0.493588≈3.949 ms**。累计的是各卡跨度之和，不是流水线墙钟时间。

合计只差 **{abs(sum_change):.2f}%**。这不是严格同 token、同路由的 Expert 速度对照：EP 每个 Attention 组一次新增 512 token，PP 为 1024；各卡捕获的全局轮次不对齐，实际专家接收量和分布也不同，且这些是带提交间隙的本地 CUDA/API 跨度。因此，不能把这点差异解释成 PP 的 Expert 内核优势，也不能认定是已量化的测量噪声。要判断相同负载下的纯 Expert 代价，应使用固定 token、固定路由和相同计算后端的单层隔离实验；本图只保留这批历史采样的真实数值及其范围限制。

## 2. 两卡：为什么原图看起来 PP 有两倍工作？

原图 Prefill 的 EP 一条是**两卡平均**，PP 却分别列了两卡。如果把 PP 相加，EP 也必须乘 2。按相同 8 请求、两张 GPU 的累计内核工作比较，Attention 基本相同，总工作量也没有翻倍。

![两卡 EP2 与合并 PP 对比](two_gpu_combined_comparison.png)

| 阶段 | 分项 | EP2，两卡合计 ms | PP2，两阶段合并 ms |
|---|---|---:|---:|
'''
    for phase in ('prefill', 'decode'):
        rows = [r for r in two if r['phase'] == phase]
        for key, label in [('attention_ms', 'Attention'), ('routed_expert_ms', '路由 Expert'),
                           ('deep_ep_dispatch_combine_ms', 'Dispatch＋Combine'), ('total_ms', '已统计内核累计')]:
            text += '| ' + phase.capitalize() + ' | ' + label + ' | ' + ' | '.join(f'{r[key]:.3f}' for r in rows) + ' |\n'
    text += '''
- Prefill 单位是整批 8 请求的全部 GPU kernel 累计：**EP 每卡均值×2；PP 两阶段相加**。PP 总计约 1254 ms，EP 约 1286 ms，不是 PP 比 EP 多一倍。
- Decode 原始单位不同，是每卡一次实际 Graph，实际每个微批 4 请求。换算成同一批 8 请求推进一次的工作：**EP 两个数据并行 Graph 相加；PP 两个槽各经过两个阶段，所以是 2×（stage 0＋stage 1）**。这是由短 profile 均值推导的工作量，不是实测同步轮次或 TPOT。
- Prefill 是完整 Attention／Expert 模块；Decode 只识别 Attention 核心与路由 GEMM，共享专家／投影等仍在其他。两者不能混算模块占比。
- 两卡的路由 Expert 代价在 PP 中确实更高，但 EP 有额外专家通信。内核累计能重叠，流水也能重叠；不能用累计之和直接预测端到端速度。完整配置的历史两卡 Prefill 吞吐是 EP 40648.91／PP 43246.90 输入 token/s；Decode 是 EP 636.57／PP 509.77 输出 token/s，PP 并非总是更快。

[合并统计 CSV：八卡](eight_gpu_combined_sample_work.csv) · [两卡](two_gpu_combined_group_work.csv) · [分母与倍率校验](group_validation.json) · [原始来源与哈希](source_manifest.json) · [旧逐阶段明细及限制](REPORT.md)。
'''
    (HERE / 'GROUP_COMPARISON.md').write_text(text)


def main():
    original = json.loads((HERE / 'validation.json').read_text())
    assert original['passed']
    manifests = json.loads((HERE / 'source_manifest.json').read_text())
    for r in manifests:
        path = HERE / r['snapshot']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r['sha256']
    two, eight = two_gpu(), eight_gpu()
    plots(two, eight)
    report(two, eight)
    evidence = dict(passed=True, new_gpu_runs=0, combined_two_gpu_rows=4, combined_eight_gpu_rows=2,
                    all_pp_stages_combined=True, both_ep_and_pp_whole_group_aggregation=True,
                    eight_gpu_sample_counts={'ep8':16,'pp2ep4':8},
                    eight_gpu_not_claimed_complete_model_or_synchronized_step=True,
                    eight_gpu_expert_mean_per_rank_layer_ms={r['configuration']:r['routed_expert_mean_per_rank_layer_ms'] for r in eight},
                    eight_gpu_expert_sums_not_a_controlled_speed_comparison=True,
                    two_gpu_decode_pp_two_slots_counted=True,
                    missing_eight_gpu_decode_preserved=True)
    (HERE / 'group_validation.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence))


if __name__ == '__main__':
    main()
