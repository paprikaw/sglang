"""Offline A100 EP/PP comparison; never imports torch or launches GPU work."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MANAGER = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')
INDEX = ROOT / 'experiments/experiment_summary'
COMM = ROOT / 'experiments/single_layer_ep4_ep8_matched_20261024/ce_rdma_r1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def csv_out(name, records):
    with (HERE / name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def md_table(headers, data):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in data])


def save(fig, stem):
    for ext in ['png', 'svg', 'pdf']:
        meta = ({'Software': 'Offline A100 historical analysis'} if ext == 'png' else
                {'Date': None} if ext == 'svg' else {'CreationDate': None, 'ModDate': None})
        fig.savefig(HERE / f'{stem}.{ext}', dpi=180, metadata=meta)
    plt.close(fig)


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / 'sources').mkdir(exist_ok=True)
    paths = {
        'qwen122_metrics.csv': INDEX / 'qwen_ep2_pp2_metrics.csv',
        'qwen122_report.md': INDEX / 'component_reports/Q01.md',
        'qwen122_ready_send_report.md': INDEX / 'component_reports/Q02.md',
        'qwen35_metrics.csv': ROOT / 'experiments/ep2_pp2_qwen35b_a100_20261025/runs/a100_qwen35b_ep2_pp2_r2/analysis/full_metrics.csv',
        'qwen35_report.md': INDEX / 'component_reports/Q06.md',
        'glm_eight_metrics.csv': MANAGER / 'glm47_a100_equal_local_chunk_20261010_r1/full_metrics.csv',
        'glm_eight_summary.md': MANAGER / 'glm47_a100_equal_local_chunk_20261010_r1/summary.md',
        'glm_eight_manifest.json': MANAGER / 'glm47_a100_equal_local_chunk_20261010_r1/manifest.json',
        'comm_comparison.csv': COMM / 'transport_comparison.csv',
        'comm_phases.csv': COMM / 'phase_metrics.csv',
        'comm_samples.csv': COMM / 'full_metrics.csv',
        'comm_validation.json': COMM / 'independent_validation.json',
        'comm_manifest.json': COMM / 'manifest.json',
        'comm_summary.md': COMM / 'summary.md',
        'old_comm_samples.csv': MANAGER / 'a100_ep_locality_causal_20261010_r1/full_metrics.csv',
        'old_comm_summary.md': MANAGER / 'a100_ep_locality_causal_20261010_r1/summary.md',
    }
    provenance = []
    for name, path in paths.items():
        target = HERE / 'sources' / name
        shutil.copy2(path, target)
        assert sha(path) == sha(target)
        provenance.append(dict(path=str(path), snapshot='sources/' + name, sha256=sha(path)))
    (HERE / 'source_manifest.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    full = []
    checked_raw = set()
    # Use the same two-slot configuration for both phases, not a phase-picked best PP.
    for r in rows(HERE / 'sources/qwen122_metrics.csv'):
        if r['configuration'] not in ['EP2', 'PP2，两槽']:
            continue
        raw = Path(r['source_path'])
        assert sha(raw) == r['source_sha256']
        checked_raw.add(str(raw))
        full.append(dict(case='qwen122_2gpu', model='Qwen3.5-122B-A10B FP8', gpus=2,
            nodes=1, phase=r['workload'], layout='EP2' if r['configuration'] == 'EP2' else 'PP2',
            requests=8, formal_rounds=3, throughput=float(r['throughput']),
            throughput_unit=r['throughput_kind'], cached_tokens_per_request=int(r['cached_tokens_per_request']),
            cache_location='GPU' if r['workload'] == 'decode' else 'none',
            new_input_tokens_per_request=int(r['fresh_input_tokens_per_request']),
            output_tokens_per_request=int(r['output_tokens_per_request']),
            cuda_graph=False, native_expert_transport='DeepEP normal NVLink / no exchange in PP',
            **{k: r[k] for k in ['ttft_s_mean', 'ttft_s_median', 'ttft_s_p99',
                'tpot_ms_mean', 'tpot_ms_median', 'tpot_ms_p99', 'e2e_s_mean', 'e2e_s_median', 'e2e_s_p99']},
            source_path=r['source_path'], source_sha256=r['source_sha256']))
    for r in rows(HERE / 'sources/qwen35_metrics.csv'):
        raw = Path(r['source'])
        assert sha(raw) == r['sha256']
        checked_raw.add(str(raw))
        full.append(dict(case='qwen35_2gpu', model='Qwen3.6-35B-A3B BF16', gpus=2,
            nodes=1, phase=r['phase'], layout=r['arm'].upper(), requests=8, formal_rounds=3,
            throughput=float(r['tokens_per_s']), throughput_unit='input_tokens/s' if r['phase'] == 'prefill' else 'output_tokens/s',
            cached_tokens_per_request=int(r['cached_context']), cache_location='GPU' if r['phase'] == 'decode' else 'none',
            new_input_tokens_per_request=int(r['new_input']), output_tokens_per_request=int(r['output']),
            cuda_graph=False, native_expert_transport='DeepEP normal NVLink / no exchange in PP',
            **{k: r[k] for k in ['ttft_s_mean', 'ttft_s_median', 'ttft_s_p99',
                'tpot_ms_mean', 'tpot_ms_median', 'tpot_ms_p99', 'e2e_s_mean', 'e2e_s_median', 'e2e_s_p99']},
            source_path=r['source'], source_sha256=r['sha256']))
    for r in rows(HERE / 'sources/glm_eight_metrics.csv'):
        phase = r['phase']
        full.append(dict(case='glm_8gpu', model='GLM-4.7-AWQ INT4', gpus=8,
            nodes=2, phase=phase, layout='EP8' if r['arm'] == 'ep8' else 'PP2xEP4',
            requests=4, formal_rounds=1, throughput=float(r['input_tok_s' if phase == 'prefill' else 'output_tok_s']),
            throughput_unit='input_tokens/s' if phase == 'prefill' else 'output_tokens/s',
            cached_tokens_per_request=0, cache_location='none',
            new_input_tokens_per_request=int(r['input_tokens_per_request']),
            output_tokens_per_request=int(r['output_tokens_per_request']),
            cuda_graph=phase == 'decode', native_expert_transport='9dc34887 pair NVLink + nonpair RDMA',
            **{k: r[k] for k in ['ttft_s_mean', 'ttft_s_median', 'ttft_s_p99',
                'tpot_ms_mean', 'tpot_ms_median', 'tpot_ms_p99', 'e2e_s_mean', 'e2e_s_median', 'e2e_s_p99']},
            source_path=str(paths['glm_eight_metrics.csv']), source_sha256=sha(paths['glm_eight_metrics.csv'])))
    assert len(full) == 12 and len(checked_raw) == 8
    comparisons = []
    for case in ['qwen122_2gpu', 'qwen35_2gpu', 'glm_8gpu']:
        for phase in ['prefill', 'decode']:
            e = next(r for r in full if r['case'] == case and r['phase'] == phase and r['layout'].startswith('EP'))
            p = next(r for r in full if r['case'] == case and r['phase'] == phase and r['layout'].startswith('PP'))
            comparisons.append(dict(case=case, phase=phase, model=e['model'], gpus=e['gpus'],
                ep_rate=e['throughput'], pp_rate=p['throughput'], ep_over_pp=e['throughput'] / p['throughput'],
                pp_vs_ep_pct=100 * (p['throughput'] / e['throughput'] - 1)))
    csv_out('full_metrics.csv', full)
    csv_out('e2e_comparisons.csv', comparisons)

    validation = json.loads((HERE / 'sources/comm_validation.json').read_text())
    assert validation['passed']
    phases = rows(HERE / 'sources/comm_phases.csv')
    samples = rows(HERE / 'sources/comm_samples.csv')
    assert len(phases) == 280 and len(samples) == 840
    fields = ['phase', 'model', 'transport', 'world', 'cohort', 'global_batch', 'call_unique_tokens', 'offset']
    for r in phases:
        aligned = [x for x in samples if all(x[k] == r[k] for k in fields)]
        assert len(aligned) == 3
        value = statistics.median(float(x['pair_cuda_ms']) for x in aligned)
        assert abs(value - float(r['pair_cuda_ms_median_ms'])) < 1e-8
    transport = rows(HERE / 'sources/comm_comparison.csv')
    selected = [r for r in transport if r['model'] == 'glm' and r['boundary'] == 'equal_call']
    assert len(selected) == 20
    # Native identity and route identity must match RDMA/CE within each actual call.
    groups = {}
    for r in phases:
        key = (r['phase'].rsplit('_', 1)[0],) + tuple(r[k] for k in
            ['model', 'world', 'cohort', 'global_batch', 'call_unique_tokens', 'offset'])
        groups.setdefault(key, []).append(r)
    for group in groups.values():
        assert len(group) == 2
        assert {r['transport'] for r in group} == {'rdma', 'ce'}
        for k in ['native_sha256', 'route_sha256', 'capacity', 'cuda_graph', 'assignments_per_gpu']:
            assert len({r[k] for r in group}) == 1
    csv_out('communication_equal_call.csv', selected)
    old = rows(HERE / 'sources/old_comm_samples.csv')
    old_medians = [dict(layout=layout, global_tokens=n,
                       median_ms=statistics.median(float(r['pair_cuda_ms']) for r in old
                           if r['phase'] == layout and int(r['global_tokens']) == n))
                   for n in [4, 1024] for layout in ['local168', 'local169', 'cross4', 'cross8']]
    csv_out('old_rdma_communication.csv', old_medians)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': HERE.name, 'axes.titleweight': 'bold'})
    colors = ['#20897c', '#5c9cba', '#8b62cb']
    names = ['2 GPUs: Qwen122B FP8\nNVLink; EP2 vs PP2',
             '2 GPUs: Qwen35B BF16\nNVLink; EP2 vs PP2',
             '8 GPUs: GLM INT4\n2 nodes; EP8 vs PP2xEP4']
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 6.3), sharey=True)
    for ax, phase in zip(axes, ['prefill', 'decode']):
        cs = [r for r in comparisons if r['phase'] == phase]
        vals = [r['ep_over_pp'] for r in cs]
        y = np.arange(3)
        ax.barh(y, vals, color=colors, height=.53)
        ax.axvline(1, color='#45454f', linestyle='--', linewidth=1.2)
        for yi, r in enumerate(cs):
            ax.text(r['ep_over_pp'] + .025, yi, f"{r['ep_over_pp']:.3f}  ({100*(r['ep_over_pp']-1):+.1f}%)", va='center', fontsize=10)
        ax.set_xlim(0, 1.4)
        ax.set_yticks(y, names)
        ax.set_title('Prefill: input throughput' if phase == 'prefill' else 'Decode: output throughput', pad=13)
        ax.set_xlabel('EP throughput / same-cohort PP throughput (PP = 1)')
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle('A100: comparable experiment types, but no H100-like Decode preference reversal', fontsize=14, y=.97)
    fig.text(.02, .055, '2-GPU workloads: 8 requests; Prefill 4096 fresh -> 1 output; Decode 4096 GPU-cached + 1 fresh -> 128 output.\n'
             '8-GPU workloads: 4 requests; Prefill 8192 fresh -> 1 output; Decode 128 fresh -> 512 output.\n'
             'Separate historical cohorts; different models, kernels and Graph modes. Two-slot PP kept fixed for both 2-GPU phases.', fontsize=9)
    fig.subplots_adjust(left=.27, right=.985, top=.84, bottom=.24, wspace=.11)
    save(fig, 'e2e_communication_domain')

    # Preserve both independently measured local nodes; no pooled median or best-of selection.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True)
    for ax, node in zip(axes, ['065', '167']):
        rs = sorted([r for r in selected if r['ep4_node'] == node], key=lambda r: int(r['global_batch_tokens']))
        x = [int(r['global_batch_tokens']) for r in rs]
        for key, label, color, style, marker in [
                ('ep4_rdma_ms', 'Local EP4: pair NVLink + RDMA', '#20897c', '--', 'o'),
                ('ep4_ce_ms', 'Local EP4: pair NVLink + PCIe CE', '#20897c', '-', 's'),
                ('ep8_rdma_ms', 'Cross-node EP8: RDMA policy', '#8b62cb', '--', 'o'),
                ('ep8_ce_ms', 'Cross-node EP8: local CE + remote RDMA', '#8b62cb', '-', 's')]:
            ax.plot(x, [float(r[key]) for r in rs], label=label, color=color,
                    linestyle=style, marker=marker, markersize=4, linewidth=1.7)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks([8, 32, 128, 512, 2048, 4096], ['8', '32', '128', '512', '2048', '4096'])
        ax.set_xlabel('Unique input tokens in each communication call')
        ax.set_title(f'Local EP4 on {node}; EP8 across 065 / 167', fontsize=11)
        ax.grid(which='both', alpha=.16)
        point = next(r for r in rs if r['global_batch_tokens'] == '1024')
        ax.text(.04, .95, f"1024 tokens, CE policy:\nEP4 {float(point['ep4_ce_ms']):.3f} ms; EP8 {float(point['ep8_ce_ms']):.3f} ms\nratio {float(point['ep8_ce_ms'])/float(point['ep4_ce_ms']):.2f}x", transform=ax.transAxes,
                va='top', fontsize=9, bbox={'facecolor': 'white', 'edgecolor': '#dddddd', 'alpha': .95})
    axes[0].set_ylabel('Dispatch + Combine latency (ms; log scale; lower is better)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(.51, .12), frameon=False)
    fig.suptitle('A100 pure communication: node locality depends on the actual transport policy', fontsize=14, y=.97)
    fig.text(.025, .025, 'Same tokens per call; BF16 hidden 5120, top-8 / 160 experts; no Attention or Expert compute; eager (no Graph).\n'
             'Same native binary within this GLM cohort. Median of 3 aligned rank-max samples. EP4 has 2x assignments/GPU versus EP8.\n'
             'Pair-internal traffic stays NVLink. Every cross-node transfer remains RDMA; CE only replaces local cross-pair traffic.', fontsize=8.5)
    fig.subplots_adjust(left=.075, right=.985, top=.85, bottom=.29, wspace=.12)
    save(fig, 'communication_transport_curves')

    main_metrics = [r for r in full if r['case'] != 'qwen35_2gpu']
    table = md_table(['部署／模型', '阶段', '配置', '吞吐 token/s ↑', '平均 TTFT 秒', '平均 TPOT 毫秒', '平均端到端 秒'],
        [[('两卡 Qwen122B' if r['gpus'] == 2 else '八卡 GLM INT4'), r['phase'], r['layout'],
          f"{r['throughput']:.2f}", f"{float(r['ttft_s_mean']):.4f}",
          f"{float(r['tpot_ms_mean']):.4f}" if r['tpot_ms_mean'] else '不适用',
          f"{float(r['e2e_s_mean']):.4f}"] for r in main_metrics])
    example_rows = [r for r in selected if r['ep4_node'] == '065' and r['global_batch_tokens'] in ['8', '128', '1024', '4096']]
    comm_table = md_table(['单次输入 token', '节点内 EP4 RDMA', '跨节点 EP8 RDMA', '节点内 EP4 CE', '跨节点 EP8 CE'],
        [[r['global_batch_tokens']] + [f"{float(r[k]):.4f}" for k in
          ['ep4_rdma_ms', 'ep8_rdma_ms', 'ep4_ce_ms', 'ep8_ce_ms']] for r in example_rows])
    report = f'''# A100 两卡／八卡：通信域与 EP、PP 的部署取舍

日期：2026-10-26。**只读取已经保存的历史数据、重算及作图，没有申请 GPU 或运行新性能实验。**

## 1. 术语与准确负载

| 项目 | 本报告的确切含义 |
|---|---|
| EP2、EP8 | 同层专家分布到两张／八张卡；不是两个／八个独立副本。两卡 Attention 为 TP1×DP2；八卡为节点内 TP4×DP2。 |
| PP2、PP2×EP4 | 两个模型层阶段。两卡 PP 每阶段本地 EP1；八卡 PP 每节点一个阶段、节点内 EP4、Attention TP4。 |
| 两卡 Prefill | 8 请求；每条没有已缓存上下文，新增输入 4096 token，生成 1 token；整批 32768 新输入。 |
| 两卡 Decode | 8 请求；每条 GPU 已有真实模型生成的 4096-token 缓存，新增输入 1 token，生成 128 token；不计前缀构建、没有 CPU HiCache 换入。 |
| 八卡 Prefill | 4 请求；每条无旧缓存、新输入 8192 token、生成 1 token；整批同为 32768 新输入，但请求分布不同。 |
| 八卡 Decode | 4 请求；每条从 128 个新输入 token 开始，生成 512 token；冷输入，不是已有长缓存的纯 Decode 起点。 |
| TTFT／TPOT／端到端 | 请求放行至首输出／首输出至末输出除以输出数量减一／放行至末输出。Prefill 仅输出一 token，TPOT 不适用。 |
| 吞吐 | Prefill 计输入 token/s，Decode 计输出 token/s，越大越好；不能由平均请求延迟的倒数推得整批吞吐。 |
| 纯通信 token | 单次发给专家的唯一输入 token，不是旧缓存长度，也不是一定等于请求数。D+C 为一次完整 Dispatch＋Combine 事件跨度，不是请求 TPOT。 |

## 2. 完整推理：有类似实验，但 A100 不是 H100 那种反转

两卡主组是 Qwen3.5-122B-A10B FP8，A100069 GPU0/1、NV12；W8A16 Marlin、BF16 激活/KV、Triton Attention；DeepEP normal NVLink；Graph/overlap 关闭。固定两槽 PP，同一对卡顺序运行，各三轮。
八卡为 GLM-4.7-AWQ INT4/Humming，168/169 各四卡；同旧 native 9dc34887，pair 内 NVLink、pair 外及跨节点 RDMA。Prefill eager、Decode Graph；每点一个正式四请求 cohort。
八卡已统一节点内 Prefill chunk1024，EP 全局2048／PP全局1024；不能称为相同全局 step 或相同 GEMM 形状。

{table}

![完整推理归一化比较](e2e_communication_domain.png)

- 两卡 122B：两槽 PP Decode 吞吐比 EP 高 **34.92%**；Prefill 则低 **5.60%**。
- 两卡 35B BF16 的补充对照：PP Prefill **31841.16** 对 EP **27644.14**（+15.18%）；PP Decode **121.76** 对 EP **90.80**（+34.09%）。与 122B 是不同精度和内核，不是模型大小单变量消融。
- 八卡 GLM：PP Prefill **1943.53** 对 EP **1322.77**（+46.93%）；Decode **101.92** 对 **100.06**（+1.85%）。四请求、单 cohort 的小 Decode 差距不足以证明稳定领先。
- 因而，**现有 A100 完整推理不能证明“跨节点后 PP 相对收益必然变大”，也没有 H100 那种 EP2 Decode 领先、跨节点 EP8 落后的明确翻转。**

### 为什么不能换一个旧 PP 数字来制造反转？

同一 Qwen122B 旧四槽 PP Decode 只有49.04 output token/s，低于 EP 的72.08；但两槽 PP 是97.26。槽位调度足以改变结论，所以本主图固定两槽而不挑选四槽。
后续 Q02 发送就绪优化的 Prefill 同进程从8160.97提升到9391.74 input token/s；那轮没有重测 EP，Decode 输出等价未通过，只保留 Prefill 局部优化结论，未混入本图。
Qwen122B 原 EP r7 测量、缓存和清理完成，但原 Guard 包装器误捕获 SystemExit(0)，状态保持失败；指标已独立复算。后续两槽 PP r8 通过。35B 的配对完整验证通过。上述操作状态与性能证据分别保留，不补写成功记录。

## 3. 同库纯通信：节点内到底走了什么，比标签更关键

最新 T01 扫描：065/167 同型号 ConnectX-6 Dx，GLM 通信向量宽5120、BF16、160专家、每 token 命中8专家；确定性均衡路由，没有模型、Attention、Expert GEMM 或 KV。
每点三次预热、三次正式采样；先取同次迭代 rank 最大跨度，再取中位数。GLM RDMA／CE 共享 c3405cad 原生库；实际同调用源行数、路由、容量、Graph 模式均匹配。

这里用**两种 EP 宽度单次调用输入 token 相同**的对照。EP4 每卡的 token–专家配对数为 EP8 的两倍，并非相同每卡工作量；完整 PP 半批对照保留在原 T01，不假造 PP 吞吐。

单位毫秒，越小越好。下表节点内为065，跨节点为065/167；另一节点独立结果完整保留在 CSV 和右图。

{comm_table}

![同库通信策略与输入量](communication_transport_curves.png)

- 8-token 小消息，RDMA 策略节点内0.1852／跨节点0.2846 ms，跨节点约1.54倍；该小点 CE 更慢。
- 1024-token、CE 策略节点内1.3348／跨节点7.0528 ms，跨节点约5.28倍；但 RDMA 策略为节点内8.4070／跨节点6.3670 ms，方向相反。
- CE 只替换同节点 NVLink pair 之外的 PCIe 搬运，跨节点仍走 RDMA；不能把 EP8 CE 称为跨节点 PCIe。EP4 的全本地 drain 优化与 EP8 远端完成路径也不同。
- 节点内、两节点的资源聚合、source 分配、QP／专家所有权和同步协议不同；不能把整段时差只叫作物理线缆延迟，事件跨度还包括 CPU 提交、打包、等待和还原。
- 旧9dc库同全局1024-token基线也出现过节点内8.7050～8.8947／跨节点EP8 6.5529 ms；该旧基线同节点 pair 外同样走RDMA。不能把它描述为全NVLink对跨节点。

## 4. 这份证据支持到哪里

**可以支持：EP／PP 的选择依赖实际通信域、传输实现、消息量和本地 Expert 执行成本。** H100 的节点内全NVLink与A100的“两组NVLink pair＋PCIe”不是同一种本地拓扑。CE让A100大消息节点内交换明显受益，但小消息、跨节点与完整调度都需另看。

**不支持：** 跨节点一定让PP获益更多；节点内一定比跨节点快；用不同模型的端到端差值唯一归因于网络；把组件中位数直接相加预测TPOT。完整推理之间还改变了模型、精度、内核、Graph、请求数、缓存起点和输出长度。

## 5. 文件与可复查入口

- [全部12行请求指标](full_metrics.csv)：包括补充35B，均值／中位数／P99；两卡每点24请求、八卡每点4请求，P99只描述样本。
- [6组吞吐对照](e2e_comparisons.csv) · [GLM两节点20行同调用通信表](communication_equal_call.csv) · [旧RDMA基线](old_rdma_communication.csv)。
- [原始路径、逐文件SHA256](source_manifest.json) · [独立离线校验](validation.json)。源表及短报告小副本在sources/，没有权重、KV或原始大trace。
- [T01全部模型／两种批次边界]({COMM / 'summary.md'})；其通信与独立真实Expert联表仍是组件估算，不是完整PP观测。
- [H100对应报告]({ROOT / 'experiments/h100_ep_pp_comm_domain_comparison_20261026/REPORT.md'})。

复算与画图：`MPLCONFIGDIR=/tmp/a100-comm-domain-mpl PYTHONDONTWRITEBYTECODE=1 /home/bxb1/conda/bin/python {HERE / 'build_report.py'}`。
'''
    (HERE / 'REPORT.md').write_text(report)
    result = dict(passed=True, no_new_gpu_samples=True, full_metric_rows=len(full),
        comparison_rows=len(comparisons), current_comm_rows=len(selected),
        independently_recomputed_comm_points=len(phases), raw_source_hashes_verified=len(checked_raw),
        within_call_transport_identity_groups=len(groups), e2e_exact_workload_match=False,
        h100_like_decode_reversal_observed=False)
    (HERE / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    (HERE / 'manifest.json').write_text(json.dumps(dict(id=HERE.name, created_at='2026-10-26',
        title='A100 historical two/eight-GPU EP/PP and communication-domain comparison',
        status='complete_offline_analysis', no_gpu_launch=True,
        sources=provenance, validation=result,
        artifacts=['REPORT.md', 'full_metrics.csv', 'e2e_comparisons.csv',
                   'communication_equal_call.csv', 'e2e_communication_domain.png',
                   'communication_transport_curves.png'],
        caveats=['Different models, kernels, cache state, batch and Graph modes.',
                 'No H100-like Decode preference reversal in selected A100 cohorts.',
                 'CE/RDMA and token count alter local-versus-cross-node direction.',
                 'Original Qwen122 EP7 Guard failure retained; raw metrics independently validated.']),
        indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
