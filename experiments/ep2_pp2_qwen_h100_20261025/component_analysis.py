"""CPU-only Decode component follow-up; retain narrow kernel attribution."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics


FIELDS = ('full_attention_core_ms', 'linear_attention_core_ms', 'routed_gemm_ms',
          'dispatch_ms', 'combine_ms', 'other_deepep_ms', 'nccl_in_graph_ms',
          'unclassified_kernel_ms')


def classify(name):
    # FA3's own split-KV reduction is Attention, not MoE Combine.
    if 'flash::FlashAttnFwd' in name or 'flash::prepare_varlen_num_blocks_kernel' in name:
        return 'full_attention_core_ms'
    if 'fused_recurrent_gated_delta_rule' in name:
        return 'linear_attention_core_ms'
    if 'deep_gemm::' in name and '(deep_gemm::GemmType)2' in name:
        return 'routed_gemm_ms'
    if 'deep_ep::' in name:
        if 'dispatch' in name:
            return 'dispatch_ms'
        if 'combine' in name:
            return 'combine_ms'
        return 'other_deepep_ms'
    if name.startswith('ncclDevKernel'):
        return 'nccl_in_graph_ms'
    return 'unclassified_kernel_ms'


def analyze(run):
    out = run / 'analysis/component_followup'
    out.mkdir(exist_ok=True)
    source = run / 'analysis/decode_graph_inventory.json'
    inventory = json.loads(source.read_text())['rows']
    assert len(inventory) == 14
    old = list(csv.DictReader((run / 'analysis/decode_graph_summary.csv').open()))
    reference = {r['rank_trace']: r for r in old}
    rows, proof = [], []
    for item in inventory:
        path = Path(item['trace'])
        if not path.is_absolute():
            path = Path.cwd() / path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path
        samples = len(item['samples'])
        assert samples > 0
        arm = path.parents[2].name
        assert arm in ('ep2', 'pp2')
        stage = None if arm == 'ep2' else int(path.name.split('-PP-')[1].split('.')[0])
        count = item['graph_kernel_counts']
        layers = 48 if arm == 'ep2' else 24
        full_main = sum(v for n, v in count.items() if 'flash::FlashAttnFwdSm90<' in n)
        linear_main = sum(v for n, v in count.items() if 'fused_recurrent_gated_delta_rule' in n)
        assert full_main == layers // 4 * samples, (path, full_main)
        assert linear_main == layers * 3 // 4 * samples, (path, linear_main)
        costs = Counter({key: 0. for key in FIELDS})
        for name, duration in item['graph_kernel_duration_ms'].items():
            costs[classify(name)] += duration / samples
        total = sum(item['graph_kernel_duration_ms'].values()) / samples
        assert abs(sum(costs.values()) - total) < 1e-8
        prior = reference[path.name]
        assert abs(costs['routed_gemm_ms'] - float(prior['routed_gemm_kernel_sum_ms'])) < 1e-8
        assert abs(sum(costs[k] for k in ('dispatch_ms', 'combine_ms', 'other_deepep_ms'))
                   - float(prior['deepep_kernel_sum_ms'])) < 1e-8
        row = dict(case=path.parent.name, arm=arm, stage='' if stage is None else stage,
                   rank_trace=path.name, layers=layers, graph_replays=samples,
                   graph_span_ms=statistics.mean(s['graph_span_ms'] for s in item['samples']),
                   **dict(costs), kernel_sum_ms=total)
        row['attention_core_ms'] = costs['full_attention_core_ms'] + costs['linear_attention_core_ms']
        rows.append(row)
        proof.append(dict(trace=str(path), sha256=item['sha256'], graph_replays=samples,
                          full_attention_main_calls=full_main, linear_attention_main_calls=linear_main))
    with (out / 'per_rank.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    labels = {'moderate': '8 请求，每条 GPU 已缓存 4096 token',
              'near_capacity_b8': '8 请求，每条 GPU 已缓存 26752 token',
              'near_capacity_b16': '16 请求，每条 GPU 已缓存 13376 token',
              'pp_own_capacity_b16': '仅 PP：16 请求，每条 GPU 已缓存 42176 token'}
    groups = []
    numeric = ['graph_span_ms', *FIELDS, 'kernel_sum_ms', 'attention_core_ms']
    for case in labels:
        for arm, stage in [('ep2', ''), ('pp2', 0), ('pp2', 1)]:
            matches = [r for r in rows if (r['case'], r['arm'], r['stage']) == (case, arm, stage)]
            if not matches:
                continue
            assert len(matches) == (2 if arm == 'ep2' else 1)
            row = dict(case=case, arm=arm, stage=stage, layers=matches[0]['layers'], ranks=len(matches))
            for key in numeric:
                row[key] = statistics.mean(r[key] for r in matches)
            groups.append(row)
    with (out / 'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(groups[0]))
        writer.writeheader(); writer.writerows(groups)
    text = ['# H100 Decode：Attention、Expert 与通信的绝对耗时', '',
        '模型为完整 Qwen3.5-122B-A10B-FP8；数据来自已通过的 H100 r7 独立诊断 trace，'
        '**没有新跑 GPU 实验**。每个实际 rank 的 Decode CUDA Graph 次数见逐卡 CSV，按原分析的 '
        '`cudaGraphLaunch` correlation 对齐，排除首 token 的 Extend。', '',
        '## 单位及能拆到的范围', '',
        '- 单位为 **毫秒／每卡／一次实际 Decode Graph**。EP2 展示两卡均值，'
        '每卡执行完整 48 层；PP2 分阶段展示，每卡执行 24 层。'
        '8 请求时每卡／每槽为 4 请求，16 请求时为 8 请求。'
        f"各 rank 只有 {min(r['graph_replays'] for r in rows)}–{max(r['graph_replays'] for r in rows)} 次 Graph，"
        '是短诊断，不是正式三轮请求指标的分项均值。',
        '- **Attention 核心**只包括 FA3 主内核、FA3 自身归并及准备、Gated Delta Rule 递归内核。'
        '不含输入输出投影、卷积、归一化、KV 写入等，不能当成完整 Attention 模块耗时。',
        '- **路由专家 GEMM**只包括 DeepGEMM grouped GEMM；不含路由、量化、排列、激活、共享专家。'
        '通用投影和共享专家未可靠拆开，统一保留在未归属内核中，不猜分。',
        '- Dispatch／Combine 是 DeepEP 内核；NCCL 单列。FA3 的 Combine 属于 Attention，'
        '没有错误计入 MoE Combine。内核名 `internode_ll` 不代表本轮跨节点：本轮是同节点 NV6 GPU 对。',
        '- 分项为 **kernel 累计运行时长**，不同流可重叠，**不能相加作 TPOT，不能除以 Graph 跨度当独占占比**。'
        'Graph 跨度是第一至最后 GPU kernel 的墙钟区间，另列展示。', '',
        '## Decode 分项', '',
        '| 请求组成 | 配置 | Attention 核心 | 路由专家 GEMM | DeepEP Dispatch | DeepEP Combine | Graph 内 NCCL | Graph GPU 跨度 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in groups:
        title = 'EP2，两卡均值' if r['arm'] == 'ep2' else f"PP2，第 {int(r['stage'])+1} 阶段"
        text.append('| ' + labels[r['case']] + ' | ' + title + ' | ' +
                    ' | '.join(f'{r[k]:.3f}' for k in ('attention_core_ms', 'routed_gemm_ms',
                    'dispatch_ms', 'combine_ms', 'nccl_in_graph_ms', 'graph_span_ms')) + ' |')
    text += ['', '## 未归属的部分，不是空闲', '',
        '| 请求组成 | 配置 | 未归属 kernel 累计毫秒 | 全部 kernel 累计毫秒 |',
        '|---|---|---:|---:|']
    for r in groups:
        title = 'EP2，两卡均值' if r['arm'] == 'ep2' else f"PP2，第 {int(r['stage'])+1} 阶段"
        text.append(f"| {labels[r['case']]} | {title} | {r['unclassified_kernel_ms']:.3f} | {r['kernel_sum_ms']:.3f} |")
    text += ['', '## PP 边界传输与 Prefill 的缺口', '',
        'PP 各阶段 Expert EP1，因此模型 Graph 内没有跨卡专家 Dispatch／Combine。'
        '**表中 0 仅表示没有这类 Graph 内内核，不表示 PP 总通信为零。** '
        '阶段激活、元数据和反馈传输在 Graph 外；trace 有 NCCL send/recv，但驻留包含等上游数据的时间，'
        '当前没有纯链路传输与依赖等待的完整分离，不能把整段驻留冒充纯通信延迟。', '',
        '正式 Prefill 测量没有独立同口径分项 trace。该诊断 trace 的初始 Extend 只有每条新增 1 token，'
        '**不能替代无旧缓存新增 4096 或长缓存新增 512 的正式 Prefill**。'
        '因此本报告不伪造这两种 Prefill 的 Attention／Expert／通信数字。', '',
        '## 可以得到的观察', '',
        '中等负载下，EP2 完整 48 层的路由 GEMM 两卡均值约 2.37 毫秒，'
        'PP 每阶段只有 24 层却仍需约 1.84–2.09 毫秒；PP 集中更多专家后的每层 GEMM 代价较高。'
        'EP2 DeepEP Dispatch＋Combine 的 kernel 累计均值约 1.91 毫秒，PP 模型层内避免了这部分。'
        '这些观测支持通信与专家计算的权衡，**尚不能精确归因正式 TPOT 的全部差额**。', '',
        '[逐卡 CSV](per_rank.csv) · [对照 CSV](comparison.csv) · [来源与分类验证](validation.json) · '
        '[原 Decode Graph 检查](../GRAPH_PROFILE.md) · [正式请求级结果](../REPORT.md)。', '']
    (out / 'REPORT.md').write_text('\n'.join(text))
    validation = dict(passed=True, gpu_run=False, source_inventory=str(source),
        source_inventory_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        rank_rows=len(rows), comparison_rows=len(groups), traces=proof,
        category_partition_kernel_sum_verified=True, prior_gemm_deepep_parity_verified=True,
        expected_full_linear_layer_call_counts_verified=True,
        unit='ms kernel cumulative duration per actual rank Decode Graph; not exclusive shares',
        classification={key:[] for key in FIELDS})
    for item in inventory:
        for name in item['graph_kernel_duration_ms']:
            bucket = validation['classification'][classify(name)]
            if name not in bucket:
                bucket.append(name)
    (out / 'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(dict(passed=True,rank_rows=len(rows),comparison_rows=len(groups),report=str(out/'REPORT.md'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    analyze(parser.parse_args().run.resolve())
