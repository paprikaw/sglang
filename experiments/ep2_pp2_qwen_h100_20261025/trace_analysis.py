"""Correlate actual Decode Graph GPU kernels; do not add stage means into TPOT."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def summarize(run):
    rows, inventory = [], []
    for arm in ('ep2', 'pp2'):
        for path in sorted((run / arm / 'traces').glob('*/*.trace.json.gz')):
            with gzip.open(path, 'rt') as handle:
                trace = json.load(handle)
            events = trace['traceEvents']
            kernels = [e for e in events if e.get('cat') == 'kernel']
            launches = [e for e in events if e.get('name') == 'cudaGraphLaunch']
            samples = []
            for launch in launches:
                correlation = launch['args']['correlation']
                selected = [e for e in kernels if e.get('args', {}).get('correlation') == correlation]
                if not selected:
                    continue
                # The wrapper disables Prefill Graph; matched Graph kernels
                # belong to actual Decode replays, not the initial Extend.
                assert all(e.get('args', {}).get('graph id') for e in selected)
                def total(predicate):
                    return sum(e['dur'] for e in selected if predicate(e['name'])) / 1000
                samples.append(dict(
                    graph_span_ms=(max(e['ts']+e['dur'] for e in selected)-min(e['ts'] for e in selected))/1000,
                    kernel_sum_ms=total(lambda n:True),
                    deepep_kernel_sum_ms=total(lambda n:'deep_ep::' in n),
                    routed_gemm_kernel_sum_ms=total(lambda n:'deep_gemm::' in n and '(deep_gemm::GemmType)2' in n)))
            if not samples:
                raise AssertionError('No correlated actual Decode Graph GPU kernels: ' + str(path))
            row = dict(arm=arm, case=path.parent.name, rank_trace=path.name,
                       graph_replays=len(samples), unit='ms per actual rank Graph replay')
            for key in samples[0]:
                row[key] = statistics.mean(s[key] for s in samples)
            rows.append(row)
            counts, durations = Counter(), Counter()
            for launch in launches:
                for e in kernels:
                    if e.get('args', {}).get('correlation') == launch['args']['correlation']:
                        counts[e['name']] += 1
                        durations[e['name']] += e['dur'] / 1000
            inventory.append(dict(trace=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  graph_kernel_counts=dict(counts),
                                  graph_kernel_duration_ms=dict(durations), samples=samples))
    assert len(rows) == 14, len(rows)
    out = run / 'analysis'
    with (out / 'decode_graph_summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out / 'decode_graph_inventory.json').write_text(json.dumps(dict(rows=inventory,
        limitations=['Profile is separate from formal metrics; profiler overhead is included here',
            'Kernel sums can overlap; not exclusive attention/expert/communication shares',
            'Routed GEMM classification excludes activation, permutation, shared expert and projections',
            'Pipeline boundary transfer and feedback occur outside model Graph',
            'Per-stage Graph times cannot be summed and called request TPOT']), indent=2) + '\n')
    text = ['## Decode Graph 资格检查', '',
        '通过 CUDA Graph launch 的 correlation ID 对齐实际 GPU kernel，排除首 token 的 Extend。'
        '以下是独立 profiler 样本，**不是正式 TPOT，也不是可相加的通信／计算独占占比**。'
        'Routed GEMM 只统计 DeepGEMM grouped expert GEMM，未包含激活、重排、共享专家；'
        '流水边界传输和反馈发生在模型 Graph 外。', '',
        '| 负载 | 配置 | rank trace | Graph 次数 | Graph GPU 跨度，毫秒 | DeepEP kernel 合计，毫秒 | Routed GEMM kernel 合计，毫秒 |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['case']} | {r['arm'].upper()} | {r['rank_trace']} | {r['graph_replays']} | "
                    f"{r['graph_span_ms']:.3f} | {r['deepep_kernel_sum_ms']:.3f} | {r['routed_gemm_kernel_sum_ms']:.3f} |")
    text += ['', '[Graph 对照 CSV](decode_graph_summary.csv) · [逐 kernel 数据与分类边界](decode_graph_inventory.json)。', '']
    (out / 'GRAPH_PROFILE.md').write_text('\n'.join(text))
    print(json.dumps(dict(profile_rows=len(rows), output=str(out / 'GRAPH_PROFILE.md'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    summarize(parser.parse_args().run)
