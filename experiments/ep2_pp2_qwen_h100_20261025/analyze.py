"""Recompute request metrics before publishing a small reproducible report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys


def analyze(run):
    sys.path.insert(0, '/home/bxb1/.codex/skills/llm-experiment-guard/scripts')
    from artifact_visibility import wait_for_stable_json
    cfg = json.loads((run / 'config.json').read_text())
    complete, visibility = wait_for_stable_json(run / 'completed.json', timeout_s=120)
    if complete.get('passed') is not True:
        raise AssertionError('No complete paired comparison')
    cleanup, cleanup_visibility = wait_for_stable_json(run / 'worker_cleanup.json', timeout_s=120)
    if cleanup.get('cleanup_verified') is not True:
        raise AssertionError('Cleanup has not passed')
    postchecks = [(p, json.loads(p.read_text()))
                  for p in (run / 'dashboard_guard/checks').glob('*-post.json')]
    passed_postchecks = [(p, d) for p, d in postchecks
                         if d.get('passed') is True and d.get('scientific_success') is True
                         and d.get('cleanup_verified') is True]
    assert passed_postchecks, 'Final Guard scientific/cache/cleanup check has not passed'
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    rows, provenance, checks = [], [], []
    for arm in ('ep2', 'pp2'):
        for path in sorted((run / arm).glob('metrics.*.json')):
            data, receipt = wait_for_stable_json(path, timeout_s=120)
            checks.append(receipt)
            assert data['valid'] and not data['profiler_enabled']
            assert len(data['rounds']) == cfg['repeats']
            phase = path.stem.split('.')[-1]
            case = data['case']
            flat = []
            rates = []
            for group in data['rounds']:
                reqs = group['requests']
                assert len(reqs) == case['requests']
                elapsed = max(r['finished'] for r in reqs) - min(r['started'] for r in reqs)
                assert abs(elapsed - group['makespan_s']) < 1e-7
                count = 1 if phase == 'prefill' else case['output_tokens']
                assert all(r['output_tokens'] == count for r in reqs)
                context = case['prefill_cached_context_tokens'] if phase == 'prefill' else case['cached_context_tokens']
                fresh = case['new_prefill_tokens'] if phase == 'prefill' else 1
                assert all(r['prompt_tokens'] == context + fresh for r in reqs)
                if context:
                    assert all(r['cached_tokens'] >= context - 64 for r in reqs)
                else:
                    assert all((r['cached_tokens'] or 0) == 0 for r in reqs)
                rates.append(case['requests'] * (fresh if phase == 'prefill' else count) / elapsed)
                flat.extend(reqs)
            row = dict(arm=arm, phase=phase, case=case['id'], requests=case['requests'],
                cached_context_per_request=case['prefill_cached_context_tokens'] if phase == 'prefill' else case['cached_context_tokens'],
                fresh_input_per_request=case['new_prefill_tokens'] if phase == 'prefill' else 1,
                output_per_request=1 if phase == 'prefill' else case['output_tokens'],
                tokens_per_s=statistics.mean(rates), tokens_per_s_min=min(rates),
                tokens_per_s_max=max(rates), tokens_per_s_std=statistics.stdev(rates),
                rounds=len(data['rounds']),
                ready_send=data['selected_ready_send'], gpu_pair='0,1' if arm == 'ep2' else '2,3')
            for metric in ('ttft_s', 'tpot_ms', 'e2e_s'):
                values = sorted(r[metric] for r in flat if r[metric] is not None)
                if values:
                    expected = dict(mean=statistics.mean(values), median=statistics.median(values),
                        p99=values[min(len(values)-1, int(.99 * len(values)))])
                    for stat, value in expected.items():
                        assert abs(data[metric][stat] - value) < 1e-7
                        row[metric + '_' + stat] = value
                else:
                    for stat in ('mean', 'median', 'p99'):
                        row[metric + '_' + stat] = None
            measured = data['fresh_input_tokens_per_s'] if phase == 'prefill' else data['output_tokens_per_s']
            assert abs(row['tokens_per_s'] - measured) < 1e-6
            rows.append(row)
            provenance.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(rows) == 12 + 2 * len(complete.get('extra_cases', [])), len(rows)
    with (out / 'full_metrics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    qualifications = {}
    tuning = {}
    for arm in ('ep2', 'pp2'):
        qualifications[arm] = [json.loads(p.read_text())
                               for p in (run / arm / 'qualification').glob('*.json')]
        assert len(qualifications[arm]) == 2 and all(q['passed'] for q in qualifications[arm])
        log = (run / 'cache' / (arm + '.log')).read_text(errors='replace')
        assert any('Decode batch' in line and 'cuda graph: True' in line
                   for line in log.splitlines()), 'No real Decode Graph replay in ' + arm
        tuning[arm] = json.loads((run / arm / 'tuning.json').read_text())
        assert tuning[arm]['output_agreement'] is True
    evidence = dict(passed=True, rows=rows, sources=provenance,
        guard_postcheck=str(passed_postchecks[-1][0]),
        logical_capacities=complete['logical_capacities'], qualifications=qualifications,
        tuning=tuning, actual_decode_graph_replay_both_arms=True,
        visibility=[visibility, cleanup_visibility] + checks,
        percentiles='Descriptive request samples, not stable service-tail estimates')
    (out / 'validated_metrics.json').write_text(json.dumps(evidence, indent=2) + '\n')
    labels = dict(moderate='中等负载', near_capacity_b8='接近 EP2 容量，8 请求',
                  near_capacity_b16='接近 EP2 容量，16 请求',
                  pp_own_capacity_b16='接近 PP2 自身容量，16 请求')
    capacities = complete['logical_capacities']
    lines = ['# H100：EP2 与 PP2 的完整双卡性能', '',
        '模型、配置、缓存与计时口径见 [实验说明](../../../README.md)。两组各用两张 H100，'
        '中等负载与 A100 对齐请求数、长度和缓存状态，但不是历史文本逐 token 重放；长缓存点另列。', '',
        f"EP2 两卡逻辑 KV 容量为 **{capacities['ep2']:,} token**，PP2 为 **{capacities['pp2']:,} token**。"
        'PP2 两阶段分别存部分层，容量不相加。共同长缓存负载接近 EP2 上限；另测 PP2 自身容量边界，'
        '这一额外点的 context 更长，不能与 EP2 当作同工作量排名。', '',
        '延迟列为 **均值 / 中位数 / P99**；P99 仅描述本次少量请求样本。'
        '吞吐只计新输入或实际输出 token；所有上下文准备和 profile 均排除在正式计时外。', '',
        '| 负载 | 阶段 | 配置 | 请求数 | 每条已缓存 token | 每条新输入 / 输出 | 吞吐 token/s | TTFT 秒 | TPOT 毫秒 | 端到端 秒 |',
        '|---|---|---|---:|---:|---:|---:|---|---|---|']
    def triple(row, key):
        return '—' if row[key + '_mean'] is None else ' / '.join(f'{row[key + "_" + stat]:.3f}' for stat in ('mean','median','p99'))
    for row in sorted(rows, key=lambda r:(r['case'], r['phase'], r['arm'])):
        lines.append(f"| {labels[row['case']]} | {row['phase']} | {row['arm'].upper()} | {row['requests']} | "
            f"{row['cached_context_per_request']:,} | {row['fresh_input_per_request']} / {row['output_per_request']} | "
            f"{row['tokens_per_s']:.2f} | {triple(row, 'ttft_s')} | {triple(row, 'tpot_ms')} | {triple(row, 'e2e_s')} |")
    lines += ['', '## 同一负载的吞吐差异', '', '| 负载 | 阶段 | PP2 相对 EP2 |', '|---|---|---:|']
    for case in complete['cases']:
        for phase in ('prefill', 'decode'):
            pair = {r['arm']:r for r in rows if r['case']==case['id'] and r['phase']==phase}
            delta = (pair['pp2']['tokens_per_s'] / pair['ep2']['tokens_per_s'] - 1)*100
            lines.append(f"| {labels[case['id']]} | {phase} | {delta:+.2f}% |")
    lines += ['', '## 实际启用的优化与性能资格', '',
        '- 两组均核验 H100 原生 FP8 W8A8 DeepGEMM；实际 Decode 日志均出现 CUDA Graph 重放。'
        '全 Attention 使用 FA3，线性 Attention 使用 Triton；没有回退到 A100 的 W8A16 路径。',
        '- EP2 每卡 Attention TP1，各处理一半请求，专家跨两卡分配；PP2 各阶段 24 层、每层全部专家，'
        '始终两个等大微批次槽位。EP2 使用原生 overlap；引擎因兼容性自动对 PP2 关闭 overlap。',
        '- 通信缓冲区在 KV 池分配前初始化；Decode 按实际本卡 batch 配置低延迟缓冲区，'
        'Graph 和线性状态缓存大小按 DP 后的局部请求数核验，没有把双卡容量相加两次。']
    pp_tune = tuning['pp2']
    tune_gain = (pp_tune['baseline_s'] / pp_tune['candidate_s'] - 1) * 100
    lines += [f"- PP Prefill 发送就绪策略短 A/B/B/A：原版中位批耗时 {pp_tune['baseline_s']:.4f} 秒，"
        f"候选 {pp_tune['candidate_s']:.4f} 秒，派生吞吐变化 {tune_gain:+.2f}%；"
        f"本次{'采用' if pp_tune['prefill_ready_send_selected'] else '未采用'}候选。对应首 token 输出检查通过；"
        'Decode 始终保留原生产策略，不把未验收的 Decode 候选算作优化结果。',
        '- 每轮缓存 Prefill 前都独立清空并预热相同基础前缀，准备时间不计时。'
        '正式请求逐条核验命中，避免上一轮新分支抢占线性 Attention 状态而引入意外重算。',
        '- 最终 EP2 线性状态池为全局 88、DP 后每卡 44；PP2 为每阶段 84。'
        '两组均启用原生 `--mamba-max-states-per-path 1`，清理不再需要的中间 chunk 状态，'
        '保留末端、分叉和锁定状态；没有更改模型计算或状态精度。实际 KV 容量已扣除状态池开销。'
        '仅准备阶段按每组 4 请求建缓存，正式 8／16 请求仍同时提交。',
        '- 所有正式指标与 profiler 分开采样；独立 trace 保存于两组 `traces/` 子目录，'
        '包含首 token 的 Extend 与后续 Decode，不能把整段 trace 的 kernel 总和称为每输出 token 耗时。']
    lines += ['', '## 核验与限制', '',
        '- 完整层数、FP8 权重、实际 allocator 容量、输出长度与旧前缀命中记录在对应运行目录。',
        '- 两组配置共用同一节点但使用不同的 GPU 对，并发运行；共享 CPU 与节点资源，不能解释为无干扰的独占整机峰值。',
        '- PP 发送优化的候选选择见各自 `tuning.json`；短局部对照不能证明全局最优。',
        '- H100 原生 W8A8 与此前 A100 W8A16 的计算内核不同，跨代差异不能仅归因于硬件。',
        '- 此前 A100 对照关闭 Graph／overlap、Decode 使用 normal 通信；本轮 H100 实际启用 Graph，'
        'EP2 Decode 使用 low-latency 内核且开启 overlap。跨代趋势是完整执行路径的比较，不是硬件单变量实验。',
        '- 本轮没有 CPU HiCache 恢复或 MTP；不能与包含这些工作的历史 TTFT 混为同一口径。', '',
        '## 保留的失败尝试', '',
        'r1 为两服务的相邻 rendezvous 端口冲突；r2 为实验对大 batch 的 context 下限判断过严，'
        '同时暴露了 DP 后状态池与 Graph 的预算问题；r3／r4 在 16 请求 Prefill 中检出基础前缀部分丢失；'
        'r5 的早期资格检查遇到首次并发导入 tokenizer 的竞争，最终改为启动 GPU 前串行解析。'
        'r6 完成三个共同负载，但 PP 自身容量点仍因沿途中间状态挤占而失去完整命中；'
        '最终运行启用原生路径状态剪枝，并在正式扫点前验证最大的 PP 自身容量负载。'
        '这些运行及其失败检查均原样保留，不计入本报告的正式结果。最终结果仅来自本报告链接的完整通过运行。', '',
        '数据：[完整 CSV](full_metrics.csv) · [逐请求重算与哈希](validated_metrics.json) · '
        '[原始运行配置](../config.json) · [实测容量](../capacity_matrix.json)。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(passed=True, rows=len(rows), report=str(out / 'REPORT.md'))))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    analyze(ap.parse_args().run)
