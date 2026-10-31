"""Partition observed root-session time using saved request-ready intervals."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics as st
from analyze_balanced_matrix import ROOT, RUNS


def covered_seconds(spans):
    total = 0.0
    left = right = None
    for start, end in sorted(spans):
        assert end >= start
        if left is None:
            left, right = start, end
        elif start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + (right - left if left is not None else 0)


def main(out=None):
    rows, sources = [], []
    for name in RUNS:
        run = ROOT/'runs'/name
        for p in sorted(run.glob('results/*/*/*/formal.r*.json')):
            repeat = p.stem.split('.')[-1]
            if not json.loads(p.with_name('completed.'+repeat+'.json').read_text())['passed']:
                continue
            raw = p.read_bytes()
            d = json.loads(raw)
            arm, case, policy = p.relative_to(run).parts[1:4]
            roots = defaultdict(list)
            for r in d['requests']:
                ready = r['started'] - r['client_dispatch_lag_ms']/1000
                roots[r['root_id']].append((ready, r['finished']))
            for root_id, spans in roots.items():
                duration = max(b for a, b in spans) - min(a for a, b in spans)
                covered = covered_seconds(spans)
                assert -1e-6 <= duration-covered <= duration+1e-6
                rows.append(dict(run=name, arm=arm, case=case, policy=policy, repeat=repeat,
                    root_id=root_id, calls=len(spans), ready_to_last_llm_s=duration,
                    with_pending_ready_request_s=covered,
                    without_pending_ready_request_s=max(0, duration-covered)))
            sources.append(dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest()))
    out = Path(out) if out is not None else ROOT/'matrix_analysis'
    if rows:
        with (out/'root_exposure.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[r['arm'], r['case'], r['policy']].append(r)
    lines = ['# AgentX：根会话的已就绪请求覆盖时间', '',
        '使用执行压缩回放的实际墙钟，不包含被跳过的原始长间隔；不是原始会话在真实时间中的完成时间。'
        '[压缩量与分支时间协议](../TIMING_AUDIT.md)。', '',
        '从根会话第一个请求就绪，到它及子代理的最后一个已捕获LLM响应结束。'
        '先对所有子请求的“已就绪、未完成”时间区间取并集，避免并行子代理重复计时；'
        '剩余时间是客户端记录中没有已就绪未完成请求的时段。两项之和等于根会话完成时间。', '',
        '| 布局 | 负载 | 完成轮数 | 根会话平均完成 s | 有已就绪请求未完成 s | 没有已就绪请求未完成 s |',
        '|---|---|---:|---:|---:|---:|']
    for (arm, case, policy), group in groups.items():
        repeats = len({r['repeat'] for r in group})
        means = [st.mean(r[k] for r in group) for k in
                 ('ready_to_last_llm_s', 'with_pending_ready_request_s', 'without_pending_ready_request_s')]
        lines.append(f'| {arm} | {case} | {repeats}/3 | ' + ' | '.join(f'{v:.3f}' for v in means) + ' |')
    lines += ['', '各根会话等权；未满三轮的行是进度。是否整组验收通过见[主表](REPORT.md)。'
        '有未完成请求的时间包含客户端排队、P/D交接及服务等待，不等于GPU计算时间；'
        '另一项包含工具／依赖等待和回放轮询，不能全部归为真实工具执行。'
        '沿用全局空闲压缩规则，不据此推算无成本加速上限。', '',
        '[各根会话原值](root_exposure.csv) · [原始请求来源](root_exposure_sources.json)']
    (out/'ROOT_EXPOSURE.md').write_text('\n'.join(lines)+'\n')
    (out/'root_exposure_sources.json').write_text(json.dumps(sources, indent=2)+'\n')
    print(json.dumps(dict(episodes=len(sources), roots=len(rows))))


if __name__ == '__main__':
    main()
