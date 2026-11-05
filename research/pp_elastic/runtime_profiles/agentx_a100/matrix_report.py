"""Incremental Markdown result index. Keep old and concurrent report sections."""
import fcntl
import json
from pathlib import Path
import time

BEGIN = '<!-- agentx-a100-a100_e10_r7-live -->'
END = '<!-- /agentx-a100-a100_e10_r7-live -->'


def update_report(run):
    run = Path(run)
    cfg = json.loads((run/'config.json').read_text())
    rows = []
    for arm, points in cfg['matrix_points'].items():
        for c in points:
            out = run/'results'/arm/f'C{c:03d}'
            m = json.loads((out/'metrics.json').read_text()) if (out/'metrics.json').exists() else None
            failed = json.loads((out/'failed.json').read_text()) if (out/'failed.json').exists() else None
            rows.append(dict(layout=arm, concurrency=c, metrics=m, failed=failed, path=str(out)))
    done = sum(bool(r['metrics'] and r['metrics']['qualified']) for r in rows)
    lines = [BEGIN, '### A100 AgentX：缺口补测（独立协议）', '',
             f'已核对 {done}/{len(rows)} 点（仅未通过点）；[运行目录]({run})。',
             '每格独立官方预热，180 秒发送＋最多1200秒排空；C 是根任务树并发。HiCache 开启，GPU 静态比例 0.90。',
             'E10引擎、A100 BF16/Triton、EP2 DeepEP normal / EP4原生NCCL all-gather、Eager；算术平均绝对毫秒。单窗口不代表重复实验的置信区间。', '',
             '| 布局 | C | 状态 | 正式请求 | 平均 TTFT ms | 平均 TPOT ms | 平均 E2E ms | 总命中 % |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        m = r['metrics']
        status = '计时完成，待独立审计' if m and m['qualified'] else ('未通过' if m or r['failed'] else '待测')
        values = [m.get(k) if m else None for k in ('profile_requests','mean_ttft_ms','mean_tpot_ms','mean_e2e_ms','total_hit_pct')]
        text = [('—' if x is None else (str(x) if i == 0 else f'{x:.2f}')) for i, x in enumerate(values)]
        lines.append('| '+ ' | '.join([cfg['arm_labels'][r['layout']], str(r['concurrency']), status]+text)+' |')
    lines += ['', '失败／取消／缓存记账缺失独立保留，不能把成功子集当作完整通过。详细日志、官方原始导出及采集开销检查均在运行目录。', END]
    section = '\n'.join(lines)
    report = Path(cfg['consolidated_report'])
    with report.with_suffix('.update.lock').open('a+') as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX)
        current = report.read_text()
        if BEGIN in current:
            a = current.index(BEGIN); b = current.index(END, a)+len(END)
            current = current[:a]+section+current[b:]
        else:
            pos = current.find('\n') + 1
            current = current[:pos]+'\n'+section+'\n'+current[pos:]
        tmp = report.with_suffix(f'.{__import__("os").getpid()}.tmp')
        tmp.write_text(current); tmp.replace(report)
    local = run/'matrix_status.json'
    tmp = local.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(updated=time.time(), qualified=done, rows=rows), ensure_ascii=False, indent=2)+'\n')
    tmp.replace(local)
