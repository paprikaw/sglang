"""One Markdown entry for new, fully qualified Decode optimization windows."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

MARKER = 'agentx-decode-pp-optimization-20261101'


def json_lines(path):
    with path.open() as source:
        for line in source:
            yield json.loads(line)


def phases(run, arm):
    root = json.loads((run/'config.json').read_text())
    out = run/'results'/arm
    export = out/'replay/agentx/profile_export.jsonl'
    if not export.exists():
        return None
    meta = [v['metadata'] for v in json_lines(export)
            if v['metadata'].get('benchmark_phase') == 'profiling']
    if not meta:
        return None
    start = min(v['request_start_ns'] for v in meta)
    end = max(v['request_end_ns'] for v in meta)
    gpu = defaultdict(float)
    counters = defaultdict(int)
    cpu = defaultdict(lambda: [0, 0.0])
    per_rank = []
    for node in root['nodes']:
        layout = next(x for x in root['node_configs'][node]['layouts'] if x['id'] == arm)
        for i, inst in enumerate(layout['instances']):
            parent = run/'evidence'/node/f'{arm}.r{i}'/'phase_timing'
            for path in sorted(parent.glob('*.jsonl')):
                samples = []
                for row in json_lines(path):
                    if row['kind'] == 'forward' and start <= row['start_ns'] <= row['end_ns'] <= end:
                        gpu[inst['role']] += row['gpu_stream_ms']
                        counters[inst['role']] += 1
                        samples.append(row)
                    elif row['kind'] == 'cpu_control' and start/1e9 <= row['second'] and row['second']+1 <= end/1e9:
                        value = cpu[f'{node}.r{i}.{path.stem}:{row["method"]}']
                        value[0] += row['calls']
                        value[1] += row['cpu_inclusive_ms']
                per_rank.append(dict(node=node, instance=i, rank=path.stem, role=inst['role'],
                    forwards=len(samples), gpu_stream_ms=sum(r['gpu_stream_ms'] for r in samples),
                    mean_batch_size=statistics.mean(r['batch_size'] for r in samples) if samples else None,
                    input_rows=sum(r.get('input_rows',0) for r in samples)))
    rows = json.loads((out/'accounted_requests.json').read_text())
    request_total = sum(r['e2e_ms'] for r in rows)
    before = sum(r['ttft_ms'] for r in rows)
    elapsed_gpu = sum(gpu.values())
    result = dict(request_window_start_ns=start, request_window_end_ns=end,
        request_wall_s=(end-start)/1e9, request_before_first_share=before/request_total,
        request_after_first_share=1-before/request_total,
        role_forward_gpu_stream_ms=dict(gpu), role_forward_counts=dict(counters),
        prefill_share_of_forward_gpu_stream_time=(gpu['prefill']/elapsed_gpu if elapsed_gpu else None),
        per_rank=per_rank, cpu_control_inclusive={k:dict(calls=v[0],ms=v[1]) for k,v in cpu.items()},
        logical_input_tokens=sum(r['prompt_tokens'] for r in rows),
        uncached_prefill_tokens=sum(r['new_tokens'] for r in rows if 'new_tokens' in r),
        uncached_prefill_accounted_requests=sum('new_tokens' in r for r in rows),
        logical_output_tokens=sum(r['output_tokens'] for r in rows),
        mean_input_tokens=statistics.mean(r['prompt_tokens'] for r in rows),
        mean_output_tokens=statistics.mean(r['output_tokens'] for r in rows),
        limitations=['CUDA event forward-stream spans include launch gaps and waits; not pure kernel busy time.',
            'GPU forward spans exclude P/D handoff, request queues and PP communication outside ModelRunner.forward.',
            'CPU control timings are inclusive and can nest; do not sum different methods.',
            'GPU values are summed over physical ranks; request time fractions are cumulative client time, not wall fractions.',
            'Boundary-crossing forwards and fractional CPU seconds are excluded.',
            'Warmup is excluded by official profiling request timestamps; closed-loop samples differ across layouts.'])
    (out/'phase_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def update_report(run):
    run = Path(run)
    root = json.loads((run/'config.json').read_text())
    lines = [f'<!-- {MARKER} -->', '### 2P6D：PP2执行／分流优化与EP2对照（11 月 1 日新增）', '',
        f'运行目录：[{run.name}]({run})。保持官方 AgentX 1.0.7、34 家族、C128、完整预热保缓存、180 秒发送与 240 秒排空 grace。',
        'GPU 静态比例 0.90、全组 640 decimal GB CPU HiCache；先核对完整请求、缓存明细和串行输出，再列性能均值。', '',
        '| 配置 | 正式请求 | 平均 TTFT ms | 平均 TPOT ms/token | 平均 E2E ms | 客户端首 token 前占比 % | P 占 forward GPU-stream 时间 % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in root['arms']:
        path = run/'results'/arm/'metrics.json'
        if not path.exists():
            lines.append(f'| {root["arm_labels"][arm]} | 待完成 | — | — | — | — | — |')
            continue
        m = json.loads(path.read_text())
        if not m['qualified']:
            lines.append(f'| {root["arm_labels"][arm]} | 未通过完整性核对 | — | — | — | — | — |')
            continue
        values = m.get('mean', m.get('means', {}))
        if not values:
            rows = json.loads((path.parent/'accounted_requests.json').read_text())
            values = {k:statistics.mean(r[k] for r in rows if r[k] is not None)
                      for k in ['ttft_ms','tpot_ms','e2e_ms']}
        p = phases(run, arm)
        before = f'{100*p["request_before_first_share"]:.2f}' if p else '—'
        gpu = (f'{100*p["prefill_share_of_forward_gpu_stream_time"]:.2f}'
               if p and p['prefill_share_of_forward_gpu_stream_time'] is not None else '—')
        lines.append(f'| {root["arm_labels"][arm]} | {m["profile_requests"]} | '
            f'{values["ttft_ms"]:.2f} | {values["tpot_ms"]:.2f} | {values["e2e_ms"]:.2f} | {before} | {gpu} |')
    lines += ['', '本组固定P=PP2和另一组D=EP4（2P6D）；对比原生PP2 D、提前激活传输＋每4微步HiCache轮询的PP2 D、替换为EP2 D，以及优化PP2的服务时间分流。所有布局继续保留HiCache写回、恢复与P/D交接检查。EP4＋EP2与EP4＋PP2的D请求上限均合计96、CPU HiCache均合计192 GB。',
        '固定分流沿用EP4:PP2的2:1加权粘性分流；服务时间分流以串行资格检查作为初始TPOT估计、完成请求更新EWMA，PP2最多接纳4个在途客户端请求。仅在请求边界改分配，P侧会话亲和不变；D侧迁移及缓存影响另记。',
        'CUDA event 记录 forward stream 的执行区间，包含其等待与提交间隙，不是纯 kernel busy 时间；未覆盖 forward 之外的流水通信及 P/D 交接。客户端首 token 前包含排队、恢复和交接，不能当作纯 Prefill。两类占比均不能直接换算为最优卡数。',
        '每组的 phase_metrics.json 保存逐 rank forward 时间、实际 batch、客户端阶段占比与输入／输出 token；CPU 控制计时为可能嵌套的 inclusive 时间。只计算正式窗口，不计预热。',
        f'[冻结配置]({run}/config.json) · [结果与阶段分项]({run}/results)。单轮闭环 pilot，尚无动态收益或普遍排名结论。', f'<!-- /{MARKER} -->']
    section='\n'.join(lines)
    report=Path(root['consolidated_report'])
    text=report.read_text()
    begin=f'<!-- {MARKER} -->'; end=f'<!-- /{MARKER} -->'
    if begin in text:
        left=text.index(begin); right=text.index(end,left)+len(end)
        text=text[:left]+section+text[right:]
    else:
        anchor='## 历史数据审计（以下保留当时结论）'
        text=text.replace(anchor,section+'\n\n'+anchor,1)
    report.write_text(text)
