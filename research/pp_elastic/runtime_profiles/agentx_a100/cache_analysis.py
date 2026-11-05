"""Join official phase/request identities to actual SGLang cache accounting."""
import fcntl
import json
from pathlib import Path
import statistics

BEGIN = '<!-- agentx-pp-prefill-layouts-20261101 -->'
END = '<!-- /agentx-pp-prefill-layouts-20261101 -->'


def analyze_arm(run, arm):
    run = Path(run)
    root = run/'results'/arm/'replay'
    records = [json.loads(line) for line in (root/'agentx/profile_export.jsonl').open()]
    routed = {r['x_request_id']: r for line in (root/'routing.jsonl').open()
              if (r := json.loads(line)).get('x_request_id')}
    phases = {}
    for r in records:
        phase = r['metadata']['benchmark_phase']
        phases[phase] = phases.get(phase, 0) + 1
    profile = [r for r in records if r['metadata']['benchmark_phase'] == 'profiling']
    failures, missing, rows = [], [], []
    for r in profile:
        m = r['metadata']; rid = m['x_request_id']; route = routed.get(rid)
        if m.get('was_cancelled') or r.get('error') or not route or not route['success']:
            failures.append(rid)
            continue
        usage = route['prefill_usage'] if route['mode'] == 'pd' else route['usage']
        details = (usage or {}).get('_sglang_cached_tokens_details')
        if not usage:
            missing.append(rid)
            continue
        cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
        # SGLang omits both optional cache objects on a cold miss, even with
        # --enable-cache-report and return_cached_tokens_details enabled.
        # This exact wire form was observed on r6; nonzero hits still require detail.
        omitted_zero = details is None and usage.get('prompt_tokens_details') is None
        if details is None and omitted_zero:
            details = dict(device=0, host=0)
        if details is None:
            missing.append(rid)
            continue
        total = usage['prompt_tokens']; device = details.get('device', 0); host = details.get('host', 0)
        storage = details.get('storage', 0) or 0
        if not (0 <= device + host + storage == cached <= total):
            missing.append(rid)
            continue
        metrics = r['metrics']
        def value(name):
            v = metrics.get(name)
            return v['value'] if v is not None else None
        rows.append(dict(x_request_id=rid, prompt_tokens=total, device=device, host=host,
            cached=cached, zero_hit_fields_omitted=omitted_zero, new_tokens=total-cached, output_tokens=value('output_sequence_length'),
            ttft_ms=value('time_to_first_token'), tpot_ms=value('inter_token_latency'),
            e2e_ms=value('request_latency'), source_trace=m.get('source_trace_id'),
            source_outer=m.get('source_outer_idx'), source_inner=m.get('source_inner_idx')))
    denominator = sum(r['prompt_tokens'] for r in rows)
    def mean(field):
        v = [r[field] for r in rows if r[field] is not None]
        return statistics.mean(v) if v else None
    manifest = json.loads((root/'manifest.json').read_text())
    result = dict(arm=arm, qualified=bool(profile) and not failures and not missing,
        phase_counts=phases, profile_requests=len(profile), accounted_requests=len(rows),
        failed_or_cancelled_profile=len(failures), missing_or_inconsistent_cache=len(missing),
        zero_hit_fields_omitted=sum(r['zero_hit_fields_omitted'] for r in rows),
        input_tokens=denominator, output_tokens=sum(r['output_tokens'] or 0 for r in rows),
        gpu_hit_pct=100*sum(r['device'] for r in rows)/denominator if denominator else None,
        cpu_hit_pct=100*sum(r['host'] for r in rows)/denominator if denominator else None,
        total_hit_pct=100*sum(r['cached'] for r in rows)/denominator if denominator else None,
        mean_new_tokens=mean('new_tokens'), mean_ttft_ms=mean('ttft_ms'),
        mean_tpot_ms=mean('tpot_ms'), mean_e2e_ms=mean('e2e_ms'),
        actual_source_families=len({r['source_trace'] for r in rows}),
        protocol='Official agentx 1.0.7, bounded filtered research subset; profiling only; token-weighted cache hits',
        replay_status=manifest['status'], failed_request_ids=failures[:20], missing_request_ids=missing[:20])
    (run/'results'/arm/'accounted_requests.json').write_text(json.dumps(rows)+'\n')
    return result


def update_report(run):
    run = Path(run)
    cfg = json.loads((run/'config.json').read_text())
    report = Path(cfg['consolidated_report'])
    rows = []
    for arm in cfg['arms']:
        p = run/'results'/arm/'metrics.json'
        rows.append((arm, json.loads(p.read_text()) if p.exists() else None))
    text = [BEGIN, '### 新增：PP Prefill 的两种 P/D 布局（自动更新）', '',
        f"运行目录：[{run.name}]({run})；C{cfg['agentx_concurrency']}，{cfg['dataset_entries']} 个完整来源家族，seed42。",
        f"正式发送窗口均为 {cfg['profile_seconds']} 秒；本次新执行组的回放总限 {cfg['replay_timeout_seconds']} 秒（含重建／预热／排空），控制器总限 {cfg['total_timeout_seconds']} 秒。导入结果保留来源运行的时限与状态。",
        '', '| 配置 | CPU HiCache／rank | 状态 | 正式请求 | GPU命中 % | CPU命中 % | 总命中 % | 平均 TTFT ms | 平均 TPOT ms | 平均 E2E ms |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):
        return '—' if v is None else f'{v:.2f}'
    for arm, d in rows:
        label = cfg['arm_labels'][arm]
        capacity = cfg['arm_cache_labels'][arm]
        status = '待完成' if d is None else ('已完成' if d['qualified'] else '未通过核对')
        if d and d.get('wrapper_timeout_after_complete_request_window'):
            status = '请求完整；收尾超时'
        values = ['—']*7 if d is None else [str(d['profile_requests'])]+[fmt(d[k]) for k in
            ['gpu_hit_pct','cpu_hit_pct','total_hit_pct','mean_ttft_ms','mean_tpot_ms','mean_e2e_ms']]
        text.append('| '+' | '.join([label,capacity,status]+values)+' |')
    text += ['', '命中分母为正式请求的原始输入 token，逐条按官方 request id／phase 关联；PD 只取 P 端。延迟为请求算术平均，warmup 不计入；中止、失败、缓存明细缺失均不冒充完整成功组。',
        '两种新布局的 CPU HiCache 均合计 640 decimal GB：4P4D 为 P4×128＋D4×32；2P6D 为 P2×224＋D6×32。GPU 静态显存比例统一 0.90，原生自动 KV 容量，GDN 预算按每张卡等额分配。物理容量与每个 owner 的缓存压力另列。',
        '这是短窗口单轮实验；固定时长闭环会推进不同请求混合，尚不能把总体命中差全部归因于布局或推广为全量 AgentX 结论。', END]
    if cfg.get('imported_arms'):
        provenance = '；'.join(f"{cfg['arm_labels'][arm]} 沿用 [{Path(info['run']).name}]({info['run']}) 的已核对结果" for arm, info in cfg['imported_arms'].items())
        text.insert(-1, provenance+'。此项没有重跑，也不算新增独立轮次；本运行只补测剩余配置。')
    if any(d and d.get('wrapper_timeout_after_complete_request_window') for _, d in rows):
        text.insert(-1, 'PD128 的 180 秒发送窗口和全部正式请求已独立核对，外层总限在官方分支排空等待期间触发；保留原失败 manifest，不称为完整官方回放通过。数据只作为完整请求窗口的测量，验证证据在来源运行的 analysis/sealed_pd_ep4_p128_d32/completion_proof.json。')
    text.insert(-1, '4P4D：011 上两个 PP2 P，013 上一个 EP4 D；全部交接跨节点。2P6D：011 上一个 PP2 P 与一个 PP2 D，013 上一个 EP4 D；D 按并发容量作约 2:1 的固定粘性分流，实际比例另列，同节点交接与跨节点交接分别统计。')
    section = '\n'.join(text)
    lock = report.with_suffix('.update.lock')
    with lock.open('a+') as handle:
        fcntl.lockf(handle, fcntl.LOCK_EX)
        current = report.read_text()
        if BEGIN in current:
            a = current.index(BEGIN); b = current.index(END, a)+len(END)
            current = current[:a]+section+current[b:]
        else:
            marker = '## 历史数据审计（以下保留当时结论）'
            current = current.replace(marker, section+'\n\n'+marker, 1)
        temporary = report.with_suffix('.md.tmp')
        temporary.write_text(current)
        temporary.replace(report)
        fcntl.lockf(handle, fcntl.LOCK_UN)

from optimization_report import update_report
