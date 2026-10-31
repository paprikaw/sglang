"""Recompute request/conversation profiles from audited, completed AgentX episodes."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
AUDIT = HERE.parent / 'agentx_data_audit_20261031'
METRICS = ['context_tokens', 'new_prefill_tokens', 'decode_tokens',
           'prefix_hit_pct', 'device_hit_pct', 'host_hit_pct',
           'ttft_ms', 'tpot_ms', 'e2e_ms']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, q):
    a = sorted(values)
    if not a:
        return None
    pos = (len(a)-1)*q
    lo = int(pos)
    return a[lo] + (a[min(lo+1, len(a)-1)]-a[lo])*(pos-lo)


def distribution(values):
    a = [v for v in values if v is not None]
    return dict(n=len(a), mean=statistics.fmean(a), **{
        'p'+str(q): percentile(a, q/100) for q in (10, 25, 50, 75, 90, 95, 99)})


def write_csv(name, rows):
    assert rows
    with (HERE/name).open('w', newline='') as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def identity(value):
    return value.split(':', 2)[-1]


def main():
    episodes = list(csv.DictReader((AUDIT/'accepted_episodes.csv').open()))
    rows, inputs = [], []
    for ep in episodes:
        source = Path(ep['source'])
        assert sha(source) == ep['sha256'], source
        data = json.loads(source.read_text())
        requests = data['requests']
        assert len(requests) == int(ep['completed_requests']) == data['completed_requests']
        inputs.append(dict(path=str(source), sha256=ep['sha256'], cohort=ep['cohort']))
        for r in requests:
            is_pd = ep['arm'].startswith('pd_')
            meta = r['prefill_meta'] if is_pd else r['meta_info']
            cached = meta['cached_tokens']
            context, decode = r['input_tokens'], r['output_tokens']
            assert 0 <= cached <= context and decode > 0
            details = meta.get('cached_tokens_details')
            if details is None:
                assert cached == 0, 'A nonzero cache hit has no location evidence'
                details = dict(device=0, host=0)
            assert details['device']+details['host'] == cached
            rows.append(dict(cohort=ep['cohort'], run=ep['run'], arm=ep['arm'],
                case=ep['case'], repeat=ep['repeat'],
                root_id=r['root_id'], conversation_id=r['conversation_id'],
                source_family=identity(r['root_id']), source_conversation=identity(r['conversation_id']),
                turn=r['turn'], context_tokens=context, cached_tokens=cached,
                device_cached_tokens=details['device'], host_cached_tokens=details['host'],
                new_prefill_tokens=context-cached, decode_tokens=decode,
                prefix_hit_pct=100*cached/context, device_hit_pct=100*details['device']/context,
                host_hit_pct=100*details['host']/context,
                ttft_ms=r['ttft_ms'], tpot_ms=r['tpot_ms'], e2e_ms=r['e2e_ms'],
                cache_metadata_role='P' if is_pd else 'colocated'))
    assert len(rows) == 34371 and len(episodes) == 21
    grouped = defaultdict(list)
    conversations = defaultdict(list)
    for row in rows:
        grouped[(row['cohort'], row['arm'], row['case'])].append(row)
        conversations[(row['cohort'], row['run'], row['arm'], row['case'],
                       row['repeat'], row['root_id'], row['conversation_id'])].append(row)
    summaries, distributions, conv_rows = [], [], []
    for key, group in sorted(grouped.items()):
        cohort, arm, case = key
        total_context = sum(r['context_tokens'] for r in group)
        total_decode = sum(r['decode_tokens'] for r in group)
        total_new = sum(r['new_prefill_tokens'] for r in group)
        sums = dict(cohort=cohort, arm=arm, case=case, episodes=3,
            requests=len(group), mean_context_tokens=total_context/len(group),
            mean_decode_tokens=total_decode/len(group), mean_new_prefill_tokens=total_new/len(group),
            token_weighted_prefix_hit_pct=100*sum(r['cached_tokens'] for r in group)/total_context,
            token_weighted_device_hit_pct=100*sum(r['device_cached_tokens'] for r in group)/total_context,
            token_weighted_host_hit_pct=100*sum(r['host_cached_tokens'] for r in group)/total_context,
            request_mean_prefix_hit_pct=statistics.fmean(r['prefix_hit_pct'] for r in group),
            context_to_decode_token_ratio=total_context/total_decode,
            new_prefill_to_decode_token_ratio=total_new/total_decode)
        summaries.append(sums)
        for metric in METRICS:
            distributions.append(dict(cohort=cohort, arm=arm, case=case,
                metric=metric, **distribution([r[metric] for r in group])))
    for key, group in sorted(conversations.items()):
        cohort, run, arm, case, repeat, root_id, conversation_id = key
        ctx = sum(r['context_tokens'] for r in group)
        output = sum(r['decode_tokens'] for r in group)
        fresh = sum(r['new_prefill_tokens'] for r in group)
        conv_rows.append(dict(cohort=cohort,run=run,arm=arm,case=case,repeat=repeat,
            root_id=root_id,conversation_id=conversation_id,
            source_conversation=identity(conversation_id),turns=len(group),
            mean_context_tokens=ctx/len(group),mean_new_prefill_tokens=fresh/len(group),
            mean_decode_tokens=output/len(group),cumulative_context_tokens=ctx,
            cumulative_new_prefill_tokens=fresh,cumulative_decode_tokens=output,
            token_weighted_prefix_hit_pct=100*(ctx-fresh)/ctx,
            token_weighted_device_hit_pct=100*sum(r['device_cached_tokens'] for r in group)/ctx,
            token_weighted_host_hit_pct=100*sum(r['host_cached_tokens'] for r in group)/ctx,
            new_prefill_to_decode_token_ratio=fresh/output,
            mean_ttft_ms=statistics.fmean(r['ttft_ms'] for r in group),
            mean_tpot_ms=statistics.fmean(r['tpot_ms'] for r in group),
            mean_e2e_ms=statistics.fmean(r['e2e_ms'] for r in group)))
    conv_distributions = []
    for key in sorted(grouped):
        group = [r for r in conv_rows if (r['cohort'],r['arm'],r['case']) == key]
        for metric in ['turns','mean_context_tokens','mean_new_prefill_tokens',
                       'mean_decode_tokens','token_weighted_prefix_hit_pct',
                       'cumulative_context_tokens','cumulative_new_prefill_tokens','cumulative_decode_tokens',
                       'new_prefill_to_decode_token_ratio','mean_ttft_ms','mean_tpot_ms','mean_e2e_ms']:
            conv_distributions.append(dict(cohort=key[0],arm=key[1],case=key[2],
                metric=metric, **distribution([r[metric] for r in group])))
    source = [r for r in rows if r['arm']=='col_ep4' and r['case']=='agentx_c8' and r['repeat']=='r1']
    assert len(source) == 171
    families=[]
    for family in sorted({r['source_family'] for r in source}):
        group=[r for r in source if r['source_family']==family]
        families.append(dict(source_family=family, conversations=len({r['conversation_id'] for r in group}),
            calls=len(group),mean_context_tokens=statistics.fmean(r['context_tokens'] for r in group),
            context_p50=percentile([r['context_tokens'] for r in group],.5),
            mean_decode_tokens=statistics.fmean(r['decode_tokens'] for r in group),
            cumulative_decode_tokens=sum(r['decode_tokens'] for r in group)))
    for name, data in [('request_profiles.csv',rows),('summary.csv',summaries),
            ('request_distributions.csv',distributions),('conversation_profiles.csv',conv_rows),
            ('conversation_distributions.csv',conv_distributions),('source_families.csv',families)]:
        write_csv(name, data)
    manifest=dict(id='qwen35b_h100_agentx_workload_profile_20261031',
        analyzer=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__))),
        episodes=21,requests=len(rows),source_families=len(families),
        source_conversations=len({r['conversation_id'] for r in source}),
        raw_sources=inputs, cache_definition='P-side prefill_meta for PD; colocated meta_info otherwise',
        distributions='Three episodes pooled within exact cohort/arm/case; percentiles use linear interpolation',
        trial_and_clone_caveat='Trace copies and repeats are not independent diverse conversations')
    (HERE/'profile_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    labels={'col_ep4':'混部 EP4','col_pp2':'混部 PP2','col_single':'混部单卡','pd_ep4':'4P4D EP4（P 端）'}
    lines=['# AgentX：C8／C128 对话、Prefill／Decode 与缓存分布', '',
        '统计来自已审计的 21 个完整 episode、34,371 条请求；每个点三轮。C8 为 8 个根会话族；C128 将同样 8 族各复制 16 份，保持缓存隔离。C 是客户端根会话上限，不是每张 GPU 的 Decode batch。', '',
        f'单份 trace 有 **8 个根会话族、{manifest["source_conversations"]} 条含子代理的独立对话、171 次调用**。C8 与 C128 的输入／输出长度分布相同，因此下面先列一份来源分布，缓存和实际新增 Prefill 再按配置分别列。', '',
        '## 口径', '',
        '- 完整 context＝本次请求输入 token，包含可复用前缀；新增 Prefill＝输入 − P 端实际 cached_tokens；Decode＝本次生成 token。三者是 token 数，不能当作 GPU 时间比例。',
        '- 前缀命中率主表按 token 加权：Σcached／Σinput；CSV 另外保留逐请求命中率的平均值和分位数。GPU 命中和 CPU HiCache 命中分列；CPU 命中仍有恢复开销。',
        '- PD 使用 `prefill_meta`；D 端 `meta_info.cached_tokens` 不用于估算 Prefill。新增 Prefill 是服务端报告的未命中输入量，不等于采样得到的精确 kernel 行数或传输字节。',
        '- 对话表以每条独立对话为单位；根会话族含全部子代理另有汇总。累计 context 会在相邻轮重复计算同一前缀，不代表显存占用。', '',
        '## 来源长度：每次调用', '',
        '| 指标（token） | 平均 | P10 | P25 | P50 | P75 | P90 | P99 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for metric,label in [('context_tokens','完整 context'),('decode_tokens','Decode 输出')]:
        d=distribution([r[metric] for r in source])
        lines.append('| '+label+' | '+' | '.join(f'{d[k]:,.1f}' for k in ('mean','p10','p25','p50','p75','p90','p99'))+' |')
    lines += ['', '## 缓存决定真正的 Prefill 工作量', '',
        '| 部署／来源版本 | 并发 | 新增 Prefill 均值 token | 总前缀命中 | GPU 命中 | CPU 命中 | 新增 P／D token 比 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f'| {labels[s["arm"]]} · {s["cohort"]} | {s["case"][8:]} | {s["mean_new_prefill_tokens"]:,.1f} | {s["token_weighted_prefix_hit_pct"]:.2f}% | {s["token_weighted_device_hit_pct"]:.2f}% | {s["token_weighted_host_hit_pct"]:.2f}% | {s["new_prefill_to_decode_token_ratio"]:.2f}:1 |')
    lines += ['', '**解释：** 完整输入／输出的 token 比对所有点都相同；新增 P／D 比却随缓存变化。高并发下单卡的 CPU 恢复比例、PP／EP 的 GPU 命中率，以及 PD 的 P 端命中必须分别观察。该表包含 E5/V3 混部与 E7/V3 PD 两个历史 cohort，不能把差异直接归因于 P/D 分离。', '',
        '## 每条对话：先对各对话求均值，再看对话之间的分布', '',
        '| 指标 | 均值 | P25 | P50 | P75 | P90 | P99 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    source_convs=[r for r in conv_rows if r['arm']=='col_ep4' and r['case']=='agentx_c8' and r['repeat']=='r1']
    for metric,label in [('turns','每对话调用数'),('mean_context_tokens','每次调用 context token'),('mean_decode_tokens','每次调用 Decode token')]:
        d=distribution([r[metric] for r in source_convs])
        lines.append('| '+label+' | '+' | '.join(f'{d[k]:,.1f}' for k in ('mean','p25','p50','p75','p90','p99'))+' |')
    lines += ['', '上表的每条对话权重相等，与逐请求均值不同。各部署下每条对话的新增 Prefill、命中率、P／D 比、延迟均值及其分布见两个 conversation CSV。', '',
        '### 一整条对话累计要做多少工作', '',
        '每条独立对话平均9.5次调用、累计生成4,139.3 token。下面按对话累计后再求平均；它衡量整段对话工作量，不能与上表“每条对话的每次调用均值”混淆。', '',
        '| 部署 · C | 平均每对话累计新增 P token | 平均每对话累计 D token | 对话前缀命中率均值 |',
        '|---|---:|---:|---:|']
    for key in sorted(grouped):
        group=[r for r in conv_rows if (r['cohort'],r['arm'],r['case'])==key]
        lines.append(f'| {labels[key[1]]} · C{key[2][8:]} | {statistics.fmean(r["cumulative_new_prefill_tokens"] for r in group):,.1f} | {statistics.fmean(r["cumulative_decode_tokens"] for r in group):,.1f} | {statistics.fmean(r["token_weighted_prefix_hit_pct"] for r in group):.2f}% |')
    lines += ['', '这里的命中率是每条对话 Σcached／Σinput 后对对话等权平均，因此与前面的全体 token 加权命中不同。累计新增 P／D 的分位数已补入 conversation_distributions.csv。', '',
        '| 来源对话 | 调用数 | 平均 context token | 平均 Decode token | C8 平均新增 P token | C8 前缀命中 |',
        '|---|---:|---:|---:|---:|---:|']
    for i,r in enumerate(sorted(source_convs,key=lambda r:r['source_conversation']),1):
        lines.append(f'| C{i:02d} · `{r["source_conversation"][:10]}` | {r["turns"]} | {r["mean_context_tokens"]:,.1f} | {r["mean_decode_tokens"]:,.1f} | {r["mean_new_prefill_tokens"]:,.1f} | {r["token_weighted_prefix_hit_pct"]:.2f}% |')
    lines += ['', 'C8 三种混部的缓存结果完全相同，所以这张来源对话表只列一次；完整 ID 和 C128 各部署的对话数据均在 CSV。', '',
        '## 新增 Prefill／命中率／延迟的请求分布', '',
        '| 部署 · C | 指标 | 平均 | P25 | P50 | P75 | P90 | P99 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    names={'new_prefill_tokens':'新增 P token','prefix_hit_pct':'前缀命中 %','ttft_ms':'TTFT ms','tpot_ms':'TPOT ms','e2e_ms':'E2E ms'}
    for d in distributions:
        if d['metric'] not in names:
            continue
        lines.append(f'| {labels[d["arm"]]} · C{d["case"][8:]} | {names[d["metric"]]} | '+' | '.join(f'{d[k]:,.2f}' for k in ('mean','p25','p50','p75','p90','p99'))+' |')
    lines += ['', 'TTFT／E2E 从 HTTP 发起时计时；客户端 ready→HTTP 等待另见已有审计，不能当作完整 ready 延迟。TPOT 是逐请求平均 token 间隔，再对请求统计，未按输出长度加权；也不是单个 token 的 ITL 尾延迟。', '',
        '## 数据入口与边界', '',
        '- [逐请求归一化数据](request_profiles.csv)：含 P 端 cache 元数据口径，无输出 token ID。',
        '- [部署均值与 token 比](summary.csv)；[逐请求各指标分布](request_distributions.csv)。',
        '- [逐对话均值和累计量](conversation_profiles.csv)；[对话间分布](conversation_distributions.csv)。',
        '- [8 个来源会话族](source_families.csv)；[原始文件路径与哈希](profile_manifest.json)；[复算脚本](analyze.py)。',
        '- [既有数据及协议审计](../agentx_data_audit_20261031/REPORT.md)：V3 子代理时序偏差、合成 token、筛选与复制、输出不一致、压缩到达间隔均保留。不能宣称官方 AgentX 得分或动态调度收益。',
        '- C8 主表尚无同一 E7/V4 PD 对照；新一轮固定使用修正后的时序，独立归档，不能直接合并成同源性能排名。', '']
    (HERE/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(summary=summaries,source_conversations=manifest['source_conversations'],
        distributions=distribution([r['context_tokens'] for r in source])),indent=2))


if __name__ == '__main__':
    main()
