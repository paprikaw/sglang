"""Post-hoc paired-call audit for one explicitly named, preset SLO grid point.

Observed pass/fail is an outcome, never an online routing feature. This audit
does not transplant call costs between layouts or estimate dynamic goodput.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent / 'matrix_analysis'
ARMS = ('col_ep4', 'col_pp2')


def exposure(rows):
    deltas = defaultdict(int)
    for row in rows:
        deltas[row['started']] += 1
        deltas[row['finished']] -= 1
    times = sorted(deltas)
    areas, counts, count, area, previous = [], [], 0, 0.0, times[0]
    for time in times:
        area += (time - previous)*count
        count += deltas[time]
        assert count >= 0
        areas.append(area); counts.append(count); previous = time
    assert count == 0

    def integral(time):
        i = bisect_right(times, time)-1
        assert i >= 0
        return areas[i]+counts[i]*(time-times[i])

    def mean(start, end):
        assert end > start
        return (integral(end)-integral(start))/(end-start)
    return mean


def atomic(name, text):
    path = ROOT/name
    temp = path.with_suffix(path.suffix+'.slo.tmp')
    temp.write_text(text); temp.replace(path)


def main():
    audit = json.loads((ROOT/'audit.json').read_text())
    accepted = {r['run'] for r in audit['runs'] if r['run_accepted']}
    indexed, sources = defaultdict(dict), []
    for source in audit['sources']:
        path = Path(source['path'])
        arm, case = path.parts[-4:-2]
        if path.parents[4].name not in accepted or arm not in ARMS or case!='agentx_c128':
            continue
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==source['sha256']
        episode = json.loads(raw)
        rows = episode['requests']; occupancy = exposure(rows)
        origin = min(r['started'] for r in rows)
        for item in rows:
            row = dict(item)
            row['episode_start_fraction'] = (row['started']-origin)/episode['wall_s']
            row['http_inflight_after_first'] = occupancy(
                row['started']+row['ttft_ms']/1000, row['finished'])
            row['fresh_input_tokens'] = row['input_tokens']-(row['meta_info'].get('cached_tokens',0) or 0)
            row['host_hit'] = ((row['meta_info'].get('cached_tokens_details') or {}).get('host',0) or 0)>0
            key = (path.stem.split('.')[-1], row['conversation_id'],row['turn'])
            assert key not in indexed[arm]
            indexed[arm][key] = row
        sources.append(source)
    assert set(indexed['col_ep4']) == set(indexed['col_pp2'])
    groups = defaultdict(list)
    repeat_counts = defaultdict(Counter)
    for key,a in indexed['col_ep4'].items():
        b = indexed['col_pp2'][key]
        assert (a['input_tokens'],a['output_tokens'])==(b['input_tokens'],b['output_tokens'])
        def passes(r):
            return r['ttft_ms']+r['client_dispatch_lag_ms']<=1000 and r['tpot_ms']<=10
        pa,pb=passes(a),passes(b)
        group = 'both' if pa and pb else 'ep_only' if pa else 'pp_only' if pb else 'neither'
        groups[group].append((a,b)); repeat_counts[group][key[0]]+=1
    if indexed['col_ep4']:
        for arm, only in (('col_ep4','ep_only'),('col_pp2','pp_only')):
            matching = [r for r in csv.DictReader((ROOT/'slo_means.csv').open())
                        if r['arm']==arm and r['case']=='agentx_c128'
                        and r['ttft_limit_ms']=='1000' and r['tpot_limit_ms']=='10'
                        and r['repeats']=='3']
            assert len(matching)==1
            fraction=(len(groups['both'])+len(groups[only]))/len(indexed[arm])
            assert abs(fraction-float(matching[0]['ready_fraction']))<1e-12
    summaries=[]
    for group in ('both','ep_only','pp_only','neither'):
        pairs=groups[group]
        if not pairs:
            continue
        result=dict(group=group, calls_all_three_repeats=len(pairs),
            calls_per_repeat=dict(repeat_counts[group]),
            mean_input_tokens=st.mean(a['input_tokens'] for a,b in pairs),
            mean_output_tokens=st.mean(a['output_tokens'] for a,b in pairs),
            first_turn_calls=sum(a['turn']==0 for a,b in pairs),
            source_kinds=dict(Counter(a['source_kind'] for a,b in pairs)),
            family_counts=dict(Counter(a['root_id'].rsplit(':',1)[1] for a,b in pairs)))
        for i,arm in enumerate(ARMS):
            rows=[p[i] for p in pairs]
            result[arm]={m:st.mean(r[m] for r in rows) for m in (
                'ttft_ms','tpot_ms','e2e_ms','fresh_input_tokens',
                'http_inflight_after_first','episode_start_fraction')}
            result[arm]['host_hit_calls']=sum(r['host_hit'] for r in rows)
        summaries.append(result)
    atomic('strict_slo_overlap.json',json.dumps(dict(
        case='agentx_c128',ready_ttft_limit_ms=1000,request_tpot_limit_ms=10,
        summaries=summaries,sources=sources,
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    names=dict(both='双方达标',ep_only='仅EP4达标',pp_only='仅PP2达标',neither='双方未达标')
    text=['# 严格延迟目标下，EP4和PP2分别满足哪些请求', '',
          '范围：已完整验收的C128三轮，同一请求身份逐项配对。目标来自既有完整网格：'
          '请求就绪至首token≤1000ms，请求平均TPOT≤10ms/token。'
          '这是看到网格差异后的解释性分析，不是预先规定的独立验证集。', '',
          '**类别按实测结果划分，不能在请求到达时提前知道，也不能直接拿来做动态路由。** '
          '两个配置中的相同请求经历不同的排队、batch、缓存状态和执行次序；'
          '将任一列的更快耗时拼接起来不是动态机制收益。', '',
          '| 实测类别 | 三轮合计请求数 | 平均输入token | 平均输出token | EP4请求E2E ms | PP2请求E2E ms | EP4 TPOT ms/token | PP2 TPOT ms/token |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        a,b=r['col_ep4'],r['col_pp2']
        text.append(f"| {names[r['group']]} | {r['calls_all_three_repeats']} | {r['mean_input_tokens']:.1f} | {r['mean_output_tokens']:.1f} | {a['e2e_ms']:.2f} | {b['e2e_ms']:.2f} | {a['tpot_ms']:.3f} | {b['tpot_ms']:.3f} |")
    text += ['', '## 缓存与请求重叠', '',
             '未缓存输入来自各自服务器返回值；不是相同工作量假设。HTTP在途数为'
             '该请求首token之后直到结束期间的全系统时间平均，包含其自身和P/D混合请求，'
             '**不是GPU Decode batch**。三轮及副本仍不是独立工作负载家族。', '',
             '| 实测类别 | EP4未缓存输入token | PP2未缓存输入token | EP4 HTTP在途均值 | PP2 HTTP在途均值 | EP4 host命中请求数 | PP2 host命中请求数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        a,b=r['col_ep4'],r['col_pp2']
        text.append(f"| {names[r['group']]} | {a['fresh_input_tokens']:.1f} | {b['fresh_input_tokens']:.1f} | {a['http_inflight_after_first']:.2f} | {b['http_inflight_after_first']:.2f} | {a['host_hit_calls']} | {b['host_hit_calls']} |")
    text += ['', '这些记录可用于提出下一步受控验证的条件，但还不能区分全部因果因素。'
             '输出长度也是当前固定长度回放的观测，不代表在线服务能预测自然终止长度。', '',
             '[每轮数量、完整特征与来源哈希](strict_slo_overlap.json) · '
             '[完整SLO网格与所有家族](FAMILY_TRADEOFFS.md)', '']
    atomic('STRICT_SLO_OVERLAP.md','\n'.join(text))
    print(json.dumps(dict(paired_calls=len(indexed['col_ep4']),groups=len(summaries))))


if __name__=='__main__':
    main()
