"""Analyze saved formal cohorts; never include profiler or cache preparation."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics


def analyze(run):
    cfg=json.loads((run/'config.json').read_text())
    complete=json.loads((run/'completed.json').read_text())
    assert complete['passed']
    groups=defaultdict(list)
    sources=[]
    for p in sorted(run.glob('*/metrics.*.json')):
        r=json.loads(p.read_text())
        groups[(r['case']['id'],r['arm'],r.get('chunk_tokens',r.get('coalescing',r.get('ready_send'))))].append(r)
        sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    assert len(sources)==cfg['expected_points']
    rows=[]
    for (case,arm,mode),points in groups.items():
        flat=[x for p in points for x in p['result']['requests']]
        meanspan=statistics.mean(p['result']['makespan_s'] for p in points)
        row=dict(case=case,configuration=arm,policy=mode,
                 samples=len(points),requests_per_sample=points[0]['case']['requests'],
                 cached_tokens_per_request=points[0]['case']['cached_tokens'],
                 new_prefill_tokens_per_request=points[0]['case']['new_tokens'],
                 throughput_new_tokens_s=statistics.mean(p['new_token_throughput'] for p in points),
                 mean_cohort_duration_ms=meanspan*1000,
                 trial_durations_ms=[p['result']['makespan_s']*1000 for p in points])
        for metric in ('ttft_s','e2e_s'):
            values=sorted(x[metric]*1000 for x in flat)
            for label,value in [('mean',statistics.mean(values)),('median',statistics.median(values)),
                                ('p99',values[min(len(values)-1,int(.99*len(values)))])]:
                row[metric.replace('_s','_ms')+'_'+label]=value
        rows.append(row)
    out=run/'analysis';out.mkdir(exist_ok=True)
    with (out/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(dict(rows=rows,sources=sources),indent=2)+'\n')
    chunk_sweep='chunk_tokens' in next(iter(groups.values()))[0]
    ready_sweep=cfg.get('optimization_kind')=='ready_send'
    has_ep=any(r['configuration']=='ep2' for r in rows)
    title='Prefill chunk 对照' if chunk_sweep else '发送就绪对照' if ready_sweep else 'Prefill 合批对照'
    text=['# H100 PP2 '+title,'',
          'Qwen3.6-35B-A3B BF16，单节点 H100。'+
          ('EP2 和 PP2 各两卡；' if has_ep else '本轮只测两卡 PP2，EP2 未重测；')+
          '请求数和输入、输出长度相同，已有历史 KV/GDN 状态在 GPU；'
          '真实前缀准备和形状预热不计时，调度与同步等待计时。',
          '', '| 负载：请求数 × 已缓存历史 / 新增 Prefill | 配置 | 新 token/s | 整批完成 ms | TTFT mean / median / P99 ms |',
          '|---|---|---:|---:|---|']
    for r in rows:
        workload=f"{r['requests_per_sample']} × {r['cached_tokens_per_request']:,} / {r['new_prefill_tokens_per_request']:,}"
        label=r['configuration'].upper()+(
            f'，每次最多{r["policy"]}新 token' if chunk_sweep else
            ('，数据就绪后发送' if ready_sweep else '，最多5ms合批') if r['policy'] else '，原版')
        text.append(f"| {workload} | {label} | {r['throughput_new_tokens_s']:.1f} | "
                    f"{r['mean_cohort_duration_ms']:.2f} | {r['ttft_ms_mean']:.2f} / {r['ttft_ms_median']:.2f} / {r['ttft_ms_p99']:.2f} |")
    text+=['','输出均为 1 token，因此请求端到端延迟等于 TTFT，没有可报告的 TPOT。',
           '完整 CSV 同时包含端到端 mean、median、P99 及每次样本的整批耗时。',
           ('PP 为同进程 4096/2048/1024/8192/8192/1024/2048/4096，每种两次；EP 未重测。'
            if chunk_sweep else 'PP 为同进程 A/B/B/A、每种两次；'+
            ('EP 为两次参考。' if has_ep else 'EP 未重测。'))+
           '样本量不支持跨机器或普遍最优的结论。',
           '', '## 候选相对原版','']
    for case in cfg['optimization_cases']:
        matched={ (r['configuration'],r['policy']):r for r in rows if r['case']==case['id'] }
        a=matched[('pp2',4096 if chunk_sweep else False)]
        for mode in ((2048,1024,8192) if chunk_sweep else (True,)):
            b=matched[('pp2',mode)]
            gain=(b['throughput_new_tokens_s']/a['throughput_new_tokens_s']-1)*100
            text.append(f"- {case['id']}，策略 {mode}：新增 token 吞吐变化 {gain:+.2f}%；"
                        f"平均 TTFT {a['ttft_ms_mean']:.2f} → {b['ttft_ms_mean']:.2f} ms。")
    text+=['','[完整指标](metrics.csv) · [原始路径及 SHA256](summary.json)',
           '', '缓存命中、输出长度、输出一致性、Guard postcheck 和清理记录须结合原始运行文件查看。','']
    (out/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(dict(report=str(out/'REPORT.md'),groups=len(rows)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();analyze(a.run.resolve())
