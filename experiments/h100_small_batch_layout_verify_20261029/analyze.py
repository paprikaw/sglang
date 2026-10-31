"""Recompute means from retained complete cohorts; never select a rank maximum."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/bxb1/.codex/skills/llm-experiment-guard/scripts')
from artifact_visibility import wait_for_stable_json


def read(p):return wait_for_stable_json(p,timeout_s=10,poll_s=.05,min_stable_s=.05)[0]
def desc(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    run=a.run.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=read(run/'config.json');rows=[];round_rows=[];sources=[];forward_rows=[]
    for layout in cfg['layouts']:
        for case in cfg['workloads']:
            point=run/'results'/layout['id']/case['id']
            if not (point/'completed.json').exists():continue
            completed=read(point/'completed.json');assert completed['passed']
            protocol=read(point/'protocol.json');assert protocol['P']==4096 and protocol['Q']==1 and protocol['O']==128
            rounds=[]
            for repeat in range(1,cfg['repeats']+1):
                p=point/f'formal.r{repeat}.json';data=read(p);assert data['performance_sample']
                g=data['result'];rs=g['requests'];assert len(rs)==case['requests']
                assert all(r['output_tokens']==128 and r['prompt_tokens']==4097 and r['cached_tokens']==4096 for r in rs)
                values=dict(layout=layout['id'],N=case['requests'],repeat=repeat,
                    mean_request_tpot_ms=statistics.mean(r['tpot_ms'] for r in rs),
                    mean_request_e2e_ms=statistics.mean(r['e2e_s']*1000 for r in rs),
                    mean_request_ttft_ms=statistics.mean(r['ttft_s']*1000 for r in rs),
                    cohort_makespan_ms=g['makespan_s']*1000,
                    output_tokens_per_s=g['output_tokens_per_s'])
                round_rows.append(values);rounds.append(values);sources.append(desc(p))
                assert len(g['forward_deltas'])==4
                for rank in g['forward_deltas']:
                    for shape,count in rank['counts'].items():
                        forward_rows.append(dict(layout=layout['id'],N=case['requests'],repeat=repeat,
                            replica=rank['replica'],pp_rank=rank['shape']['pp_rank'],tp_rank=rank['shape']['tp_rank'],
                            kind_and_batch=shape,count=count))
            saved=read(point/'metrics.decode.json')['metrics']
            row=dict(layout=layout['id'],N=case['requests'],rounds=len(rounds))
            for name in ['mean_request_tpot_ms','mean_request_e2e_ms','mean_request_ttft_ms','cohort_makespan_ms','output_tokens_per_s']:
                samples=[r[name] for r in rounds];value=statistics.mean(samples)
                assert abs(saved[name]['mean']-value)<1e-8,(name,saved[name]['mean'],value)
                row[name]=value;row[name+'_stdev']=statistics.stdev(samples)
            rows.append(row)
    if not rows:
        print(json.dumps(dict(completed_points=0,expected=cfg['expected_points'])));return
    write_csv(out/'mean_metrics.csv',rows);write_csv(out/'formal_rounds.csv',round_rows)
    write_csv(out/'forward_histograms.csv',forward_rows)
    lookup={(r['layout'],r['N']):r for r in rows};comparisons=[]
    for n in [c['requests'] for c in cfg['workloads']]:
        ep=lookup.get(('ep4',n))
        if ep:
            for layout in ('two_pp2','four_ep1','pp4'):
                other=lookup.get((layout,n))
                if other:
                    for metric in ('mean_request_tpot_ms','mean_request_e2e_ms','cohort_makespan_ms'):
                        comparisons.append(dict(N=n,layout=layout,metric=metric,ep4_ms=ep[metric],candidate_ms=other[metric],
                            candidate_latency_reduction_percent=(1-other[metric]/ep[metric])*100,
                            ep4_over_candidate=ep[metric]/other[metric]))
    if comparisons:write_csv(out/'comparisons.csv',comparisons)
    text=['# 35B 小 batch：同预算四张 H100 整模型验证','',
          f'完成 {len(rows)}/{cfg["expected_points"]} 个点。每点 5 轮，以下全部为算术平均值。','',
          '固定同一 H100 节点、完整 BF16 35B、GPU 热缓存 P4096、新输入 Q1、输出 128 token；N 是全组同时请求数。'
          '不启用 HiCache，不把 CPU 权重启动时间或冷 Prefill 放进 Decode 计时。','',
          '2×PP2 保留已验证的 policy9 lookahead，PP4 是原生调度；所有布局使用相同的小 token Expert GEMM 家族。'
          'PP2 的请求分组以每副本的本地序号确定，等待合批计入请求延迟。','',
          '| 全组 N | 布局 | 平均 TPOT ms/token | 平均请求 E2E ms | 平均 TTFT ms | 整批完成 ms | 输出 tok/s |',
          '|---:|---|---:|---:|---:|---:|---:|']
    for n in [c['requests'] for c in cfg['workloads']]:
        for layout in ('ep4','two_pp2','four_ep1','pp4'):
            r=lookup.get((layout,n))
            if r:text.append(f'| {n} | {layout} | {r["mean_request_tpot_ms"]:.3f} | {r["mean_request_e2e_ms"]:.2f} | {r["mean_request_ttft_ms"]:.2f} | {r["cohort_makespan_ms"]:.2f} | {r["output_tokens_per_s"]:.1f} |')
    text+=['','TPOT 先逐请求计算 `(最后 token 到达 − 首 token 到达) / 127`，再对请求和 5 轮取平均；'
           '平均请求 E2E 包含 TTFT，整批完成时间是不同的指标。各指标分别判断反转，不能混用。','',
           '四卡布局按顺序测量，每点重复 5 轮；尚未做布局交替复测。自然生成的路由允许随数值与 batch 变化，'
           '因此这些结果验证部署效果，不能单独归因于通信或专家数量。','',
           '实际 rank／stage batch 与 Graph 资格计数保存在 [forward_histograms.csv](forward_histograms.csv)。'
           '这些是每步参与计算的请求数；它们与微基准的 token–expert assignment 行数不同。','',
           '原始样本：[formal_rounds.csv](formal_rounds.csv)；汇总：[mean_metrics.csv](mean_metrics.csv)。',
           f'原始运行目录：{run}','']
    (out/'REPORT.md').write_text('\n'.join(text))
    manifest=dict(experiment_id='qwen35b_h100_small_batch_layout_verify_20261029',run=str(run),
        completed_points=len(rows),expected_points=cfg['expected_points'],new_gpu_measurement=True,
        all_rounds_retained=True,statistic='arithmetic mean of complete formal cohorts; per-request means are reported separately',
        source_manifest=desc(run/'launch_manifest.json'),sources=sources)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(completed_points=len(rows),expected=cfg['expected_points'],output=str(out),means=rows)))

if __name__=='__main__':main()
