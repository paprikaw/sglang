"""Derive bounded findings from the accepted four-model surface; CPU only."""
import argparse,csv,json,statistics,sys
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
LABELS={'glm47':'GLM-4.7 AWQ','hy3':'HY3 BF16','qwen35b':'Qwen3.6-35B BF16','qwen122b':'Qwen3.5-122B FP8'}

def main(run):
    active=run.parents[1];sys.path.insert(0,str(active))
    from common import save,sha
    report=active/'report'
    result=json.loads((run/'profile/result.json').read_text())
    assert result['passed'] is True
    cells={}
    for row in result['rows']:
        if row['mode']=='cuda_graph':cells.setdefault((row['model'],row['T'],row['E']),[]).append(row)
    def cell(model,t,e):
        xs=cells[model,t,e];assert len(xs)==2
        pads={x.get('actual_padded_rows') for x in xs};assert len(pads)==1
        return statistics.mean(x['timing']['median_ms'] for x in xs),pads.pop()
    derived=[]
    for model in LABELS:
        maxe=max(e for m,t,e in cells if m==model)
        for t in [128,512,2048,8192,16384,32768]:
            for high in sorted({32,maxe}):
                if high>t:continue
                a,pa=cell(model,t,2);b,pb=cell(model,t,high)
                derived.append(dict(model=model,T_assignment_rows=t,E_low=2,E_high=high,
                    low_ms=a,high_ms=b,latency_ratio=b/a,padded_low=pa,padded_high=pb,
                    padding_row_ratio=pb/pa if pa is not None else None))
    with (report/'comparisons.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(derived[0]));w.writeheader();w.writerows(derived)
    lines=['## 直接回答：小模型是否也会随 token 增大而缩小专家数差距？','',
        '以下数值均从这次通过验证的 H100 数据重算；T 是单卡接收的 token–expert 配对行数。固定 E2→E32 的共同范围：','',
        '| 模型 | T128 | T8192 | T32768 |','|---|---:|---:|---:|']
    for model,label in LABELS.items():
        ratios=[cell(model,t,32)[0]/cell(model,t,2)[0] for t in [128,8192,32768]]
        lines.append('| '+label+' | '+' | '.join(f'{r:.2f}×' for r in ratios)+' |')
    lines+=['','数值是 E32/E2 耗时比，接近 1 表示这个专家数区间的耗时接近。应逐模型读取该表，不把曲线靠拢预设为所有后端的共同定律。','',
        '对 Qwen 的完整 E256 范围，需要保留另一个结论：','']
    for model in ['qwen35b','qwen122b']:
        a,_=cell(model,32768,2);b,_=cell(model,32768,256);c,_=cell(model,32768,128)
        lines.append(f"- {LABELS[model]}：T32768 时 E256/E2 仍为 **{b/a:.2f}×**；E256/E128 为 **{b/c:.2f}×**。")
    lines+=['','因此，适合写入 benchmark 的问题是“差距在什么 T/E 与后端配置下缩小到多少”，不能只给出“Prefill 大了就一样快”的无条件判断。','',
        '### 补齐行数与有效计算量','',
        '固定 T 时，非补齐的有效 GEMM FLOPs 相同；DeepGEMM 实际执行的补齐行数则可能随 E 改变。例如 T128：','']
    for model in ['hy3','qwen35b','qwen122b']:
        _,pa=cell(model,128,2);_,pb=cell(model,128,32)
        assert pa==256 and pb==4096
        lines.append(f'- {LABELS[model]}：E2 为 {pa} 行，E32 为 {pb} 行，实际补齐行数相差 {pb/pa:.0f} 倍。')
    lines+=['','到 T32768，E2、E32、E128、E256 均无需这种额外的 128 行补齐（在各模型已测 E 范围内）。此时残留的性能差异不能归因于额外补齐行数；它与更多专家权重、更小的单专家 GEMM 和 kernel 调度粒度等因素一致，但本实验没有把这些原因逐项隔离。',
        '', 'GLM 使用另一条 grouped masked AWQ 路径，不适用上面的 DeepGEMM 补齐行数解释。',
        '', '### 从组件实验到 EP/PP 的边界','',
        '这些数据可以检验“Expert 服务时间不能只按有效 FLOPs 估算”，并给出单层后端的 T/E 依赖关系。将它们用于 EP/PP 选型时，还需测实际路由偏斜、各微批重复激活的专家、dispatch/combine、PP 调度与通信暴露；当前图不能独立证明某种整模型并行布局更快。','']
    addition='\n'.join(lines)
    (report/'FINDINGS.md').write_text(addition)
    p=report/'REPORT.md';text=p.read_text();anchor='## 二维图\n'
    assert text.count(anchor)==1
    p.write_text(text.replace(anchor,addition+'\n'+anchor,1))
    sources=json.loads((report/'sources.json').read_text())
    sources['findings_generator']={'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__))}
    sources['artifacts']={p.name:sha(p) for p in report.iterdir() if p.is_file() and p.name!='sources.json'}
    save(report/'sources.json',sources)
    print(addition)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
