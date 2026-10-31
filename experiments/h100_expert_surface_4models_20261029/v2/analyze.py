"""Recompute the four H100 Expert surfaces and archive source-linked figures."""
import argparse,collections,csv,datetime,hashlib,json,math,statistics,sys
from pathlib import Path
from common import ROOT,MODELS,PLAN,GUARD_SCRIPTS,save,sha

LABELS={'glm47':'GLM-4.7 AWQ W4A16','hy3':'HY3 BF16','qwen35b':'Qwen3.6-35B-A3B BF16','qwen122b':'Qwen3.5-122B-A10B FP8'}
PAIRS={'glm47':(20,40),'hy3':(24,48),'qwen35b':(128,256),'qwen122b':(128,256)}
def table(headers,rows):return '\n'.join(['| '+' | '.join(map(str,headers))+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def main(run):
    sys.path.insert(0,str(GUARD_SCRIPTS))
    from artifact_visibility import wait_for_stable_json
    result,visibility=wait_for_stable_json(run/'profile/result.json',timeout_s=15)
    if result.get('passed') is not True:raise RuntimeError('The complete four-model experiment has not passed')
    if not all(x.get('passed') is True for x in result['cleanup']):raise RuntimeError('GPU cleanup is unverified')
    report=ROOT/'report';report.mkdir(exist_ok=True)
    save(report/'result_visibility.json',visibility)
    raw=result['rows'];groups=collections.defaultdict(list)
    flat=[]
    for row in raw:
        xs=row['raw_ms'];assert len(xs)==30 and all(math.isfinite(x) and x>0 for x in xs)
        median=statistics.median(xs);assert math.isclose(median,row['timing']['median_ms'],rel_tol=1e-12)
        key=(row['model'],row['T'],row['E'],row['mode']);groups[key].append(row)
        flat.append(dict(model=row['model'],worker=row['worker'],node=row['node'],gpu=row['physical_gpu'],backend=row['backend'],
          mode=row['mode'],T_assignment_rows=row['T'],E_active=row['E'],resident=row['resident'],n=len(xs),median_ms=median,mean_ms=statistics.mean(xs),
          block_spread_percent=(max(row['block_median_ms'])/min(row['block_median_ms'])-1)*100,
          padded_rows=row.get('actual_padded_rows'),capacity_per_expert=row.get('capacity_per_expert'),gemm_flops=row['gemm_flops'],
          effective_tflops=row['effective_tflops'],raw_ms=json.dumps(xs)))
    agg={}
    for key,rows in groups.items():
        assert len(rows)==2 and len({r['node'] for r in rows})==2
        medians=[r['timing']['median_ms'] for r in rows]
        agg[key]=dict(median_mean_ms=statistics.mean(medians),replica_min_ms=min(medians),replica_max_ms=max(medians),
            replica_spread_percent=(max(medians)/min(medians)-1)*100)
    bymodel=collections.defaultdict(list)
    for w in result['workers']:bymodel[w['model']].append(w)
    for model,ws in bymodel.items():
        assert len(ws)==2 and ws[0]['weight_digest']==ws[1]['weight_digest'],'Independent replicas loaded different weights'
    def write_csv(path,rows):
        with path.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_csv(report/'full_metrics.csv',flat)
    aggregate=[dict(model=k[0],T_assignment_rows=k[1],E_active=k[2],mode=k[3],**v) for k,v in sorted(agg.items())]
    write_csv(report/'aggregate_metrics.csv',aggregate)
    def ratio(model,t,e1,e2,mode='cuda_graph'):
        a=agg[(model,t,e1,mode)]['median_mean_ms'];b=agg[(model,t,e2,mode)]['median_mean_ms'];return b/a
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm,Normalize
    import numpy as np
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    for normalized in [False,True]:
        fig,axes=plt.subplots(2,2,figsize=(16,10),layout='constrained')
        for ax,(model,m) in zip(axes.flat,MODELS.items()):
            es=m['experts'];ts=PLAN['tokens'];z=np.full((len(es),len(ts)),np.nan)
            for yi,e in enumerate(es):
                for xi,t in enumerate(ts):
                    if e>t:continue
                    value=agg[(model,t,e,'cuda_graph')]['median_mean_ms']
                    z[yi,xi]=value/agg[(model,t,1,'cuda_graph')]['median_mean_ms'] if normalized else value
            values=z[np.isfinite(z)]
            norm=LogNorm(vmin=max(1,values.min()),vmax=max(1.01,values.max())) if normalized else LogNorm(vmin=values.min(),vmax=values.max())
            im=ax.imshow(z,origin='lower',aspect='auto',cmap='viridis',norm=norm)
            ax.set_xticks(range(len(ts)),[str(t) if t<1024 else f'{t//1024}k' for t in ts]);ax.set_yticks(range(len(es)),es)
            ax.set_xlabel('Received token-expert assignment rows T');ax.set_ylabel('Active experts E')
            ax.set_title(LABELS[model]+'\n'+m['backend'],fontsize=11)
            for yi in range(len(es)):
                for xi in range(len(ts)):
                    v=z[yi,xi]
                    if np.isfinite(v):ax.text(xi,yi,f'{v:.1f}' if normalized else f'{v:.3f}' if v<1 else f'{v:.2f}',ha='center',va='center',fontsize=6.4,color='white' if norm(v)<.65 else 'black')
            fig.colorbar(im,ax=ax,shrink=.8,label='Latency / E=1 latency' if normalized else 'Expert compute core (ms)')
        fig.suptitle('H100 Expert T-by-E surfaces — warm CUDA Graph\nMean of two independent process medians; 30 CUDA-event samples per process',fontsize=14)
        stem='expert_surface_relative' if normalized else 'expert_surface_ms'
        fig.savefig(report/(stem+'.png'),dpi=180);fig.savefig(report/(stem+'.pdf'));plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for ax,(model,m) in zip(axes.flat,MODELS.items()):
        for e in m['experts']:
            ts=[t for t in PLAN['tokens'] if t>=e];ys=[agg[(model,t,e,'cuda_graph')]['median_mean_ms'] for t in ts]
            ax.plot(ts,ys,marker='.',label=f'E={e}')
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_title(LABELS[model]);ax.set_xlabel('Assignment rows T');ax.set_ylabel('Expert compute core (ms)')
        ax.grid(alpha=.25);ax.legend(ncol=3,fontsize=8)
    fig.savefig(report/'expert_curves.png',dpi=180);fig.savefig(report/'expert_curves.pdf');plt.close(fig)
    anchor_ts=[128,512,2048,8192,16384,32768]
    common_rows=[[LABELS[model]]+[f'{ratio(model,t,2,32):.2f}×' for t in anchor_ts] for model in MODELS]
    pair_rows=[[LABELS[model],f'{a} → {b}']+[f'{ratio(model,t,a,b):.2f}×' if t>=b else '—' for t in anchor_ts] for model,(a,b) in PAIRS.items()]
    quality=[]
    for model,ws in bymodel.items():
        ns=[n for n in result['numerical'] if n['worker'].startswith(model+'_')]
        spreads=[v['replica_spread_percent'] for k,v in agg.items() if k[0]==model and k[3]=='cuda_graph']
        quality.append([LABELS[model],f"{MODELS[model]['resident']}",f"{ws[0]['selected_weight_bytes']/2**30:.3f}",
          f"{max(n['nrms'] for n in ns)*100:.2f}%",f'{statistics.median(spreads):.1f}%',f'{max(spreads):.1f}%'])
    now=datetime.datetime.now().astimezone().isoformat()
    text=f'''# H100：四个模型的 Expert token × 活跃专家数二维测量

生成时间：{now}。

## 口径

- T 是单卡实际接收的 token–expert 配对行数，取 8 到 32768 的 2 次幂。它不是请求 batch、全局 Prefill token 总数或缓存上下文长度。同一个原始 token 路由到八个专家会形成八行，此处直接控制配对行。
- E 是有非零输入的真实 routed experts 数。路由均衡，每个活跃专家处理 floor(T/E) 或 ceil(T/E) 行；E 大于 T 的格子不测。每个模型始终加载相同的一组 resident 专家。
- 测真实一层的 W13 GEMM、SiLU/multiply、W2 GEMM；FP8 路径包含两次 GEMM 之间的激活量化。初始输入量化、输入排列/对齐、权重读取/转换、JIT、router、Attention、shared expert、通信和模型加载都不计时。该结果没有 TTFT、TPOT 或完整模型吞吐的含义。
- 每个模型在两台已预约 H100 节点上各运行一个独立单 GPU 进程，八张卡并行。每点每种模式 5 次准备预热，3 个旋转/逆序计时块、每块 3 次即时预热及 10 次 CUDA event，共 30 个事件样本。事件样本不是 30 次独立进程。
- eager 与 CUDA Graph 数据分开保存。图表采用两份独立进程的 Graph 中位数的算术平均，不把 60 个事件视为 60 次独立实验。
- GLM-4.7 使用 AWQ/Humming grouped masked；T≤8192 时保持原 A100 实验的 8192 行固定槽位，T 更大时槽位等于 T，同一 T 的所有 E 共用相同槽位大小。
- HY3 和 35B 使用 H100 原生 DeepGEMM BF16 grouped contiguous；122B 使用 FP8 W8A8 grouped contiguous。预先按专家分组并做 128 行对齐；对齐产生的额外 GEMM 行已保存。它们不包含完整 EP/PP 的本地适配、masked/contiguous 切换或调度差异。
- 这是单层组件实验，未修改完整模型层数，未运行完整模型，未创建、替换或驱逐 CPU 权重/KV 缓存。

## 二维图

![Expert latency](expert_surface_ms.png)

![Relative to E1](expert_surface_relative.png)

![Expert curves](expert_curves.png)

## 所有模型的共同专家数对照

下面固定 T，报告 **E32 耗时 / E2 耗时**。越接近 1，表示这个专家数范围内耗时越接近；小于 1 时应保留其方向，不能解释为普遍等价。

{table(['模型']+[f'T={t}' for t in anchor_ts],common_rows)}

## 与部署专家数有关的补充对照

下面只是均衡路由下的局部活跃专家数对照，并不代表已经运行 EP/PP。Qwen 的 128/256 是对应 resident 数量的一种满覆盖情形，实际 Decode 通常不会全部激活。

{table(['模型','活跃专家数']+[f'T={t}' for t in anchor_ts],pair_rows)}

## 验证与波动

两份独立进程加载的全部所选权重逐张量 SHA256 汇总一致。每个 T/E 在计时外，对第一个和最后一个活跃专家的最多两行输出进行独立 dense reference 检查。BF16/GLM 使用相对 L2≤2%，FP8 使用≤12%，并同时检查最大绝对误差与有限性；下表给出实际最大误差。

{table(['模型','resident experts','所选权重 GiB','最大参考相对 L2','跨进程差距中位数','跨进程最大差距'],quality)}

只有两份独立进程；不提供稳定置信区间或服务尾延迟结论。保留每点各进程的中位数、三块中位数和全部事件样本，以识别编译后的时钟、缓存和运行波动。

## 如何解释

固定 T 时，有效两次 GEMM 的计算量为 6×hidden_size×intermediate_size×T，不随 E 额外相乘。E 增加会改变活跃权重范围、每个专家矩阵的行数、对齐/填充与 kernel 执行粒度。因此需要逐模型、逐后端检查曲线是否靠拢，不能预设一定收敛。

原 A100 HY3 使用 Triton，这次 HY3 使用 DeepGEMM；大模型之间和不同精度之间的绝对耗时不可视为纯模型规模对照。四个模型的二维图也不能直接代替 EP/PP 端到端比较。

## 可追溯来源

- [冻结配置]({run/'config.json'})、[计时计划]({run/'measurement_plan.json'})、[逐张量权重选择]({run/'selection.json'})、[节点/GPU 映射]({run/'workers.json'})。
- [完整结果]({run/'profile/result.json'})、[Guard 规格与源版本]({run/'dashboard_spec.json'})。
- [逐进程计时 CSV](full_metrics.csv)、[两份独立进程汇总 CSV](aggregate_metrics.csv)、[来源校验](sources.json)。
- [绝对延迟 PDF](expert_surface_ms.pdf)、[相对延迟 PDF](expert_surface_relative.pdf)、[曲线 PDF](expert_curves.pdf)。
'''
    (report/'REPORT.md').write_text(text)
    sources=dict(created_at=now,run=str(run),result=dict(path=str(run/'profile/result.json'),sha256=sha(run/'profile/result.json')),
        frozen_spec=dict(path=str(run/'dashboard_spec.json'),sha256=sha(run/'dashboard_spec.json')),
        worker_results=[dict(path=str(run/'workers'/w['worker']/'result.json'),sha256=sha(run/'workers'/w['worker']/'result.json')) for w in result['workers']],
        artifacts={p.name:sha(p) for p in report.iterdir() if p.is_file() and p.name!='sources.json'})
    save(report/'sources.json',sources)
    print(table(['model']+[str(t) for t in anchor_ts],common_rows))
    print(table(['model','pair']+[str(t) for t in anchor_ts],pair_rows))
    print('REPORT',report/'REPORT.md')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
