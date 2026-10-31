"""Auditable scientific plots and prose for an existing-data component model."""
from pathlib import Path
import csv
import json
import hashlib
import importlib.util
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent
VIS=Path('/home/bxb1/.codex/visualizations/2026/10/27/01a17ce3-b560-7392-8c80-d43100c9d6b0/ep-pp-tradeoff.html')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_visual():
    template=(ROOT/'visual-template.html').read_text()
    assert template.count('__DATA_PAYLOAD__')==1
    fragment=template.replace('__DATA_PAYLOAD__',(ROOT/'data.json').read_text().strip().replace('</','<\\/'))
    assert len(fragment.encode())<1_000_000
    VIS.write_text(fragment)
    return fragment


def build():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch,Rectangle
    from matplotlib.colors import ListedColormap
    import numpy as np
    data=json.loads((ROOT/'data.json').read_text())
    with (ROOT/'component_estimates_event01.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    for item in json.loads((ROOT/'sources.json').read_text())['inputs']:
        assert sha(item['path'])==item['sha256'],item['path']
    selected={(int(r['case_id']),r['mode']):r for r in rows if r['compute_process']==r['comm_round']=='1'}
    tokens=data['tokens']
    palette=['#d6eae4','#f4d7ce','#ece3bd','#e1e4e8']
    codes=dict(PP=0,EP=1,boundary=2,missing=3)
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42})
    for mode in ('cuda_graph','eager'):
        fig,axes=plt.subplots(4,1,figsize=(19,11.5),layout='constrained',gridspec_kw={'height_ratios':[1.3,1.3,3.6,3.6]})
        for mi,(model,ax) in enumerate(zip(data['models'],axes)):
            cases=[c for c in data['cases'] if c['model']==mi]
            groupkeys=list(dict.fromkeys((c['ep'],c['low']) for c in cases))
            values=np.array([[codes[next(c for c in cases if (c['ep'],c['low'])==pair and c['n']==n)['status'][mode]['classification']] for n in tokens] for pair in groupkeys])
            ax.imshow(values,aspect='auto',vmin=-.5,vmax=3.5,cmap=ListedColormap(palette))
            for i,pair in enumerate(groupkeys):
                for j,n in enumerate(tokens):
                    c=next(c for c in cases if (c['ep'],c['low'])==pair and c['n']==n)
                    r=selected[(c['id'],mode)];status=c['status'][mode]['classification']
                    if status=='missing':text='--'
                    else:
                        margin=float(r['pp_extra_budget_ms'])
                        text=f'{"~" if status=="boundary" else status}\n{margin:+.3f}' if abs(margin)<10 else f'{status}\n{margin:+.2f}'
                    ax.text(j,i,text,ha='center',va='center',fontsize=10,color='#22252a')
                    if n>2048:
                        ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,hatch='//',edgecolor=(.3,.4,.6,.15),linewidth=0))
            names=[]
            for ep,low in groupkeys:
                alt=f'PP{ep}' if low==1 else f'PP{ep//low} x EP{low}'
                names.append(f'EP{ep} vs {alt}')
            ax.set_yticks(range(len(names)),names)
            ax.set_xticks(range(len(tokens)),[str(n) for n in tokens],rotation=40,ha='right')
            ax.set_title(f'{model["label"]} | minimum EP{model["min_ep"]}',loc='left')
            ax.axvline(tokens.index(2048)+.5,color='#55708d',linewidth=1)
        axes[-1].set_xlabel('Original unique tokens across the EP group; each row compares the same GPU budget')
        fig.suptitle(f'H100 Expert / communication trade-off | {mode}\n'
            'Cell number = saved communication - extra compute (ms), event 1; color = conservative range across all raw samples\n'
            'Ideal saturated pipeline; PP transfers, bubbles and overlap unmeasured. Hatched columns also change chunking.',fontsize=13)
        fig.legend(handles=[Patch(facecolor=palette[0],label='PP candidate'),Patch(facecolor=palette[1],label='EP candidate'),
            Patch(facecolor=palette[2],label='Boundary / variable'),Patch(facecolor=palette[3],label='Missing: no extrapolation')],loc='outside lower center',ncol=4)
        for ext in ('png','pdf'):fig.savefig(ROOT/f'tradeoff_{mode}.{ext}',dpi=160)
        plt.close(fig)

    # The editable inline fragment is thread-scoped. Its data is local and
    # self-contained; no remote calls and no dependency on a running server.
    fragment=write_visual()

    sources=json.loads((ROOT/'sources.json').read_text())
    report=['# H100：通信与 Expert 计算的布局权衡','',
        '**这是已有组件数据的成本模型，不是新测得的 PP/EP 端到端性能。** 没有启动 GPU 实验，也没有拟合、插值、平均或中位数。',
        'GLM/HY3 按用户约束最小 EP4，只比较相同 8 卡预算下 EP8 与 PP2×EP4；35B/122B 另比较 EP2/4/8 与同卡数的纯 PP2/4/8。','',
        '## 判断什么','',
        '比较同一总输入 N、同一 GPU 数量、均衡 top-8 路由、理想饱和流水下的归一化每 MoE 层工作。EP 减少的是每卡活跃专家数，不改变每 token 的 top-8，也不减少全模型总参数。',
        '定义 `预算 M = (EP 专家通信 − PP侧剩余专家通信) − (PP侧 Expert 计算 − EP Expert 计算)`，单位 ms。',
        '- M > 0：PP侧有 M ms 余量支付尚未测得的交接、流水等待等额外成本；只有这些净额外成本小于预算时，这个组件优势才兑现。',
        '- M < 0：减少专家数带来的计算收益在组件模型下更大，倾向 EP；Attention、overlap 等未测差异仍可改变结果。',
        '- 各组件用全部 30 次事件、两份独立计算测量和两轮通信分别取 min/max，按最不利组合传播。整个区间为正标 PP，为负标 EP，跨零标边界。这是观测样本的保守包络，不是置信区间。','',
        '## 一张图看区域','',
        '![组件权衡](tradeoff_cuda_graph.png)','',
        '颜色覆盖全部原始样本；格内数值只取计算测量1与通信轮次1的事件1，没有做平均。正值偏向 PP，负值偏向 EP。阴影区域中分块也发生变化，不能将全部计算差额归因于活跃专家数。','',
        '## 已有数据支持的观察','',
        '以下观察均对应 CUDA Graph；Eager 单独保存，不与 Graph 合并。','',
        '1. **GLM：在有完整组件的 N16–32768 范围，8卡 EP8 对 PP2×EP4 的包络均偏 PP2×EP4。** 它保留节点内 EP4，将专家交换留在 NVLink 域内；跨节点流水交接仍要另外支付。N8 缺 EP4/N4 通信，不填猜测值。',
        '2. **HY3：存在中间工作量的权衡区。** N32/64/128 的全部样本包络跨零；其中事件1在 N64/128 偏 EP8，N256 起包络偏 PP2×EP4。这里不能把单次负值写成稳定的 EP8 胜出。',
        '3. **35B/122B：在4卡、N64–2048，EP4 对纯 PP4 均为 EP 候选。** 更多活跃 Expert 带来的 grouped GEMM/padding 成本大于本机 NVLink 通信；这不是按模型参数量简单排序。',
        '4. **跨节点 EP8 对比纯 PP8，35B 在 N256，122B 在 N256/512 仍有 EP 候选区域。** 但同样的 8 卡预算还有 PP2×EP4；不能把赢过纯 PP8 写成赢过全部布局。',
        '5. **较大 N 更偏减少通信，但原因不止 Expert 数。** EP 按 LL 容量做多次交换；纯 PP 假定允许把一个微批合并成一次 Expert 执行，因此同时受通信次数、padding 和计算调用粒度影响。N8192 时 EP4 对纯 PP4 的模型倾向 PP，不能把该点外推为真实服务的交叉点。','',
        '## 几个绝对值例子','',
        '| 模型 | 总 token | 同卡比较 | PP多算(ms) | PP少通信(ms) | PP额外预算(ms) | 全样本判断 |',
        '|:--|--:|:--|--:|--:|--:|:--|']
    examples=[('glm47',8,'PP2×EP4',128),('glm47',8,'PP2×EP4',2048),
        ('hy3',8,'PP2×EP4',64),('hy3',8,'PP2×EP4',128),('hy3',8,'PP2×EP4',512),
        ('qwen35b',4,'PP4',128),('qwen122b',4,'PP4',128),
        ('qwen35b',8,'PP8',256),('qwen122b',8,'PP8',256)]
    for model,gpus,alt,n in examples:
        r=next(r for r in rows if r['model']==model and r['gpus']==str(gpus) and r['pp_layout']==alt and
            r['N_original']==str(n) and r['mode']=='cuda_graph' and r['compute_process']==r['comm_round']=='1')
        report.append(f'| {model} | {n} | EP{gpus} ↔ {alt} | '+
            ' | '.join(f'{float(r[k]):.6f}' for k in ('extra_compute_ms','saved_comm_ms','pp_extra_budget_ms'))+
            f' | {r["classification_all_samples"]} |')
    report+=['','## 如何对齐工作量','',
        '令 GPU 数 R=P×Q。基线为 EPR；候选为 PPP×EPQ。每个 PP 微批有 N/P 个原始 token，采用 P 个微批。理想稳态下，PP 每卡的层数减少 P 倍、每层执行 P 个微批，这两个因子抵消，因而可比较以下归一化服务时间。这个抵消不适用于单批完成延迟或未填满的流水。',
        '路由严格复用通信实验的 8 个专家组：每 token 在各组选择 `token_id mod (E_total/8)`。一个 EPq 组收到 B 个连续原始 token 时，每卡配对行数 `T=8B/q`，活跃专家 `A=(8/q)×min(B,E_total/8)`。CPU 枚举已验证活跃数、行数和均衡分布。',
        '- 基线 EPR：保留原实测 LL 全组2048-token分块；每块查 `G(8B/R,A)`，求和只是计算成本估算。通信采用同 N 的完整调用序列实测，不乘单次通信。',
        '- PP2×EP4：组输入 N/2，沿用 EP4 的全组2048-token LL 分块；通信取原实测 `C4(N/2)`，计算按这些块查表求和。',
        '- 纯 PP：一个微批 N/R 原始 token，假定可一次执行 Expert，即查 `G(8N/R,A)`。它没有专家跨卡交换，但仍有未知流水交接成本。查表 T 超过32768时标缺，不外推。实际优化的 PP 是否采用这种微批大小/内核，尚未验证。',
        '- N≤2048 的候选每次 Expert 的 T 可直接对齐；更大 N 的调用次数和 T 不同，图中已标阴影。并未将原 Expert 图的横轴 T 与通信图的原始 N 当成相同 token 口径。',
        '- 首个事件的时间差由独立组件测量相减得到，不代表这些事件在同一次端到端执行中相邻发生。范围也只覆盖已经观测的样本，不保证未来变异。','',
        '## 尚不能下的结论','',
        '没有测 PP 交接、气泡、Attention、路由/packing、初始 FP8 输入量化、完整调度、通信计算竞争或 overlap。计算图的 contiguous/masked 内核也不等于所有 PP 实现的实际路径。小 token 的 PP 候选尤其不能直接解释为 Decode 更快。',
        'GLM 是 AWQ W4A16/Humming（use_tma_bzp=False），HY3/35B 是 BF16 DeepGEMM，122B 是 FP8 DeepGEMM；通信四组均为 BF16。每个模型只与自己比较，不能按模型大小替代硬件内核因素。',
        'EP2/4 为单节点 H100 NVLink，EP8 跨两个4卡节点，使用特定 SM90 DeepEP LL native。没有8卡单节点 EP8，也没有 normal-mode Prefill 或生产级 tail 延迟数据。',
        '原 EP8 长等待样本完整保留。例如35B/N2048 Graph 两轮事件1为94.979683与2.094272 ms。用全部样本包络减少依赖单个事件，但没有定位长等待原因。','',
        '## 来源和复现','',
        '[逐配置绝对值 CSV](component_estimates_event01.csv) · [完整模型数据](data.json) · [来源与哈希](sources.json) · [路由和算术核对](validation.json)',
        f'[原 Expert 报告]({sources["inputs"][0]["path"].rsplit("/",1)[0]}/REPORT.md) · '+
        f'[原通信报告]({sources["inputs"][3]["path"].rsplit("/",1)[0]}/REPORT.md)',
        'CPU 复现：在新目录运行 analyze.py，再运行 report.py。保存的组件数据和原GPU实验未被修改。交互图还可切换两份计算测量、通信轮次、模式及全部30次事件。','']
    (ROOT/'REPORT.md').write_text('\n'.join(report))
    print(json.dumps(dict(visual=str(VIS),bytes=len(fragment.encode()),scientific_figures=4,report=str(ROOT/'REPORT.md'))))


if __name__=='__main__':
    if '--visual-only' in sys.argv:
        print(json.dumps(dict(visual=str(VIS),bytes=len(write_visual().encode()))))
    else:
        build()
