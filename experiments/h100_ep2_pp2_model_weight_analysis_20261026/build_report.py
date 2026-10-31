#!/usr/bin/env python3
"""Offline comparison of saved H100 results; never imports torch or starts GPUs."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path('/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling')
OUT = Path(__file__).resolve().parent
SMALL = ROOT / 'experiments/ep2_pp2_qwen35b_h100_20261025/runs/h100_qwen35b_ep2_pp2_r3/analysis'
LARGE = ROOT / 'experiments/ep2_pp2_qwen_h100_20261025/runs/h100_ep2_pp2_r7/analysis'
LARGE_PREFILL = ROOT / 'experiments/ep2_pp2_qwen_h100_20261025/runs/h100_prefill_components_r1/analysis'
MODELS = [
    dict(label='35B BF16', name='Qwen3.6-35B-A3B', layers=40, precision='BF16',
         metrics=SMALL/'full_metrics.csv', prefill=SMALL/'prefill_components.csv',
         decode=SMALL/'decode_components.csv', evidence=SMALL/'validated_metrics.json'),
    dict(label='122B FP8', name='Qwen3.5-122B-A10B-FP8', layers=48, precision='FP8 W8A8',
         metrics=LARGE/'full_metrics.csv', prefill=LARGE_PREFILL/'comparison.csv',
         decode=LARGE/'component_followup/comparison.csv', evidence=LARGE/'validated_metrics.json'),
]


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])


def f(row, key):
    return float(row[key])


def plot(comparisons, prefill):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(11.4,4.4))
    ax=axes[0]
    labels=['35B BF16\nPrefill','122B FP8\nPrefill','35B BF16\nDecode','122B FP8\nDecode']
    order=[comparisons[0],comparisons[2],comparisons[1],comparisons[3]]
    vals=[100+r['pp_rate_change_pct'] for r in order]
    bars=ax.bar(range(4), vals,color=['#208676','#208676','#8660bd','#8660bd'],width=.65)
    ax.axhline(100,color='black',lw=1,ls='--',label='EP2 = 100% for each cell')
    ax.set_xticks(range(4),labels)
    ax.set_ylim(0,130)
    ax.set_ylabel('PP2 / EP2 throughput (%)')
    ax.set_title('Identical request composition')
    for b,v,r in zip(bars,vals,order):
        ax.text(b.get_x()+b.get_width()/2,v+2,f"{r['pp_rate_change_pct']:+.1f}%",ha='center')
    ax.legend(loc='lower left',fontsize=8)
    ax=axes[1]
    for offset,key,label,color in [(-.18,'pp_extra_expert','Extra routed Expert in PP','#c97738'),
                                 (.18,'ep_dispatch_combine','Dispatch + Combine in EP','#208676')]:
        x=[i+offset for i in range(2)]
        bars=ax.bar(x,[r[key] for r in prefill],width=.34,label=label,color=color)
        for b,r in zip(bars,prefill):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+3,f"{r[key]:.1f}",ha='center')
    ax.set_xticks([0,1],['35B BF16','122B FP8'])
    ax.set_ylim(0,170)
    ax.set_ylabel('Cumulative kernel-ms / GPU / Prefill cohort')
    ax.set_title('Prefill: larger Expert penalty')
    ax.legend(fontsize=8,loc='upper right')
    fig.text(.5,.015,'Historical means; not size-only or precision-controlled. Kernel sums are not additive wall latency.',
             ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.06,1,1])
    for suffix in ('png','svg'):
        fig.savefig(OUT/f'model_comparison.{suffix}',dpi=180)
    plt.close(fig)


def absolute_components():
    """Keep all measured segments; unknown shared Decode work is not zero."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    for model in MODELS:
        for phase in ('prefill', 'decode'):
            for source in read_csv(model[phase]):
                if source['case'] != 'moderate':
                    continue
                def get(*names):
                    return next((float(source[n]) for n in names if n in source), None)
                rows.append(dict(model=model['name'], phase=phase, arm=source['arm'],
                    stage=source['stage'],
                    unit='kernel-ms/card/global-prefill-cohort' if phase=='prefill' else 'kernel-ms/card/actual-decode-graph',
                    attention_ms=get('attention_ms', 'attention_core_ms'),
                    routed_expert_ms=get('routed_expert_ms', 'routed_gemm_ms'),
                    shared_expert_ms=get('shared_expert_ms'),
                    dispatch_ms=get('dispatch_ms'), combine_ms=get('combine_ms'),
                    other_kernel_ms=get('other_kernel_ms', 'other_ms', 'unclassified_kernel_ms'),
                    nccl_residence_ms=get('nccl_residence_ms', 'nccl_ms', 'nccl_in_graph_ms'),
                    graph_span_ms=get('graph_span_ms')))
    write_csv(OUT/'absolute_components.csv', rows)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5))
    for index, model in enumerate(MODELS):
        for col, phase in enumerate(('prefill', 'decode')):
            ax = axes[index, col]
            selected = [r for r in rows if r['model']==model['name'] and r['phase']==phase]
            selected.sort(key=lambda r:(r['arm']!='ep2',r['stage']))
            keys = ['attention_ms','routed_expert_ms']
            labels = ['Attention','Routed Expert']
            if phase=='prefill':
                keys += ['shared_expert_ms']; labels += ['Shared Expert']
            keys += ['dispatch_ms','combine_ms','other_kernel_ms','nccl_residence_ms']
            labels += ['Dispatch','Combine','Other kernels','NCCL residency']
            for offset, row, color in zip((-.25,0,.25), selected, ('#178f7d','#8762c2','#cf9142')):
                values = [row[k] for k in keys]
                name = 'EP2 (GPU mean)' if row['arm']=='ep2' else 'PP2 stage '+row['stage']
                bars = ax.bar([i+offset for i in range(len(keys))], values, .23, label=name, color=color)
                for bar, value in zip(bars, values):
                    ax.annotate(f'{value:.2f}' if phase=='prefill' else f'{value:.3f}',
                        (bar.get_x()+bar.get_width()/2, value), xytext=(0,4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7.5, rotation=90)
            ax.set_xticks(range(len(keys)), labels, rotation=18, ha='right')
            ax.set_title(model['label']+' — '+phase.capitalize())
            ax.set_ylabel('Cumulative GPU kernel ms / card / '+('whole B8 cohort' if phase=='prefill' else 'actual Graph'))
            ax.set_ylim(0, max(r[k] for r in selected for k in keys)*1.40)
            ax.legend(fontsize=8, loc='upper right')
            if phase=='decode':
                spans=' / '.join(f"{r['graph_span_ms']:.3f}" for r in selected)
                ax.text(.02,.98,'Graph span (EP / PP0 / PP1): '+spans+' ms',
                    transform=ax.transAxes, fontsize=8, va='top')
    fig.text(.5,.015,'All segment labels are absolute milliseconds. Kernel sums may overlap; PP stages are separate, not summed TPOT.\n'
        'Decode Attention / Expert scope is narrower than Prefill. NCCL residency includes waiting; zero in Graph does not mean no PP transfer.',
        ha='center',fontsize=9,color='#444')
    fig.tight_layout(rect=[0,.075,1,1])
    for suffix in ('png','svg'):
        fig.savefig(OUT/f'absolute_components.{suffix}',dpi=180)
    plt.close(fig)
    rendered = {}
    for phase in ('prefill','decode'):
        body=[]
        for r in rows:
            if r['phase']!=phase: continue
            digits=2 if phase=='prefill' else 3
            body.append([r['model'],'EP2 卡均值' if r['arm']=='ep2' else 'PP2 stage '+r['stage']]+
                ['未单独分类' if r[k] is None else f'{r[k]:.{digits}f}' for k in
                 ('attention_ms','routed_expert_ms','shared_expert_ms','dispatch_ms','combine_ms','other_kernel_ms','nccl_residence_ms')]+
                ([] if phase=='prefill' else [f"{r['graph_span_ms']:.3f}"]))
        rendered[phase] = table(['模型','配置／阶段','Attention','路由 Expert','共享 Expert','Dispatch','Combine','其他内核','NCCL 驻留']+
            ([] if phase=='prefill' else ['Graph 总跨度']),body)
    return rows, rendered


def main():
    metrics, prefill, decode, comparisons, sources = [], [], [], [], []
    checked = []
    for model in MODELS:
        cells = read_csv(model['metrics'])
        metrics.extend(dict(model=model['name'], precision=model['precision'], **r) for r in cells)
        evidence = json.loads(model['evidence'].read_text())
        assert evidence['passed'] is True
        for field in ('metrics', 'prefill', 'decode', 'evidence'):
            p = model[field]
            sources.append(dict(model=model['name'], role=field, path=str(p),
                                sha256=digest(p), bytes=p.stat().st_size))
        for phase in ('prefill', 'decode'):
            selected = [r for r in cells if r['case']=='moderate' and r['phase']==phase]
            assert len(selected)==2
            ep = next(r for r in selected if r['arm']=='ep2')
            pp = next(r for r in selected if r['arm']=='pp2')
            for k in ('requests', 'cached_context_per_request', 'fresh_input_per_request', 'output_per_request'):
                assert ep[k]==pp[k]
            assert ep['requests']=='8'
            assert ep['cached_context_per_request']==('0' if phase=='prefill' else '4096')
            assert ep['fresh_input_per_request']==('4096' if phase=='prefill' else '1')
            assert ep['output_per_request']==('1' if phase=='prefill' else '128')
            comparisons.append(dict(model=model['name'], phase=phase,
                ep_rate=f(ep,'tokens_per_s'), pp_rate=f(pp,'tokens_per_s'),
                pp_rate_change_pct=100*(f(pp,'tokens_per_s')/f(ep,'tokens_per_s')-1),
                ep_ttft_s=f(ep,'ttft_s_mean'), pp_ttft_s=f(pp,'ttft_s_mean'),
                ep_tpot_ms=ep['tpot_ms_mean'], pp_tpot_ms=pp['tpot_ms_mean'],
                ep_e2e_s=f(ep,'e2e_s_mean'), pp_e2e_s=f(pp,'e2e_s_mean'),
                ep_rate_min=f(ep,'tokens_per_s_min'), ep_rate_max=f(ep,'tokens_per_s_max'),
                pp_rate_min=f(pp,'tokens_per_s_min'), pp_rate_max=f(pp,'tokens_per_s_max'),
                ep_ready_send=ep['ready_send'], pp_ready_send=pp['ready_send']))

        pc = [r for r in read_csv(model['prefill']) if r['case']=='moderate']
        ep = next(r for r in pc if r['arm']=='ep2')
        pp = sorted([r for r in pc if r['arm']=='pp2'], key=lambda r:r['stage'])
        assert len(pp)==2 and [r['stage'] for r in pp]==['0','1']
        assert all(f(r,'dispatch_ms')==f(r,'combine_ms')==0 for r in pp)
        extra = statistics.mean(f(r,'routed_expert_ms') for r in pp)-f(ep,'routed_expert_ms')
        comm = f(ep,'dispatch_ms')+f(ep,'combine_ms')
        prefill.append(dict(model=model['name'], unit='kernel-ms/card/global-prefill-cohort',
            ep_attention=f(ep,'attention_ms'),
            pp_attention_stage0=f(pp[0],'attention_ms'), pp_attention_stage1=f(pp[1],'attention_ms'),
            ep_routed_expert=f(ep,'routed_expert_ms'),
            pp_routed_expert_stage0=f(pp[0],'routed_expert_ms'),
            pp_routed_expert_stage1=f(pp[1],'routed_expert_ms'),
            pp_routed_expert_stage_mean=statistics.mean(f(r,'routed_expert_ms') for r in pp),
            pp_extra_expert=extra, ep_dispatch=f(ep,'dispatch_ms'), ep_combine=f(ep,'combine_ms'),
            ep_dispatch_combine=comm, extra_expert_over_dc_pct=100*extra/comm,
            pp_nccl_stage0=f(pp[0],'nccl_residence_ms'), pp_nccl_stage1=f(pp[1],'nccl_residence_ms')))

        dc = [r for r in read_csv(model['decode']) if r['case']=='moderate']
        ep = next(r for r in dc if r['arm']=='ep2')
        pp = sorted([r for r in dc if r['arm']=='pp2'], key=lambda r:r['stage'])
        assert len(pp)==2
        attn_key = 'attention_ms' if model['layers']==40 else 'attention_core_ms'
        pp_expert_per_layer = sum(f(r,'routed_gemm_ms') for r in pp)/model['layers']
        ep_expert_per_layer = f(ep,'routed_gemm_ms')/model['layers']
        decode.append(dict(model=model['name'], unit='kernel-ms/actual-decode-graph',
            ep_layers=model['layers'], pp_layers_per_stage=model['layers']//2,
            ep_attention_core=f(ep,attn_key),
            pp_attention_core_stage0=f(pp[0],attn_key), pp_attention_core_stage1=f(pp[1],attn_key),
            ep_routed_gemm=f(ep,'routed_gemm_ms'),
            pp_routed_gemm_stage0=f(pp[0],'routed_gemm_ms'), pp_routed_gemm_stage1=f(pp[1],'routed_gemm_ms'),
            ep_dispatch=f(ep,'dispatch_ms'), ep_combine=f(ep,'combine_ms'),
            ep_gemm_ms_per_layer=ep_expert_per_layer, pp_gemm_ms_per_layer=pp_expert_per_layer,
            pp_gemm_per_layer_change_pct=100*(pp_expert_per_layer/ep_expert_per_layer-1),
            ep_graph_span=f(ep,'graph_span_ms'), pp_graph_span_stage0=f(pp[0],'graph_span_ms'),
            pp_graph_span_stage1=f(pp[1],'graph_span_ms')))
        checked.append(dict(model=model['name'], source_metric_validation=True,
            original_guard_pass=evidence.get('guard_original_passed'),
            logical_capacities=evidence.get('logical_capacities')))

    write_csv(OUT/'full_metrics.csv', metrics)
    write_csv(OUT/'matched_workloads.csv', comparisons)
    write_csv(OUT/'prefill_components.csv', prefill)
    write_csv(OUT/'decode_components.csv', decode)
    absolute_rows, absolute_tables = absolute_components()
    (OUT/'source_manifest.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')

    e2e_rows = []
    for c in comparisons:
        e2e_rows.append([c['model'], c['phase'],
            f"{c['ep_rate']:,.2f} / {c['pp_rate']:,.2f}", f"{c['pp_rate_change_pct']:+.2f}%",
            f"{c['ep_ttft_s']:.3f} / {c['pp_ttft_s']:.3f}",
            '—' if not c['ep_tpot_ms'] else f"{float(c['ep_tpot_ms']):.3f} / {float(c['pp_tpot_ms']):.3f}",
            f"{c['ep_e2e_s']:.3f} / {c['pp_e2e_s']:.3f}"])
    prefill_rows = [[r['model'],f"{r['ep_attention']:.2f}",
        f"{(r['pp_attention_stage0']+r['pp_attention_stage1'])/2:.2f}",
        f"{r['ep_routed_expert']:.2f}",f"{r['pp_routed_expert_stage_mean']:.2f}",
        f"+{r['pp_extra_expert']:.2f}", f"{r['ep_dispatch_combine']:.2f}",
        f"{r['extra_expert_over_dc_pct']:.1f}%"] for r in prefill]
    decode_rows = [[r['model'],
        f"{r['ep_attention_core']:.3f}",
        f"{r['pp_attention_core_stage0']:.3f} / {r['pp_attention_core_stage1']:.3f}",
        f"{r['ep_routed_gemm']:.3f}",
        f"{r['pp_routed_gemm_stage0']:.3f} / {r['pp_routed_gemm_stage1']:.3f}",
        f"{r['ep_dispatch']+r['ep_combine']:.3f}",
        f"{r['ep_gemm_ms_per_layer']:.4f} / {r['pp_gemm_ms_per_layer']:.4f}"] for r in decode]

    report = '''# H100 双卡：较重模型是否更适合 EP，而较轻模型更适合 PP？

**结论：冷 Prefill 的均值和分项支持这个方向的线索；Decode 不支持把它推广为模型大小规律。** 这是已保存实验的离线复核，没有申请 GPU、启动引擎或新增性能样本。

## 1. 比较的模型、布局与请求

小模型为 **Qwen3.6-35B-A3B BF16，40 层**；大模型为 **Qwen3.5-122B-A10B-FP8，48 层，FP8 W8A8**。两者都是每层 256 个路由专家、每个 token 激活 8 个，均混合全 Attention 和线性 Attention；KV 使用 BF16。

每种配置都用两张 H100 80GB，来自历史节点 178 的两对 NVLink NV6：EP2 用 GPU0/1，PP2 用 GPU2/3，原正式比较在同一节点并行。**没有跨节点通信，也不是 PCIe 与 RDMA 的对照。**

- **EP2：** 两个 Attention 数据并行副本，每张卡的 Attention 不做 TP；专家跨两卡分配，每卡每层 128 个专家，执行全部层。
- **PP2：** 两个等大流水槽位；每个阶段负责一半层，每层全部 256 个专家在本卡，Attention 不做 TP。没有层内跨卡 Dispatch／Combine，但仍有阶段传输、反馈和调度。
- **相同冷 Prefill：** 整批 8 个请求，每条没有已缓存前缀，新增输入 4096 token，只生成 1 token；整批新增输入共 32,768 token。
- **相同 Decode：** 整批 8 个请求，每条已有 4096 token 的真实模型前缀驻留 GPU，本轮新增输入 1 token，再生成 128 token。PP 分两个 4 请求槽位。没有 CPU HiCache 换入、合成 KV 或 MTP。

正式指标每格 3 轮；上下文准备、形状预热和独立 profile 不计时。Prefill 吞吐只计新增输入，Decode 只计生成输出，分母是整批完成时间。TTFT 是提交到首 token；TPOT 是首 token 后到末 token 的时间除以之后的 127 个 token；端到端是提交到末 token。下表延迟均为请求均值。

## 2. 相同请求条件下的端到端结果

所有成对数字按 **EP2 / PP2** 排列。吞吐越高越好，延迟越低越好；Prefill 为输入 token/s，Decode 为输出 token/s。

'''+table(['模型','阶段','吞吐 EP2 / PP2','PP 吞吐变化','TTFT，秒','TPOT，毫秒','端到端，秒'],e2e_rows)+'''

冷 Prefill：模型变大后，PP 的吞吐领先从 **14.15% 缩小为 6.39%**，方向与假设一致，但没有发生 EP/PP 胜负反转。小模型 EP 的三轮吞吐范围为 **61,246–78,270**，PP 为 **76,622–78,155**；范围重叠、EP 波动明显，不能把两个均值之差当成稳定的模型大小效应。

Decode：两个模型都是 EP2 更快。PP 的相对吞吐损失反而从小模型的 **25.08% 缩小到大模型的 19.92%**；TPOT 绝对差约为 **3.50 / 3.89 毫秒**。因此不能说模型越重，端到端就越偏向 EP。PP 的首 token 更早与后续 Decode 更慢可以同时发生。

## 3. Prefill 分项：确实看到了更大的 Expert 代价

单位：**实际 GPU 内核累计毫秒／每卡／整批 8 请求**。下方第一张表将 PP 两阶段取卡均值，第二张表逐阶段列出全部分项。Attention 包含模块内投影，Expert 为路由专家模块，包含其内部重排、激活和 GEMM；共享专家没有混入该列。

'''+table(['模型','EP Attention','PP Attention','EP Expert','PP Expert','PP 多出的 Expert','EP Dispatch＋Combine','额外 Expert / EP D+C'],prefill_rows)+'''

### Prefill 全部分段绝对值（毫秒）

'''+absolute_tables['prefill']+'''

两种布局每卡 Attention 累计量在各模型内部接近。PP 增加的 Expert 模块工作从约 **66 毫秒增至 118 毫秒**，而 EP 的 Dispatch＋Combine 都在 **130 毫秒左右**；两者之比从约 50% 升至 91%。这支持“较重的本地 Expert 执行成本更容易吃掉省下的通信”的解释，与冷 Prefill 的端到端均值方向吻合。

但这些是累计内核时间，通信可能包含等待并与计算重叠，**不能用 130−118 直接预测可获得的端到端收益**。PP 另外保留边界通信；NCCL 累计驻留还包含等上游数据和控制，不能当成纯传输。没有采 HBM 流量或 SRAM 命中率，因此这里没有证明额外成本全部来自权重加载。

PP 把每层专家从 128 个集中到 256 个，同时把本卡层数减半；总常驻专家参数未必增加，但每次 grouped GEMM 的专家数、每专家 token 行数、chunk 组织和权重复用改变。应使用这些实际执行形状解释差异，而不是仅用整模型参数量。

## 4. Decode 分项：Expert 代价增长，通信节省也在增长

单位：**实际一次 Decode Graph 的 GPU 内核累计毫秒**。Attention 只包含 FA3／线性 Attention 核心；Expert 只包含路由 grouped GEMM。该范围比 Prefill 表窄，不能跨两表直接计算阶段占比。PP 两列分别是先后两个阶段；每卡只执行 20 或 24 层，EP 每卡执行 40 或 48 层。

'''+table(['模型','EP Attention 核心','PP 两阶段 Attention 核心','EP Expert GEMM','PP 两阶段 Expert GEMM','EP Dispatch＋Combine','Expert 每层 EP / PP'],decode_rows)+'''

### Decode 全部分段绝对值（毫秒）

'''+absolute_tables['decode']+'''

共享 Expert、投影、路由等未在这份 Decode 分类中单独拆开，包含于“其他内核”；不能将“未单独分类”写成零。NCCL 仅统计 Graph 内驻留；PP 的 Graph 外阶段传输不在零值内。分段累计可因不同 stream 重叠而超过 Graph 总跨度，不能相加来计算 TPOT。

按每层归一化，PP 的 Expert GEMM 比 EP 高约 **56%（35B）／66%（122B）**。沿单个微批依次经过两阶段，PP Expert 累计为 **2.886 / 3.922 毫秒**，比 EP 多 **1.040 / 1.554 毫秒**；它们是工作量累计，不是流水稳态每 token 的额外延迟。但 EP 的通信累计也由约 **1.26 毫秒增至 1.91 毫秒**；不能只计 Expert 的增加、不计随模型形状改变的通信。两个 PP 阶段可以跨槽重叠，Graph 外还存在传输、反馈和调度，以上内核工作量不能直接加成请求 TPOT。短 profile 也不足以对正式 TPOT 的全部差额作排他归因。

## 5. 容量会改变另一个方向的结论

各自历史启动的逻辑 KV token 容量：35B 为 EP **3,199,872**、PP **3,677,824**，约 1.15 倍；122B 为 EP **237,952**、PP **750,848**，约 3.16 倍。这是具体启动预算下的逻辑容量，不是显存带宽或跨模型等工作量。

已有长缓存实验在模型之间用了不同缓存长度：35B 16／32 请求各缓存 179,968／89,984 token；122B 8／16 请求各缓存 26,752／13,376 token。因此只能在每个模型内部比较 EP/PP，不能将它们当作“只改模型大小”的横向证据。所有这些两者都能驻留的 Decode 点，EP 仍更快；完整数据保留在 full_metrics.csv。

若 EP 容量先耗尽，而 PP 仍能驻留同样请求，PP 还可能节省排队和 CPU 换入时间。这甚至可能让较大模型更需要 PP，但本轮没有同工作量的 EP 溢出／HiCache 对照，不能报告已测得的容量胜出倍数。

## 6. 目前可采用的判断，而非大小阈值

**PP 有利的条件：关键路径上实际省掉的专家通信时间，大于专家集中增加的计算时间加流水边界、反馈和调度成本；若容量不足，再计入省掉的排队／换入成本。**

这里“专家重”应指本卡每层实际命中专家的权重字节、GEMM 形状及每专家 token 行数，不只是模型名称里的总参数量。现有 Prefill 数据支持这一机制线索，Decode 数据不支持“更大的模型一定更偏 EP”的总体结论。

证据限制：两模型精度与 GEMM 内核不同，层数、隐藏维度、专家宽度也不同；EP 支持原生 overlap，而此版本 PP 不支持；122B 的正式 Prefill 开了发送就绪适配，35B 原始组未开。Prefill 为 eager，Decode 开 CUDA Graph。后续 35B PP 单独 chunk 优化没有混入本比较。每格仅 3 轮、profile 为短诊断；小模型原始 Guard postcheck 曾因 rankless 单卡加载日志解析失败，补充只读缓存审查通过，原始失败状态不改写。

## 7. 数据入口与可重算性

- [全部正式均值／中位数／P99、三轮范围](full_metrics.csv)：26 行原始汇总，增加模型标识，不替换源数据。
- [相同请求条件下的四组对照](matched_workloads.csv)。
- [Prefill 完整分阶段值与派生计算](prefill_components.csv)；[Decode 完整分阶段值及每层归一化](decode_components.csv)。
- [源路径、SHA256](source_manifest.json)；[离线校验](validation.json)；[可重算脚本](build_report.py)。

![H100 双卡相同请求条件下的吞吐及 Prefill 分项](model_comparison.png)

![Prefill 与 Decode 全部分段绝对耗时，毫秒](absolute_components.png)
'''
    (OUT/'REPORT.md').write_text(report)
    plot(comparisons, prefill)

    # All inputs must still match the evidence inventory after generating tables.
    assert all(digest(Path(s['path']))==s['sha256'] for s in sources)
    validation = dict(passed=True, new_gpu_runs=0, input_files=len(sources),
        formal_metric_rows=len(metrics), matched_cells=len(comparisons),
        equal_request_composition_checked=True, inputs_hash_verified=True,
        separate_prefill_decode_scopes=True, absolute_component_rows=len(absolute_rows),
        all_plot_segments_have_absolute_ms_labels=True, inherited_validation=checked,
        limitations=['Not a precision-controlled or size-only experiment.',
                     'Original small-model Guard failure is preserved.',
                     'Kernel cumulatives are not additive critical-path latency.'])
    (OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    manifest = dict(id=OUT.name,title='H100 two-GPU EP2/PP2 model-weight evidence analysis',
        created_at='2026-10-26',status='complete_offline_analysis',new_gpu_runs=0,
        implementation=dict(engine='SGLang',source_commit='870269702285d94bdd926a50cc29ad1a20bcf60c',
            worktree=str(ROOT/'.engine-worktrees/glm47-ep16-canonical-profile-20261004'),
            new_engine_changes=False,source_state='Inherited historical runs; not a live checkout claim'),
        deployment=dict(historical_node='178',gpus='H100 80GB',ep_pair=[0,1],pp_pair=[2,3],
            interconnect='Both physical pairs NVLink NV6; no cross-node network',new_allocation=False),
        source_files=sources,key_results=dict(prefill='PP mean advantage shrinks from 14.15 to 6.39 percent',
            decode='Both favor EP; PP relative throughput deficit shrinks, not grows',
            expert='Extra Prefill Expert cumulative rises 66.37 to 117.73 kernel-ms/card'),
        caveats=validation['limitations'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='ok',output=str(OUT),validation=validation),ensure_ascii=False))


if __name__ == '__main__':
    main()
