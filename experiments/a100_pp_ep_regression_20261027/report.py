"""Generate a concise report and static scientific charts from saved analysis."""
from collections import defaultdict
import json
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parent


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,type=Path);a=ap.parse_args()
    run=a.run.absolute();src=ROOT/'analysis'/run.name;d=json.loads((src/'result.json').read_text())
    assert len(d['metrics'])==15 and d['audit']['guard_postcheck']
    out=ROOT/'report'/run.name;out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((run/'config.json').read_text())
    m={(r['workload'],r['arm'],r['policy']):r for r in d['metrics']}
    def row(c,arm,p=0):return m[(c['id'],arm,p)]
    lines=['# A100 NVLink: EP2 and PP2 regression points','',
        '完整 Qwen3.6-35B-A3B，BF16 权重／激活／KV、FP32 GDN。节点 spartan-gpgpu069 的物理 GPU0/1，租约内实测为 NV12。EP2 的 Attention 为 DP2×TP1，Expert 为 EP2；PP2 为两个各含20层的 TP1/EP1 stage、两个最多B/2请求的槽位。Decode实际每槽B/2请求，Prefill按每Attention实例4096-token预算分块。两种布局使用相同原生 Triton Expert 家族；A100 DeepEP normal BF16、Triton Attention，CUDA Graph 与 overlap schedule 均关闭。', '',
        'Decode：每请求真实 GPU 缓存4096 token + 新输入1 token，生成128 token；Prefill：无缓存，新增128或4096 token，生成1 token。B为全局请求数。重复型与32主题多样输入沿用 H100 构造，输入指纹可逐项核验。权重加载、缓存准备、JIT和profile不计入正式性能；每配置3次暖机、5轮正式cohort。PP0为原生调度；PP9为提前发射下一槽的调度优化，Expert算子保持相同。', '',
        '吞吐按每轮总token数除以整批从首个请求开始至最后输出的时长，再取5轮均值；Decode计输出token，Prefill计本次新输入token。TTFT为客户端首token延迟，TPOT为首末token间隔除以127，E2E为客户端请求完成时间；延迟取全部正式请求均值。固定请求组的入队等待计入延迟。', '',
        '| Workload | EP2 tok/s | PP2 native tok/s | PP2 lookahead tok/s | PP reference / EP |',
        '|---|---:|---:|---:|---:|']
    for c in cfg['workloads']:
        ep,pp=row(c,'ep2'),row(c,'pp2')
        opt=row(c,'pp2',9) if c['phase']=='decode' else None
        best=opt if opt else pp
        lines.append(f"| {c['id']} | {ep['throughput_tok_s']:.2f} | {pp['throughput_tok_s']:.2f} | {opt['throughput_tok_s']:.2f} | {best['throughput_tok_s']/ep['throughput_tok_s']:.3f}× |" if opt else
                     f"| {c['id']} | {ep['throughput_tok_s']:.2f} | {pp['throughput_tok_s']:.2f} | — | {best['throughput_tok_s']/ep['throughput_tok_s']:.3f}× |")
    scheduling=[]
    for c in cfg['workloads']:
        if c['phase']=='decode':
            scheduling.append(f"{c['id']}: {(row(c,'pp2',9)['throughput_tok_s']/row(c,'pp2')['throughput_tok_s']-1)*100:+.2f}%")
    lines[2:2]=['已完成6组workload、15个计时点。PP9相对PP原生调度的变化为 '+ '；'.join(scheduling)+'。具体PP／EP取舍与所有延迟见下表。','']
    lines+=['','| Workload / layout | TTFT ms | TPOT ms | E2E ms | Throughput SD |',
            '|---|---:|---:|---:|---:|']
    for r in d['metrics']:
        t=f"{r['tpot_ms']:.2f}" if r['tpot_ms'] is not None else '—'
        lines.append(f"| {r['workload']} / {r['arm']} p{r['policy']} | {r['ttft_ms']:.2f} | {t} | {r['e2e_ms']:.2f} | {r['throughput_stdev']:.2f} |")
    lines+=['','独立profile将GPU kernel经CPU launch correlation归到实际层和算子；Decode丢弃前20个forward，选之后16个；Prefill覆盖整次冷输入cohort。以下对两rank取均值，单位为每卡每次实际层调用的微秒。Attention含投影及核心；Expert分开routed GEMM和其他路由执行开销，shared Expert另存CSV。PP为EP1，Dispatch/Combine无跨卡专家交换。GPU区间统计不能直接相加还原客户端TPOT。', '',
            '| Workload / layout | Expert total µs | Expert GEMM µs | Dispatch + Combine µs | Attention µs |',
            '|---|---:|---:|---:|---:|']
    grouped=defaultdict(list)
    for r in d['components']:grouped[(r['workload'],r['arm'],r['policy'])].append(r)
    for key,values in grouped.items():
        lines.append(f"| {key[0]} / {key[1]} p{key[2]} | {st.mean(r['expert_total'] for r in values)*1000:.2f} | {st.mean(r['expert_gemm'] for r in values)*1000:.2f} | {st.mean(r['dispatch_combine'] for r in values)*1000:.2f} | {st.mean(r['attention'] for r in values)*1000:.2f} |")
    def component(case,arm,key):return st.mean(r[key] for r in grouped[(case,arm,0)])*1000
    c='decode_b32_diverse'
    lines[2:2]=[f"B32多样输入的独立profile中，routed Expert为EP {component(c,'ep2','expert_total'):.1f} / PP {component(c,'pp2','expert_total'):.1f} µs，Dispatch+Combine为 {component(c,'ep2','dispatch_combine'):.1f} / 0 µs，Attention为 {component(c,'ep2','attention'):.1f} / {component(c,'pp2','attention'):.1f} µs。PP本地Expert更慢、Attention接近、专家交换减少，与本轮端到端PP更快的方向一致；这些GPU分项不直接相加还原TPOT。",'']
    lines+=['','同布局PP0/PP9在所有正式轮次核验完整输出序列一致，并核验实际B/2请求组、发送接收payload hash、每轮lookahead确实执行且所有权完全回收。EP/PP的输出一致率单独列出，避免把不同BF16归约顺序造成的生成分歧当作已经通过的质量验证。', '',
            '| Workload | EP/PP complete request outputs equal |','|---|---:|']
    for r in d['cross_layout_output_agreement']:lines.append(f"| {r['workload']} | {r['equal_requests']}/{r['total_requests']} |")
    lines+=['','本轮覆盖35B的6个workload、12个部署组合，另有3个PP原生Decode对照，共15个计时点。A100原生eager/Triton/DeepEP-normal与已有H100 Graph/FA3/low-latency结果属于不同硬件原生执行配置；跨GPU结果可用于部署选择，不能将差值只归因于GPU带宽。其他模型与其余workload仍待补测。', '',
            '初次公共Guard不支持Qwen而在准入前停止；一次NVLink检查遇到ANSI颜色码，在模型启动前停止。随后较小的GDN状态池在B32暖机触发精确缓存检查，运行已清理；该次已完成的B8数据作为诊断保留，不混入本表。最终采用EP全局168／每rank84、PP每stage164个GDN状态，保留每条路径最终prefix state，避免容量排队或前缀淘汰。最终结果只取此处完整的新运行。', '',
            f"[吞吐图]({out}/throughput.png) · [Expert／通信／Attention图]({out}/components.png)", '',
            f"[Metrics CSV]({src}/metrics.csv) · [Components CSV]({src}/components.csv) · [Recomputed result]({src}/result.json) · [Configuration]({run}/config.json) · [Topology]({run}/topology_qualification.json)"]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,phase in zip(axes,['decode','prefill']):
        cases=[c for c in cfg['workloads'] if c['phase']==phase];x=np.arange(len(cases));width=.24
        ep=[row(c,'ep2')['throughput_tok_s'] for c in cases]
        pp=[row(c,'pp2')['throughput_tok_s']/v for c,v in zip(cases,ep)]
        ax.bar(x-width/2 if phase=='prefill' else x-width,[1]*len(cases),width,label='EP2',color='#168e89')
        ax.bar(x+width/2 if phase=='prefill' else x,pp,width,label='PP2 native',color='#d29135')
        if phase=='decode':ax.bar(x+width,[row(c,'pp2',9)['throughput_tok_s']/v for c,v in zip(cases,ep)],width,label='PP2 lookahead',color='#8f63c4')
        labels=[f"B{c['requests']}\n{c['family']}" if phase=='decode' else f"B{c['requests']} Q{c['new_input']}\n{c['family']}" for c in cases]
        ax.set_xticks(x,labels);ax.set_title(phase.capitalize());ax.set_ylabel('Throughput / EP2 throughput');ax.axhline(1,color='#777',ls='--',lw=.7);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8)
    fig.suptitle('A100 NVLink GPU0/1 · Qwen35B BF16 · 5 unprofiled cohorts/configuration');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'throughput.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,c,title in zip(axes,['decode_b32_diverse','prefill_b8_q4096_repeated'],['Decode · B32 diverse','Prefill · B8 Q4096']):
        keys=['expert_total','dispatch_combine','attention'];x=np.arange(3);width=.34
        for arm,offset,color in [('ep2',-width/2,'#168e89'),('pp2',width/2,'#d29135')]:
            bars=ax.bar(x+offset,[component(c,arm,k) for k in keys],width,label='EP2' if arm=='ep2' else 'PP2 native',color=color)
            ax.bar_label(bars,fmt='%.0f',fontsize=9,padding=3)
        ax.set_xticks(x,['Routed Expert','Dispatch + Combine','Attention']);ax.set_title(title)
        ax.set_ylabel('GPU µs / card / actual layer call');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=9);ax.margins(y=.15)
    fig.suptitle('A100 NVLink: local Expert cost versus token exchange');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'components.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    print(out)


if __name__=='__main__':main()
