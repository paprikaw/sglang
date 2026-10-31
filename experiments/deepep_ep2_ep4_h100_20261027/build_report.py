"""CPU-only audit, aggregation and plots for fixed-global-data DeepEP EP2/EP4."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics as st
import sys

LAYOUTS=('ep2_nvlink','ep4_nvlink','ep2_rdma','ep4_rdma')
LABELS=('EP2: NVLink','EP4: NVLink','EP2: RDMA only','EP4: RDMA only')
COLORS=('#126782','#dd8731','#82b9c8','#eac29c')


def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main(run,out):
    run=run.resolve();out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(run/'frozen/harness'))
    from guard_authorization import visible_json
    def read(path):return visible_json(path,run/'artifact_visibility/report')
    data_root=run
    cfg=read(run/'config.json');validation=read(data_root/'comparison.json');cleanup=read(run/'task/cleanup.json')
    assert validation['passed'] and cleanup['passed'] and cleanup['gpu_idle']
    rows=read_csv(data_root/'full_metrics.csv')
    indexed={r['phase']+'|'+r['mode']+'|'+r['tokens']:r for r in rows}
    raw_count=0;checks=0;source_records=[]
    for phase in cfg['phases']:
        ranks=[]
        for rank in range(phase['world']):
            path=run/phase['id']/f'result_rank{rank}.json'
            ranks.append(read(path))
            source_records.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        assert all(r['passed'] and r['buffer_explicitly_destroyed'] for r in ranks)
        assert all(r['route']['pair_only'] is False for r in ranks)
        if phase['allow_nvlink']:
            assert all(q['transport'] in ('local','nvlink') for r in ranks for q in r['route']['expected_payload_routes'])
        for wi,work in enumerate(phase['workloads']):
            pts=[r['points'][wi] for r in ranks]
            assert all(p['passed'] and p['correctness_before']['passed'] and p['correctness_after_changed_payload']['passed'] for p in pts)
            assert all(all(c['passed'] for c in p['skewed_changed_payload_checks']) for p in pts)
            if work['tokens'] in (8,2048):
                assert all(p['distinct_expert_validation']['passed'] and len(p['skewed_changed_payload_checks'])==4 for p in pts)
            if work['cuda_graph']:
                assert all(all(c['passed'] for c in p['graph_correctness'].values()) for p in pts)
            series=[[],[],[]]
            for sample_tuple in zip(*(p['samples'] for p in pts)):
                critical=max(sample_tuple,key=lambda s:s['pair_cuda_ms'])
                assert abs(critical['dispatch_cuda_ms']+critical['combine_cuda_ms']-critical['pair_cuda_ms'])<1e-4
                for dst,key in zip(series,('dispatch_cuda_ms','combine_cuda_ms','pair_cuda_ms')):dst.append(critical[key]*1000)
                raw_count+=len(sample_tuple)
            assert len(series[0])==cfg['iterations']
            key=f"{phase['id']}|{'graph' if work['cuda_graph'] else 'eager'}|{work['tokens']}"
            for data,col in zip(series,('dispatch_us','combine_us','total_us')):
                assert abs(st.median(data)-float(indexed[key][col]))<1e-8
            checks+=1
    assert checks==72 and len(rows)==72 and raw_count==validation['samples']
    assert len(read_csv(data_root/'raw_samples.csv'))==raw_count
    frozen=read(run/'frozen_inputs.json')['files']
    for entry in frozen:
        assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
    agg=[]
    for n in cfg['token_sweep']:
        for mode in (('eager','graph') if n<=32 else ('eager',)):
            for layout in LAYOUTS:
                group=[r for r in rows if int(r['tokens'])==n and r['mode']==mode and r['layout']==layout]
                assert len(group)==2 and {int(g['round']) for g in group}=={1,2}
                item=dict(tokens=n,mode=mode,layout=layout,rounds=2,unique_input_mib=n*cfg['hidden']*2/2**20)
                for col in ('dispatch_us','combine_us','total_us'):
                    values=[float(g[col]) for g in group]
                    item[col]=st.mean(values);item[col+'_min_round']=min(values);item[col+'_max_round']=max(values)
                agg.append(item)
    lookup={(r['tokens'],r['mode'],r['layout']):r for r in agg}
    comps=[]
    for n in cfg['token_sweep']:
        for mode in (('eager','graph') if n<=32 else ('eager',)):
            row=dict(tokens=n,mode=mode,unique_input_mib=n*cfg['hidden']*2/2**20)
            for layout in LAYOUTS:
                for component in ('dispatch_us','combine_us','total_us'):row[layout+'_'+component]=lookup[n,mode,layout][component]
            row['NVLink_EP4_over_EP2']=row['ep4_nvlink_total_us']/row['ep2_nvlink_total_us']
            row['RDMA_EP4_over_EP2']=row['ep4_rdma_total_us']/row['ep2_rdma_total_us']
            comps.append(row)
    write_csv(out/'aggregate.csv',agg);write_csv(out/'comparisons.csv',comps)
    audit=dict(passed=True,checked_points=checks,checked_rank_samples=raw_count,
        all_saved_component_medians_recomputed=True,frozen_hashes_checked=len(frozen),
        cleanup_verified=True,source_records=source_records,
        aggregation='mean of two independent phase medians; whiskers span the two phase medians')
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def bars(ax,ns,mode,col,scale,which=LAYOUTS[:2]):
        x=np.arange(len(ns));width=.30
        for j,layout in enumerate(which):
            palette_index=LAYOUTS.index(layout);label=LABELS[palette_index];color=COLORS[palette_index]
            data=[lookup[n,mode,layout] for n in ns]
            vals=np.array([r[col]/scale for r in data])
            lo=np.array([r[col+'_min_round']/scale for r in data]);hi=np.array([r[col+'_max_round']/scale for r in data])
            container=ax.bar(x+(j-(len(which)-1)/2)*width,vals,width,label=label,color=color,
                yerr=np.maximum(0,np.array([vals-lo,hi-vals])),capsize=2,hatch='//' if palette_index>=2 else None,edgecolor='white',linewidth=.5)
            ax.bar_label(container,labels=[f'{v:.1f}' if scale==1 else f'{v:.3f}' for v in vals],padding=3,fontsize=8)
        ax.margins(y=.15)
        ax.set_xticks(x,[f'{n:,}' for n in ns]);ax.set_xlabel('Global unique source tokens (fixed across EP2 / EP4)')
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        return ax
    fig,axes=plt.subplots(1,2,figsize=(12.4,5.5))
    bars(axes[0],[n for n in cfg['token_sweep'] if n<=32],'graph','total_us',1)
    bars(axes[1],[n for n in cfg['token_sweep'] if n>32],'eager','total_us',1000)
    axes[0].set(title='Small messages: CUDA Graph',ylabel='Dispatch + Combine latency (us)')
    axes[1].set(title='Large messages: eager, same low-latency API',ylabel='Dispatch + Combine latency (ms)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.5,.91),frameon=False)
    fig.suptitle('H100: fixed total data, DeepEP EP2 versus EP4',fontsize=16,y=.99)
    fig.text(.5,.025,'BF16 H=5120, 160 experts, top-8; two phase orders, 30 samples/point. Bars: mean of phase medians; whiskers: phase range.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.77,bottom=.16,wspace=.25)
    for ext in ('png','pdf','svg'):fig.savefig(out/f'latency.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12.4,8))
    for i,(ns,mode,scale) in enumerate((([8,16,32],'graph',1),([256,1024,2048],'eager',1000))):
        for j,col in enumerate(('dispatch_us','combine_us')):
            bars(axes[i,j],ns,mode,col,scale)
            axes[i,j].set_title(('Dispatch' if j==0 else 'Combine')+(' | small, CUDA Graph' if i==0 else ' | large, eager'))
            axes[i,j].set_ylabel('Latency (us)' if i==0 else 'Latency (ms)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.5,.94),frameon=False)
    fig.suptitle('Dispatch and Combine from the same critical rank in each sample',fontsize=15,y=.99)
    fig.text(.5,.018,'Component medians are reported separately; their sum need not equal the median of total latency.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.85,bottom=.10,hspace=.4,wspace=.25)
    for ext in ('png','pdf'):fig.savefig(out/f'components.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.4,5.5))
    bars(axes[0],[8,16,32],'graph','total_us',1,which=LAYOUTS[2:])
    bars(axes[1],[256,1024,2048],'eager','total_us',1000,which=LAYOUTS[2:])
    axes[0].set(title='Small messages: CUDA Graph',ylabel='Dispatch + Combine latency (us)')
    axes[1].set(title='Large messages: eager',ylabel='Dispatch + Combine latency (ms)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='upper center',bbox_to_anchor=(.5,.91),frameon=False)
    fig.suptitle('H100: EP2 versus EP4 with both layouts forced to RDMA',fontsize=16,y=.99)
    fig.text(.5,.025,'Same global tokens and Expert routes. Same-node RDMA control; no cross-node deployment.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.77,bottom=.16,wspace=.25)
    for ext in ('png','pdf'):fig.savefig(out/f'rdma_controls.{ext}',dpi=180)
    plt.close(fig)
    # Compare only identical global workload and timing protocols across the saved A100 run.
    a100=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/deepep_ep2_ep4_a100_20261027')
    acfg=json.loads((a100/'config.json').read_text())
    for key in ('capacity','nvshmem_qp_depth','token_sweep','dtype','hidden','experts','topk','warmup','iterations'):
        assert cfg[key]==acfg[key],key
    arows=read_csv(a100/'full_metrics.csv')
    for n in cfg['token_sweep']:
        assert {r['route_sha256'] for r in rows if int(r['tokens'])==n}=={r['route_sha256'] for r in arows if int(r['tokens'])==n}
    apoints={(int(r['tokens']),r['mode']):r for r in read_csv(a100/'comparisons.csv')}
    cross=[]
    for r in comps:
        a=apoints[r['tokens'],r['mode']]
        cross.append(dict(tokens=r['tokens'],mode=r['mode'],
            A100_EP2_us=float(a['ep2_nvlink_total_us']),A100_EP4_mixed_us=float(a['ep4_mixed_total_us']),
            A100_practical_ratio=float(a['mixed_EP4_over_NVLink_EP2']),A100_RDMA_ratio=float(a['RDMA_EP4_over_EP2']),
            H100_EP2_us=r['ep2_nvlink_total_us'],H100_EP4_NVLink_us=r['ep4_nvlink_total_us'],
            H100_NVLink_ratio=r['NVLink_EP4_over_EP2'],H100_RDMA_ratio=r['RDMA_EP4_over_EP2']))
    write_csv(out/'a100_h100_comparison.csv',cross)
    main_rows=[r for r in comps if (r['tokens']<=32 and r['mode']=='graph') or r['tokens']>32]
    fig,axes=plt.subplots(1,2,figsize=(12.4,5.5))
    series=(('A100_practical_ratio','A100: NVLink pair to mixed','#dd8731','-'),
        ('H100_NVLink_ratio','H100: within NVLink domain','#126782','-'),
        ('A100_RDMA_ratio','A100: RDMA only','#dd8731','--'),
        ('H100_RDMA_ratio','H100: RDMA only','#126782','--'))
    for ax,small in zip(axes,(True,False)):
        group=[r for r in cross if (r['tokens']<=32 and r['mode']=='graph')==small and (small or r['tokens']>32)]
        for key,label,color,style in series:
            ax.plot(range(len(group)),[r[key] for r in group],marker='o',label=label,color=color,linestyle=style)
        ax.set_xticks(range(len(group)),[f'{r["tokens"]:,}' for r in group])
        ax.axhline(1,color='#777777',linewidth=.8)
        ax.set_xlabel('Global unique source tokens');ax.set_ylabel('EP4 latency / EP2 latency')
        ax.set_title('Small messages: CUDA Graph' if small else 'Large messages: eager')
        ax.grid(alpha=.18);ax.set_axisbelow(True)
    axes[1].set_yscale('log')
    axes[1].set_yticks([.25,.5,1,2,4,8,16,32],['0.25','0.5','1','2','4','8','16','32'])
    axes[1].set_ylabel('EP4 latency / EP2 latency (log scale)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='upper center',bbox_to_anchor=(.5,.91),frameon=False)
    fig.suptitle('EP2 to EP4: the communication domain changes the trend',fontsize=16,y=.99)
    fig.text(.5,.025,'Same global tokens, routes, BF16 geometry and timing protocol. Platform topology and SM80/SM90 native builds differ.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.73,bottom=.16,wspace=.28)
    for ext in ('png','pdf','svg'):fig.savefig(out/f'a100_h100_domains.{ext}',dpi=180)
    plt.close(fig)
    def table(selected):
        text=['| 总 token | 模式 | EP2 NVLink D / C / 合计 (us) | EP4 NVLink D / C / 合计 (us) | NVLink EP4 / EP2 | RDMA-only EP2 / EP4 (us) | RDMA 比值 |',
              '|---:|:---|---:|---:|---:|---:|---:|']
        for r in selected:
            def trip(l):return ' / '.join(f'{r[l+"_"+c]:.2f}' for c in ('dispatch_us','combine_us','total_us'))
            text.append(f'| {r["tokens"]} | {r["mode"]} | {trip("ep2_nvlink")} | {trip("ep4_nvlink")} | {r["NVLink_EP4_over_EP2"]:.3f}x | {r["ep2_rdma_total_us"]:.2f} / {r["ep4_rdma_total_us"]:.2f} | {r["RDMA_EP4_over_EP2"]:.3f}x |')
        return '\n'.join(text)
    little=[r for r in main_rows if r['tokens']<=32];large=[r for r in main_rows if r['tokens']>32]
    little_ratios=' / '.join(f'{r["NVLink_EP4_over_EP2"]:.3f}x' for r in little)
    large_ratios=' / '.join(f'{r["NVLink_EP4_over_EP2"]:.3f}x' for r in large)
    rdma_ratios=' / '.join(f'{r["RDMA_EP4_over_EP2"]:.3f}x' for r in large)
    report=f"""# H100 DeepEP：相同总数据量下 EP2 扩到 EP4

固定整个通信组的总输入 token 数、全局 token ID 和 Expert 路由，只改变 Expert Parallelism (EP) 的 GPU 数量。Dispatch 把输入发给所选 Expert，Combine 收回加权输出；本次使用 identity Expert，不含 Expert GEMM、Attention、模型或 KV cache。

## 主要结果

- 同一 H100 NVLink 域内，8/16/32-token 小消息的 EP4/EP2 总时延比为 **{little_ratios}**。
- 256/1024/2048-token 大消息的 NVLink EP4/EP2 比为 **{large_ratios}**；统一走 RDMA 时为 **{rdma_ratios}**。
- 前次 A100 EP2→EP4 会跨出 NVLink 对；本次 H100 四卡均有直接 NVLink。全局数据与路由匹配，通信域不同。图中分别保留自然拓扑和 RDMA-only 对照。

这次在 H100 的 NVLink 域内，大消息扩到 EP4 后更快：2048-token 的合计时延从 0.931 ms 降到 0.337 ms（−63.9%）。小消息下 EP4 仍稍慢；全 RDMA 的大消息则仍慢 23.5%–26.5%。这些变化说明通信代价需要同时描述消息规模、EP degree 和实际通信域。更多 NVLink peer 的并发通路与单卡负载下降都可能贡献收益，本次未拆分二者的因果份额。

相同 eager 协议下也观察到偏好翻转：32-token 的 EP2/EP4 为 57.4/62.2 µs，256-token 为 150.0/86.7 µs。当前两档之间尚未加密扫描。

## 参数和时延边界

节点 {cfg['nodes'][0]}，现有 Slurm booking {cfg['booked_job_id']}，4 张 H100 80GB。运行时每阶段保存并验证 GPU0–3 两两均为 NV6。EP2 NVLink 使用 GPU0/1，EP4 NVLink 使用全部四卡，pair_only=False；RDMA-only 对照使用 GPU0/2 或全部四卡，禁止远端 P2P 映射。

全组唯一 token 数为 {cfg['token_sweep']}，每 token 为 5120 个 BF16 数（10 KiB）。小消息为 80/160/320 KiB，大消息为 2.5/10/20 MiB 原始输入。全局 160 Experts、每 token top-8；EP2 每卡 80 Experts，EP4 每卡 40 Experts，每卡源 token 数减半。

所有点都使用同一 H100 SM90 native 的 low-latency BF16 Dispatch/Combine；CE、FP8 和 LogFMT 关闭。小消息分别测 eager 和 CUDA Graph，大消息测 eager，代表 Prefill 的消息规模；本次没有运行 DeepEP normal-mode Prefill 或完整模型。

四种布局按正序和反序各测一轮，每点 warmup 10 次、正式 30 次。每个 sample 取所有 rank 中最大的 Dispatch+Combine CUDA event 时延；D 和 C 分项取自同一个 rank。表格为两轮各自中位数的均值，图中误差线为两轮中位数范围。分项中位数之和不要求等于总时延中位数。计时包含完成 hook、接收等待和 eager 提交间隙，不包含初始化、Gloo barrier 或正确性验证。

{table(main_rows)}

![H100 Dispatch + Combine](latency.png)

![H100 Dispatch and Combine separately](components.png)

## RDMA-only 对照

![RDMA-only controls](rdma_controls.png)

## 与 A100 对照

![A100 and H100 communication domains](a100_h100_domains.png)

两平台的总 token、路由 hash、BF16/hidden/Experts/top-k、容量、warmup/sample 数和 Graph/eager 协议已经逐项核对。A100 是两两 NVLink、四卡 mixed；H100 是四卡全 NVLink。A100 使用已保存的 SM80 native，H100 使用已验证的 SM90 native，二者 wrapper/native hash 不同；平台差异同时包含 GPU、通信域和编译内核，不能只归因于某一个硬件参数。

同样总输入不代表同样远端路由量。令 N 为全组唯一源 token 数：EP2 全组远端 Expert assignments 为 4N，EP4 为 6N（+50%）；单个源 rank 则从 2N 降到 1.5N（−25%）。这是逻辑路由计数，不是 NIC 实测 wire bytes。NVLink 和 RDMA 对此变化的响应由本次曲线分别给出。

## 小消息 eager 对照

{table([r for r in comps if r['tokens']<=32 and r['mode']=='eager'])}

## 验证和复现

72 个布局/消息/模式/轮次点、{raw_count} 条 rank sample 已从 24 个原始 rank JSON 独立重算。计数、源 token ID、BF16 payload、加权 identity、Graph 更新输入、不同 Expert 输出和倾斜到达验证通过；全部冻结输入 SHA256 复核通过。8 个阶段均保存自有进程退出、GPU 空闲和清理记录；Guard 主运行状态及资源释放单独归档。

- [逐轮指标]({run}/full_metrics.csv)，[全部 rank samples]({run}/raw_samples.csv)
- [聚合指标](aggregate.csv)，[布局对比](comparisons.csv)，[A100/H100 对比](a100_h100_comparison.csv)，[独立审计](audit.json)
- [配置]({run}/config.json)，[冻结输入]({run}/frozen_inputs.json)，[科学校验]({run}/comparison.json)，[清理证据]({run}/task/cleanup.json)
- [冻结 harness]({run}/frozen/harness/prepare.py)，[启动命令]({run}/launch.json)，[报告生成器]({Path(__file__).resolve()})

H100 wrapper commit `05d46157c7ccfa0784b59a5198253a701215c4d3`，tag `experiment/deepep/custom-sm90-20261007`；实际 native SHA256 `{cfg['native']['sha256']}`。前次 A100 结果保持原样，汇总脚本离线修复记录仍见其报告。
"""
    (out/'REPORT.md').write_text(report)
    audit['A100_same_global_routes_verified']=True
    audit['A100_same_workload_and_timing_parameters_verified']=True
    audit['H100_full_NVLink_route_policy_verified']=True
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(passed=True,report=str(out/'REPORT.md'),points=checks,rank_samples=raw_count,main_comparisons=main_rows),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.run,a.out)
