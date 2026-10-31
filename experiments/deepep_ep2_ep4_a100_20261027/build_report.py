"""CPU-only audit, aggregation and plots for fixed-global-data DeepEP EP2/EP4."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics as st
import sys

LAYOUTS=('ep2_nvlink','ep4_mixed','ep2_rdma','ep4_rdma')
LABELS=('EP2: NVLink','EP4: NVLink + RDMA','EP2: RDMA only','EP4: RDMA only')
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
    data_root=run/'analysis_v1'
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
            row['mixed_EP4_over_NVLink_EP2']=row['ep4_mixed_total_us']/row['ep2_nvlink_total_us']
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
    def bars(ax,ns,mode,col,scale):
        x=np.arange(len(ns));width=.18
        for j,(layout,label,color) in enumerate(zip(LAYOUTS,LABELS,COLORS)):
            data=[lookup[n,mode,layout] for n in ns]
            vals=np.array([r[col]/scale for r in data])
            lo=np.array([r[col+'_min_round']/scale for r in data]);hi=np.array([r[col+'_max_round']/scale for r in data])
            ax.bar(x+(j-1.5)*width,vals,width,label=label,color=color,
                yerr=np.maximum(0,np.array([vals-lo,hi-vals])),capsize=2,hatch='//' if j>=2 else None,edgecolor='white',linewidth=.5)
        ax.set_xticks(x,[f'{n:,}' for n in ns]);ax.set_xlabel('Global unique source tokens (fixed across EP2 / EP4)')
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        return ax
    fig,axes=plt.subplots(1,2,figsize=(12.4,5.5))
    bars(axes[0],[n for n in cfg['token_sweep'] if n<=32],'graph','total_us',1)
    bars(axes[1],[n for n in cfg['token_sweep'] if n>32],'eager','total_us',1000)
    axes[0].set(title='Small messages: CUDA Graph',ylabel='Dispatch + Combine latency (us)')
    axes[1].set(title='Large messages: eager, same low-latency API',ylabel='Dispatch + Combine latency (ms)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.5,.91),frameon=False)
    fig.suptitle('A100: fixed total data, DeepEP EP2 versus EP4',fontsize=16,y=.99)
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
    def table(selected):
        text=['| 总 token | 模式 | EP2 NVLink D / C / 合计 (us) | EP4 mixed D / C / 合计 (us) | EP4 / EP2 | RDMA-only EP2 / EP4 (us) | RDMA 比值 |',
              '|---:|:---|---:|---:|---:|---:|---:|']
        for r in selected:
            def trip(l):return ' / '.join(f'{r[l+"_"+c]:.2f}' for c in ('dispatch_us','combine_us','total_us'))
            text.append(f'| {r["tokens"]} | {r["mode"]} | {trip("ep2_nvlink")} | {trip("ep4_mixed")} | {r["mixed_EP4_over_NVLink_EP2"]:.2f}x | {r["ep2_rdma_total_us"]:.2f} / {r["ep4_rdma_total_us"]:.2f} | {r["RDMA_EP4_over_EP2"]:.2f}x |')
        return '\n'.join(text)
    main_rows=[r for r in comps if (r['tokens']<=32 and r['mode']=='graph') or r['tokens']>32]
    report=f'''# A100 DeepEP：相同总数据量下 EP2 扩到 EP4

固定整个通信组的总输入 token 数、token ID、Expert 路由、隐藏维度和精度，只改变 Expert Parallelism (EP) 的 GPU 数量。Dispatch 把输入发到对应 Expert，Combine 把加权输出收回；这里用 identity Expert，没有 Expert GEMM、Attention、模型或 KV cache。

## 主要发现

- 实际双卡 NVLink 扩到四卡 mixed：8/16/32-token 小消息的合计时延为 1.64/2.22/3.19 倍；256/1024/2048-token 大消息为 9.91/17.67/21.64 倍。两个分项均随跨 NVLink 对的通信增加而变慢。
- 统一走 RDMA：小消息 EP4/EP2 为 1.02/1.11/1.20 倍，大消息稳定在 1.47–1.50 倍。单卡输入减半，并没有使全组通信更快。
- 四卡 mixed 与双卡 RDMA 的大消息时延几乎相同（差约 1%）：路由计数显示，两者均有 4N 个 assignments 走 RDMA；四卡全 RDMA 则有 6N 个。这与共享 RDMA 通路受总远端数据量约束相符，未直接测量 NIC wire bytes 或饱和率。

这组数据支持按实际通信域和消息规模分别 profile EP 的通信代价；不能用同一个常数近似不同 EP degree、不同拓扑和大消息的 Dispatch/Combine。

## 数据与时延

每个输入 token 是 5120 个 BF16 数，即 10 KiB。全组扫描 {cfg['token_sweep']} 个唯一 token；小消息为 80/160/320 KiB，大消息为 2.5/10/20 MiB 原始输入。每个 token 路由到 160 个 Expert 中的 8 个。EP2 每卡负责 80 个 Expert，EP4 每卡负责 40 个。每卡输入 token 数随卡数加倍而减半。

测试节点 {cfg['nodes'][0]}，Slurm booking {cfg['booked_job_id']}。实际 GPU 为 NVIDIA A100 80GB PCIe，GPU0/1 和 GPU2/3 分别组成 NVLink 对，两个 GPU 对之间没有直接 NVLink。EP2 NVLink 使用 GPU0/1；EP4 mixed 使用四卡、对内 NVLink、跨对 RDMA。RDMA-only 对照分别使用 GPU0/2 和四卡，禁用远端 P2P 映射。NIC 为 ConnectX-6 Dx，mlx5_bond_0；本次没有跨节点。

所有点都使用同一份 DeepEP low-latency BF16 Dispatch/Combine，CE、FP8 和 LogFMT 均关闭。小消息分别测 eager 和 CUDA Graph，大消息测 eager。大消息代表 Prefill 的消息规模，尚未覆盖 DeepEP normal-mode Prefill 后端。

每点 warmup 10 次、计时 30 次；四种布局按正序和反序各跑一轮。每个 sample 先取各 rank 中最大的 Dispatch+Combine CUDA event 时延，Dispatch 和 Combine 分项取自同一个 rank。表格是两轮各自中位数的均值，图中误差线为两轮中位数范围。分项中位数之和不要求等于总时延中位数。计时包括完成 hook、接收等待，以及 eager 的提交间隙；Gloo barrier、正确性检查和初始化不计时。

{table(main_rows)}

![Total Dispatch + Combine latency](latency.png)

![Separate Dispatch and Combine latency](components.png)

## 小消息 eager 对照

{table([r for r in comps if r['tokens']<=32 and r['mode']=='eager'])}

## 怎样理解 GPU 数量的影响

相同总输入量不等于相同跨 GPU 传输量。这里路由保持不变且均匀分布：EP2 的全组远端 Expert assignments 为 4N，EP4 为 6N，增加 50%；单个源 rank 的远端 assignments 则从 2N 降到 1.5N，减少 25%。N 指全组唯一输入 token 数。这里统计的是路由 assignments，不是 NIC 实测 wire bytes。

因此扩 EP 同时减少单卡工作量、增加远端通信比例和参与者；在这台机器上还会跨越 NVLink 对的边界。实际部署比较应看 EP2 NVLink 与 EP4 mixed；RDMA-only 对照用于观察统一传输策略下的变化。本实验只测通信，不能单独得出完整 PP/EP 的吞吐或 TPOT 结论。

## 校验与复现

72 个布局/消息/模式/轮次点，{raw_count} 条 rank sample 已从原始 rank 文件独立重算；计数、源 token ID、BF16 payload、加权 identity、Graph 更新输入、不同 Expert 输出和倾斜到达检查通过。所有冻结输入哈希复核通过；8 个阶段的自有进程退出、GPU 空闲检查与资源释放由 Guard 保存。

前置失败 r1/r2/r3 没有有效计时：依次为 EP2 不接受 pair_only、容量与 QP depth 不匹配、扩大 QP depth 后 NIC 队列创建失败。失败记录保持原样。r4 的 8 个 GPU 阶段全部完成并释放资源；原始 Guard postcheck 因汇总脚本重复传入 iteration 字段而失败，状态和错误保持原样。analysis_v1 离线修复汇总并从 24 个原始 rank 文件独立复核，不重跑或改变 GPU 数据。r4 使用 capacity=1024/rank、QP depth=4096；同样配置用于全部布局和消息档。

- [完整逐轮指标]({data_root}/full_metrics.csv)，[所有 rank samples]({data_root}/raw_samples.csv)
- [聚合指标](aggregate.csv)，[布局对比](comparisons.csv)，[独立审计](audit.json)
- [配置]({run}/config.json)，[冻结输入]({run}/frozen_inputs.json)，[校验]({data_root}/comparison.json)，[清理证据]({run}/task/cleanup.json)
- [冻结 harness]({run}/frozen/harness/prepare.py)，[启动命令]({run}/launch.json)
- [本报告及图表生成器]({Path(__file__).resolve()})

DeepEP commit `81a1b5960f5fc73bce33045a56e5b6970e2bcb3c`，tag `experiment/deepep/ce-glm-ep8-validated-20261010`；native SHA256 `{cfg['native']['sha256']}`。测试参数、source/native 记录与 Graph 协议均与原始结果一起保存。
'''
    (out/'REPORT.md').write_text(report)
    print(json.dumps(dict(passed=True,report=str(out/'REPORT.md'),points=checks,rank_samples=raw_count,comparisons=comps),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.run,a.out)
