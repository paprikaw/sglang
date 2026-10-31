"""Validate and model the measured HY3 expert surface; CPU-only analysis."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import tarfile

import numpy as np
from scipy.optimize import nnls

ROOT=Path(__file__).resolve().parent
TOKENS=(8,32,128,512,2048,8192)
EXPERTS=(2,4,8,12,24,48)


def save(path,value):
    path.write_text(json.dumps(value,indent=2,default=str)+'\n')


def table(path,rows):
    with path.open('w',newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def features(rows,kind='tile_weight'):
    names={'flops_only':['intercept','T'],
           'tile_weight':['intercept','T','padded_rows','active_experts']}[kind]
    values=np.array([[dict(intercept=1,T=r['T'],padded_rows=r['actual_padded_rows'],
                           active_experts=r['E'])[k] for k in names] for r in rows],float)
    return names,values


def fit(X,y):
    # Relative-error weighted NNLS. Coefficients are empirical, not hardware counters.
    scale=np.maximum(X.max(axis=0),1)
    coefs,_=nnls(X/scale/y[:,None],np.ones(len(y)))
    return coefs/scale


def errors(y,pred):
    relative=np.abs(pred-y)/y
    return dict(n=len(y),mean_absolute_percentage_error=float(relative.mean()),
                median_absolute_percentage_error=float(np.median(relative)),
                p95_absolute_percentage_error=float(np.quantile(relative,.95)),
                max_absolute_percentage_error=float(relative.max()))


def plain(r):
    t=r['timing']['median_ms'];bs=r['block_median_ms']
    return dict(cohort=r['cohort'],component=r['component'],mode=r['mode'],T=r['T'],
        E=r['E'],resident=r['resident'],rows_per_expert=r['T']/r['E'],
        active_weight_MiB=r['E']*36,
        config=json.dumps(r.get('config'),sort_keys=True),
        padded_rows=r.get('actual_padded_rows'),m_tiles=r.get('m_tiles'),
        allocated_slots=r.get('allocated_slots'),capacity=r.get('capacity'),
        **r['timing'],block_min_median_ms=min(bs),block_max_median_ms=max(bs),
        block_relative_range=(max(bs)-min(bs))/t,
        useful_rows_per_s=r['T']*1000/t,
        useful_tflops=6*4096*1536*r['T']/t/1e9)


def main(run,out):
    if out.exists():raise RuntimeError('Immutable output directory already exists')
    r=json.loads((run/'profile/result.json').read_text())
    spec=json.loads((run/'dashboard_spec.json').read_text())
    cleanup=json.loads((run/'profile/task/cleanup.json').read_text())
    posts=list((run/'dashboard_guard/checks').glob('*-post.json'))
    assert len(posts)==1
    post=json.loads(posts[0].read_text())
    assert r['passed'] and post['passed'] and cleanup['passed'] and len(r['rows'])==112
    assert len(r['numerical'])==112 and all(x['passed'] for x in r['numerical'])
    for x in r['rows']:
        samples=x['raw_ms'];cs=x['counts']
        assert len(samples)==30 and len(x['block_median_ms'])==3
        assert all(math.isfinite(v) and v>0 for v in samples)
        assert math.isclose(statistics.median(samples),x['timing']['median_ms'])
        assert math.isclose(statistics.mean(samples),x['timing']['mean_ms'])
        assert sum(cs)==x['T'] and sum(v>0 for v in cs)==x['E']
        assert max(v for v in cs if v)-min(v for v in cs if v)<=1
        if x['component']=='expert_core':
            bm=x['config']['BLOCK_SIZE_M']
            assert x['actual_padded_rows']==sum(math.ceil(v/bm)*bm for v in cs if v)
    surface=[x for x in r['rows'] if x['cohort']=='surface' and x['mode']=='cuda_graph']
    matched=[x for x in r['rows'] if x['cohort']=='ep_pp_matched' and x['component']=='expert_core' and x['mode']=='cuda_graph']
    assert len(surface)==32 and len(matched)==12
    y=np.array([x['timing']['median_ms'] for x in surface])
    models={};predictions={};cross_predictions={}
    for kind in ('flops_only','tile_weight'):
        names,X=features(surface,kind);coefs=fit(X,y);pred=X@coefs
        cv=np.zeros_like(y)
        for total in TOKENS:
            train=np.array([x['T']!=total for x in surface])
            cv[~train]=X[~train]@fit(X[train],y[train])
        _,mx=features(matched,kind)
        my=np.array([x['timing']['median_ms'] for x in matched])
        models[kind]=dict(features=names,coefficients_ms=dict(zip(names,coefs.tolist())),
            training=errors(y,pred),leave_one_T_level_out=errors(y,cv),
            matched_cohort_holdout=errors(my,mx@coefs))
        predictions[kind]=pred;cross_predictions[kind]=cv
    validation=[]
    for i,x in enumerate(surface):
        validation.append(dict(cohort='surface',T=x['T'],E=x['E'],resident=x['resident'],
            measured_ms=float(y[i]),predicted_ms=float(predictions['tile_weight'][i]),
            leave_T_out_ms=float(cross_predictions['tile_weight'][i]),
            flops_only_leave_T_out_ms=float(cross_predictions['flops_only'][i])))
    _,mx=features(matched);coefs=np.array(list(models['tile_weight']['coefficients_ms'].values()))
    for x,p in zip(matched,mx@coefs):
        validation.append(dict(cohort='matched_holdout',T=x['T'],E=x['E'],resident=x['resident'],
                               measured_ms=x['timing']['median_ms'],predicted_ms=float(p)))
    knees=[]
    for total in TOKENS:
        ps=[x for x in surface if x['T']==total];best=min(x['timing']['median_ms'] for x in ps)
        accepted=[x['E'] for x in ps if x['timing']['median_ms']<=1.1*best]
        last=max(accepted)
        knees.append(dict(T=total,criterion='within 110% of sampled minimum at fixed T',
                          sampled_best_ms=best,accepted_experts=accepted,
                          largest_sampled_accepted=last,next_sampled_E=next((x['E'] for x in ps if x['E']>last),None),
                          not_a_hardware_upper_bound=True))
    comparisons=[]
    def point(total,resident,component):
        return next(x for x in r['rows'] if x['cohort']=='ep_pp_matched' and x['mode']=='cuda_graph'
                    and x['T']==total and x['resident']==resident and x['component']==component)
    for total in TOKENS:
        item=dict(T=total,EP8_active=min(total,24),PP_active=min(total,48))
        for component,prefix in [('expert_core','core'),('bridge_full_compute','bridge'),('layout_prepare_only','prepare')]:
            a=point(total,24,component);b=point(total,48,component)
            v=a['timing']['median_ms'];w=b['timing']['median_ms']
            item.update({prefix+'_EP8_ms':v,prefix+'_PP_ms':w,prefix+'_PP_extra_ms':w-v,
                         prefix+'_PP_over_EP8':w/v})
        comparisons.append(item)
    out.mkdir(parents=True)
    for src,name in [(run/'profile/result.json','raw_result.json'),(run/'selection.json','weight_selection.json'),
                     (run/'dashboard_spec.json','frozen_spec.json')]:shutil.copyfile(src,out/name)
    table(out/'full_metrics.csv',[plain(x) for x in r['rows']])
    table(out/'surface_graph_metrics.csv',[plain(x) for x in surface])
    table(out/'matched_comparison.csv',comparisons)
    table(out/'model_validation.csv',validation)
    save(out/'cost_model.json',dict(formula='a + b*T + c*T_padded + d*E_active',models=models,
        fit_protocol='relative-error weighted nonnegative least squares, CUDA Graph surface only',
        cross_validation='Leave every complete T level out; refit and predict that entire level',
        caveat='Empirical within this backend/geometry; not a physical bandwidth or occupancy estimate; no SLO guarantee',
        bridge_extra_dimension='allocated slots = E_resident*C; controlled common C=ceil(T/24), not native transport capacity'))
    save(out/'knees.json',knees)
    save(out/'validation.json',dict(passed=True,numerical_checks=112,
        max_nrms=max(x['nrms'] for x in r['numerical']),cleanup=cleanup,guard_postcheck=post))
    with tarfile.open(out/'harness.tar.gz','w:gz') as archive:
        for p in sorted(ROOT.glob('*.py')):archive.add(p,arcname=p.name)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    matrix=np.full((len(TOKENS),len(EXPERTS)),np.nan)
    for x in surface:matrix[TOKENS.index(x['T']),EXPERTS.index(x['E'])]=x['timing']['median_ms']
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.4),layout='constrained')
    for ax,data,title,label in [(axes[0],matrix,'Expert-core latency','ms; lower is better'),
              (axes[1],np.array(TOKENS)[:,None]/matrix,'Useful row throughput','rows / ms; higher is better')]:
        im=ax.imshow(data,norm=LogNorm(np.nanmin(data),np.nanmax(data)),cmap='viridis',aspect='auto')
        ax.set_xticks(range(len(EXPERTS)),EXPERTS);ax.set_yticks(range(len(TOKENS)),[f'{t:,}' for t in TOKENS])
        ax.set_xlabel('Active experts E (48 resident)');ax.set_ylabel('Actual local token-expert rows T')
        ax.set_title(title)
        for i in range(len(TOKENS)):
            for j in range(len(EXPERTS)):
                if np.isfinite(data[i,j]):
                    position=(math.log(data[i,j])-math.log(np.nanmin(data)))/(math.log(np.nanmax(data))-math.log(np.nanmin(data)))
                    ax.text(j,i,f'{data[i,j]:.3f}' if ax is axes[0] else f'{data[i,j]:.0f}',ha='center',va='center',
                            color='black' if position>.6 else 'white',fontsize=9)
                else:ax.text(j,i,'N/A',ha='center',va='center',color='#888888')
        fig.colorbar(im,ax=ax,shrink=.8,label=label)
    fig.suptitle('HY3 BF16 / A100 / production Triton defaults / CUDA Graph\nBalanced assignments; no router, Attention, shared expert or communication')
    fig.savefig(out/'surface.png',dpi=180);fig.savefig(out/'surface.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    shown=(128,512,2048,8192);xs=np.arange(len(shown));width=.36
    for ax,component,title in [(axes[0],'expert_core','Expert core (alignment excluded)'),
                               (axes[1],'bridge_full_compute','Production LL-layout bridge + compute')]:
        for resident,offset,label,color in [(24,-width/2,'EP8 local: 24 experts','#2d719b'),
                                            (48,width/2,'PP2xEP4 local: 48 experts','#d88532')]:
            ps=[point(t,resident,component) for t in shown];ys=[x['timing']['median_ms'] for x in ps]
            low=[max(0,y-min(p['block_median_ms'])) for p,y in zip(ps,ys)]
            high=[max(0,max(p['block_median_ms'])-y) for p,y in zip(ps,ys)]
            bars=ax.bar(xs+offset,ys,width,label=label,color=color,yerr=[low,high],capsize=3)
            ax.bar_label(bars,fmt='%.2f',padding=4,fontsize=8)
        ax.set_xticks(xs,[f'{t:,}' for t in shown]);ax.set_xlabel('Equal actual local assignments T')
        ax.set_ylabel('CUDA Graph median (ms)');ax.set_title(title);ax.legend(fontsize=8)
        ax.set_ylim(0,4.3);ax.grid(axis='y',alpha=.16)
    fig.suptitle('Matched local work, not full PP/EP latency\nError bars: three block medians; bridge uses a controlled common per-expert stride')
    fig.savefig(out/'ep_pp_compute.png',dpi=180);fig.savefig(out/'ep_pp_compute.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for kind,label,color in [('flops_only','FLOPs-only baseline','#9b9999'),('tile_weight','Rows + rounded tiles + active weights','#287f9c')]:
        axes[0].scatter(y,cross_predictions[kind],label=label,color=color,alpha=.8,s=32)
    limits=(min(y)*.8,max(y)*1.3)
    axes[0].plot(limits,limits,'k--',alpha=.5);axes[0].set_xscale('log');axes[0].set_yscale('log')
    axes[0].set_xlabel('Measured ms');axes[0].set_ylabel('Predicted ms, entire T level held out')
    axes[0].legend(fontsize=8);axes[0].set_title('Leave-one-T-level-out validation')
    for total in TOKENS:
        ps=[x for x in surface if x['T']==total]
        axes[1].plot([x['E'] for x in ps],[x['timing']['median_ms'] for x in ps],marker='o',label=f'T={total:,}')
    axes[1].set_xlabel('Active experts E');axes[1].set_ylabel('Measured expert-core ms')
    axes[1].set_xticks(EXPERTS);axes[1].legend(fontsize=8,ncol=2);axes[1].set_title('Fixed useful work, varying expert working set')
    axes[1].grid(alpha=.2)
    fig.savefig(out/'model_validation.png',dpi=180);fig.savefig(out/'model_validation.pdf');plt.close(fig)

    c=models['tile_weight']['coefficients_ms'];cv=models['tile_weight']['leave_one_T_level_out']
    lines=['# HY3：T × expert 二维拐点与 PP／EP 计算侧模型','',
        '## 测量口径','',
        'A100 80GB PCIe，节点065的GPU0；HY3 BF16第1层真实专家权重，H=4096、I=1536，模型192个routed experts、top-8。'
        '主网格固定48个resident experts，只改变活跃数E和实际接收的token–expert配对行数T；路由均衡。'
        '不加载完整模型，不测Attention、shared expert、router或网络。',
        '',
        'T不是request数，也不是新增token数。一次原始token可能命中8个expert；本轮每条配对作为已dispatch的一行输入，只计算一次。'
        '每expert三块BF16权重合计36MiB；48个resident共1.6875GiB。活跃权重字节数不是实测HBM流量。',
        '',
        '采用本机生产SGLang的Triton路径。当前A100/HY3缺少调优JSON，实际回退默认配置；因此这是当前实现的拐点，不是A100硬件极限。'
        '每点5次准备预热、每块前3次预热、3块×10次CUDA event；Graph为主，eager为对照。权重加载、JIT、compact输入的对齐在计时外。'
        'Graph核心包含W13 GEMM、SiLU/multiply、W2 GEMM和本地top-1输出处理。',
        '',
        '## 主网格：CUDA Graph median（ms）','',
        '| T | E2 | E4 | E8 | E12 | E24 | E48 |','|---:|---:|---:|---:|---:|---:|---:|']
    for total in TOKENS:
        lines.append('| '+f'{total:,}'+' | '+' | '.join('—' if not np.isfinite(v) else f'{v:.4f}' for v in matrix[TOKENS.index(total)])+' |')
    lines += ['', '![T与E曲面](surface.png)','',
        'expert轴的“平台”定义为固定T时不超过该行采样最小延迟的110%，不是带宽或expert数量的硬上限。']
    for knee in knees:lines.append(f"- T={knee['T']:,}：采样范围内E={knee['accepted_experts']}；下一采样E={knee['next_sampled_E']}。")
    lines += ['',
        'T轴上，E48时T128→512为1.4013→1.4730ms，行数增加4倍、延迟增加5.1%；T2048为1.7510ms，T8192为3.0300ms。'
        '这支持“存在可利用的增加行数余量”，不支持无限扩batch或已经找到效率上限：最大采样T8192仍有更高rows/s。',
        '', '## 代码为何产生这些特性','',
        '实际对齐工作量是 T_pad = Σ ceil(m_e / BLOCK_M) × BLOCK_M，理想有效FLOPs=6HIT。'
        '当前默认BF16配置在输入M≤resident expert数时选BLOCK_M=16，其他时候选64；不是统一的128行。'
        'T128/E48时，每expert约2–3行，却至少占一个64行块，T_pad=3072；E24为1536。'
        '同时E增大使独立权重工作集增大。T2048时E24与E48的T_pad同为3072，延迟仍为1.0619和1.7510ms，说明只按padding或FLOPs解释也不够。'
        '分块、权重复用与实际活跃行数需要同时建模。本轮未采HBM/L2/occupancy计数器，不能把全部差额断言为DRAM读权重。',
        '', '## 可校准计算模型','',
        '拟合 f(T,E) = a + b T + c T_pad + d E，单位ms。使用相对误差加权非负最小二乘，系数是经验参数，不等于硬件带宽/算力。','',
        f"本轮 a={c['intercept']:.7f}，b={c['T']:.9f}，c={c['padded_rows']:.9f}，d={c['active_experts']:.7f}。",'',
        f"整档T留出交叉验证：MAPE={100*cv['mean_absolute_percentage_error']:.2f}%，"
        f"中位绝对百分误差={100*cv['median_absolute_percentage_error']:.2f}%，最坏={100*cv['max_absolute_percentage_error']:.2f}%。"
        f"仅按T/FLOPs的基线MAPE={100*models['flops_only']['leave_one_T_level_out']['mean_absolute_percentage_error']:.2f}%。"
        f"没有参与拟合的12个EP/PP配对控制点MAPE={100*models['tile_weight']['matched_cohort_holdout']['mean_absolute_percentage_error']:.2f}%。"
        '边界点误差不小，模型可筛配置，不能直接承诺绝对SLO；实际选择应优先使用邻近实测值，并留余量。',
        '', '## EP8 对比 PP2×EP4：如何映射','',
        '同一个全局N个新增token，均衡top-8路由、Attention TP4、MoE前先scatter去除TP副本：'
        'EP8每卡24个resident expert，平均接收T=N；PP2×EP4分两个N/2 microbatch，每卡48个resident expert，'
        '每stage每次平均T=(N/2)×8/4=N。因此同资源、等全局工作量的局部计算对照是f(T,24)与f(T,48)。'
        'N不足以覆盖全部expert时，活跃数最多T；不能硬设每卡都命中24或48个expert。',
        '', '| 每卡T | EP8核心ms | PP核心ms | PP核心多耗时ms | PP/EP核心 | EP8完整适配ms | PP完整适配ms |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for x in comparisons:
        lines.append(f"| {x['T']:,} | {x['core_EP8_ms']:.4f} | {x['core_PP_ms']:.4f} | {x['core_PP_extra_ms']:.4f} | {x['core_PP_over_EP8']:.3f} | {x['bridge_EP8_ms']:.4f} | {x['bridge_PP_ms']:.4f} |")
    lines += ['', '![局部配对对照](ep_pp_compute.png)','',
        'LL完整适配包括生产bf16_deepep布局处理、重新对齐、expert计算和输出mask，仍没有网络。'
        '这里为控制变量，两组共用每expert的stride C=ceil(T/24)；这不是完整engine的实际接收容量。'
        '适配会处理resident×C个分配槽，因而其模型还需要容量C维度。不能把完整适配与compact时间的差直接视为一种可外推的纯转换成本。',
        '',
        '对PP的含义：小T8两种局部核心几乎相同；T128–2048时PP更多expert、每expert更少行，需要付出约0.65ms/层/卡的额外核心成本。'
        '只有节省的跨节点Dispatch/Combine、容量允许的额外batch收益足以抵消这个成本、PP边界/反馈和气泡时，PP才更好。'
        'T8192样本差额更小，但区块漂移较大，不能把6.9%差距认定为稳定的硬件优势。',
        '',
        '一阶稳态比较（两个等大slot、stage均衡、同全局batch）可写成：'
        'EP8归一化周期≈L[A+f24+D8]；PP2归一化周期≈L[A+f48+D4]+H_PP。'
        '即D8−D4 > f48−f24+H_PP/L时PP在该简化模型中有利。'
        'D为该层通信关键路径，H_PP包括边界传输、反馈、失衡与气泡；存在重叠时应测关键路径而非直接累加独立microbench。'
        '这是筛选条件，不是本轮已经测得的整模型TPOT。容量改变batch之后，要重新代入T和Attention曲面。',
        '', '## 置信边界与证据','',
        '112个输出/reference数值校验全部通过，最大NRMS约0.001145；112条计时行、每条30个样本，Guard postcheck、GPU退出与租约释放通过。'
        '3块是同一进程内采样，不是3次独立启动；P99仅描述这30个样本。'
        '主网格个别长T点的区块median相对范围达24.0%，因此10%平台边缘和小于10%的排名不稳。'
        'Graph反复重放是warm-replay，不等同于80层模型连续运行的cache状态。',
        '',
        '源码锚点：fused_moe_triton_config.py:246–260（默认tile）；moe_align_block_size.py（逐expert补齐）；'
        'fused_moe_triton_kernels.py:419–440（M/N块和有效token）；triton.py:338（LL BF16接计算核）；'
        'bf16_deepep.py（allocated capacity处理）；communicator.py:1191（Attention TP后scatter）。'
        '完整路径、commit、输入SHA、原始样本、模型留出预测、均衡counts均保存于本目录。','']
    (out/'summary.md').write_text('\n'.join(lines))
    save(out/'manifest.json',dict(id='hy3_expert_surface_20261022_'+run.name,title='HY3 T/expert knees and PP/EP local compute cost model',
        created_at=datetime.now(timezone.utc).isoformat(),status='complete_verified',
        implementation=dict(engine='SGLang',sources=spec['sources'],backend=r['backend'],
                            tuned_config=False,dirty=False,old_implementation=False),
        model=dict(path=spec['model_cache']['model_path'],layer=1,H=4096,I=1536,experts=192,topk=8,
                   precision='BF16',shared_expert_included=False,selected_experts=list(range(48))),
        testbed=dict(node='spartan-gpgpu065',job_id='33463294',actual_gpu=0,registered_node_gpus=4,gpu=r['gpu']),
        workload=dict(T=list(TOKENS),E=list(EXPERTS),balanced=True,request_context='not applicable: expert-only',
                      weight_load_timed=False,attention=False,communication=False,full_model_loaded=False),
        engine_parameters=dict(resident_surface=48,matched_residents=[24,48],timing_modes=['eager','cuda_graph'],
                               warmups=5,immediate_warmups=3,blocks=3,samples_per_block=10,
                               synthetic_activations=True,real_expert_weights=True),
        validation=dict(numerical_checks=112,guard_postcheck=True,cleanup=True,lease_id=r['authorization']['lease_id'],hardware_counters=False),
        artifacts=dict(original_run=str(run),surface='surface.png',paired='ep_pp_compute.png',model='cost_model.json'),
        key_results=dict(knees=knees,models=models,matched=comparisons)))
    checksums={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    save(out/'checksums.json',checksums)
    assert all(hashlib.sha256((out/name).read_bytes()).hexdigest()==value for name,value in checksums.items())
    print(json.dumps(dict(output=str(out),verified_files=len(checksums),models=models,matched=comparisons),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'runs/r2')
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    main(args.run.resolve(),args.out.resolve())
