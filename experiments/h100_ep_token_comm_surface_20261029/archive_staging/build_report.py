"""CPU-only absolute raw communication tables/figures, with no aggregation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def load_rows(path):
    with Path(path).open() as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        for field in ('round','hidden','experts','topk','ep','global_original_tokens','chunks','capacity_per_rank',
                'event_1based','critical_rank','physical_gpu','received_assignments_per_rank'):
            row[field]=int(row[field])
        for field in ('dispatch_ms','combine_ms','total_ms'):
            row[field]=float(row[field])
        assert row['event_1based']==1
        assert abs(row['dispatch_ms']+row['combine_ms']-row['total_ms'])<1e-3
    return rows


def build(run,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((run/'config.json').read_text())
    check=json.loads((run/'comparison.json').read_text())
    assert check['passed'] and check['aggregation']=='none' and check['formal_points']==624
    rows=load_rows(run/'absolute_event01.csv')
    assert len(rows)==624
    lookup={(r['shape'],r['round'],r['mode'],r['ep'],r['global_original_tokens']):r for r in rows}
    assert len(lookup)==624
    for name in ('absolute_event01.csv','all_critical_events.csv','all_rank_events.csv','comparison.json',
            'config.json','frozen_inputs.json','cpu_tests.json','launch.json'):
        shutil.copy2(run/name,out/name)
    tokens=cfg['token_sweep']
    chunk_limit=cfg['global_chunk_limit']
    shapes=cfg['shapes']
    figures=[]
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42})
    for repeat in (1,2):
        for mode in ('graph','eager'):
            for metric,label in (('total_ms','Dispatch + Combine'),('dispatch_ms','Dispatch'),('combine_ms','Combine')):
                figure,axes=plt.subplots(4,1,figsize=(18,14),layout='constrained')
                vals=[r[metric] for r in rows if r['round']==repeat and r['mode']==mode]
                norm=LogNorm(vmin=min(vals),vmax=max(vals))
                for model,axis in zip(shapes,axes.flat):
                    data=np.array([[lookup[(model['id'],repeat,mode,ep,n)][metric] for n in tokens] for ep in (2,4,8)])
                    img=axis.imshow(data,aspect='auto',cmap='YlOrRd',norm=norm)
                    axis.set_xticks(range(len(tokens)),[str(n) for n in tokens],rotation=45,ha='right')
                    axis.set_yticks(range(3),['EP2 / 1 node','EP4 / 1 node','EP8 / 2 nodes'])
                    axis.set_xlabel('Original unique tokens across the entire EP group')
                    axis.set_title(f'{model["label"]}: H={model["hidden"]}, experts={model["experts"]}, top-8, BF16')
                    axis.axvline(tokens.index(chunk_limit)+.5,color='#2456a6',linewidth=2)
                    for i in range(3):
                        for j in range(len(tokens)):
                            value=data[i,j]
                            text=f'{value:.3f}' if value<10 else f'{value:.2f}'
                            axis.text(j,i,text,ha='center',va='center',fontsize=12,
                                color='white' if norm(value)>.69 else '#171717')
                colorbar=figure.colorbar(img,ax=axes,shrink=.8,pad=.02)
                colorbar.set_label('Absolute latency (ms); color scale in ms')
                figure.suptitle(f'H100 {label} | {mode} | round {repeat} | raw event 1/30, aligned critical rank\n'
                    f'One call through {chunk_limit} tokens; 32768 tokens = {32768//chunk_limit} measured chunks on every EP degree',fontsize=16)
                stem=f'{metric}_{mode}_round{repeat}'
                figure.savefig(out/f'{stem}.png',dpi=160)
                figure.savefig(out/f'{stem}.pdf')
                plt.close(figure)
                figures.append(stem)
    lines=['# H100 EP × token 通信二维表：绝对毫秒值','',
        '横轴是**整个 EP 组的原始输入 token 总数**，取 8、16、32、64、128、256、512、1024、2048、4096、8192、16384、32768；纵轴是 EP2、EP4、EP8。',
        '每 token 路由到 8 个 Expert。每卡实际收到的 token–expert 行数 = `总 token × 8 / EP`；前一张 Expert 单卡图使用的是后一个口径，不能直接按相同横轴列相加。','',
        '**每格固定显示第 1 次原始 CUDA event 的绝对毫秒值，没有对 30 次采样、两轮或 GPU 求平均，也没有中位数或 relative 指标。** '
        '一次交换要等各 rank 完成，因此每次取该次 Dispatch+Combine 用时最大的 rank；Dispatch 和 Combine 均来自这个相同 rank。原始单次值会波动，完整 30 次数据已保留。','',
        '## 测量范围','',
        '- 现有预约中的 H100 80GB：spartan-gpgpu011（33758541）、spartan-gpgpu171（33646189），共 8 张。EP2/EP4 在单节点 NVLink 内；EP8 使用两节点，节点内 NVLink、节点间 RDMA。',
        '- 四个模型仅借用 hidden size / Expert 数 / top-k；全部交换 BF16 激活，不加载模型，不做 Expert GEMM、Attention、路由计算或 KV cache。122B 本次也是 BF16 通信，不能据此代表 FP8 通信。',
        '- 同一已验证的 H100 SM90 DeepEP low-latency native；CE、PCIe payload、FP8、LogFMT 关闭。记录实际 native hash、CPU host-memory NIC handler 激活证据、peer 映射和拓扑。',
        '- 每个节点的标准 Slurm 作业步使用该预约已有的 64 个 CPU，避免 4 个持续运行的 CPU NIC 代理与主线程共挤在 4 个 CPU 上；每个 rank 都记录并验证实际 CPU affinity。',
        '- 全局路由和 token ID 在 EP2/EP4/EP8 保持一致。两轮分开展示，第二轮交换 EP2/EP4 所在节点、反转模型顺序。',
        '- 每点初始化、分配、输入构造、正确性验证、Gloo barrier 与 Graph capture 不计时。Eager 先预热 10 次；Graph 先做 10 次 Eager 预热，捕获和验证后再预热 10 次 Graph replay；各正式记录 30 次。计时包含收发 hook、接收等待和 Eager 的 host feed 空隙。',
        '- EP2 / EP4 / EP8 的源 token 容量固定为每卡 1024 / 512 / 256，所有大小沿用该容量；`num_qps_per_rank = Experts / EP`，NVSHMEM QP depth 固定 4096。',
        '- **全组分块上限固定 2048 token，三个 EP 相同。8–2048 为一次 D+C，4096 / 8192 / 16384 / 32768 分别为 2 / 4 / 8 / 16 次。表中是完成全部这些交换的实测总时延，分项也是同一次测量中各块时长之和；没有把一次耗时乘调用数。**',
        '- 这个分块协议是在 DeepEP LL 缓冲容量与 QP 深度约束下采用的受控通信协议；32768 列不是一个不分块的 DeepEP LL 调用，也不是完整模型 Prefill/Decode 延迟。它不包含计算重叠或 PP 调度。','',
        '## 主图：Graph，第一轮','',
        '![绝对 Dispatch+Combine 毫秒值](total_ms_graph_round1.png)','',
        '## 从绝对值可以看到什么','']
    def latency(shape,repeat,ep,n):
        return lookup[(shape,repeat,'graph',ep,n)]['total_ms']
    lines += [
        f'- 小消息不一定因 EP 增加而更快。例如 GLM、8 token，第一轮 EP2 / EP4 分别为 '
        f'{latency("glm47",1,2,8):.6f} / {latency("glm47",1,4,8):.6f} ms；2048 token 时则为 '
        f'{latency("glm47",1,2,2048):.6f} / {latency("glm47",1,4,2048):.6f} ms。第二轮同一大档分别为 '
        f'{latency("glm47",2,2,2048):.6f} / {latency("glm47",2,4,2048):.6f} ms。这里观察到固定开销与分摊负载的不同影响。',
        f'- 跨节点代价需要单独看：GLM、2048 token，EP8 两轮分别为 '
        f'{latency("glm47",1,8,2048):.6f} / {latency("glm47",2,8,2048):.6f} ms。EP4 与 EP8 同时改变了 GPU 数和通信域，不能将差异全部归因于 EP 数。',
        f'- 原始值中存在长等待：35B、2048 token、EP8，两轮事件 1 分别为 '
        f'{latency("qwen35b",1,8,2048):.6f} / {latency("qwen35b",2,8,2048):.6f} ms；'
        f'GLM、32768 token、EP8 则分别为 {latency("glm47",1,8,32768):.6f} / '
        f'{latency("glm47",2,8,32768):.6f} ms。未删除或平滑这些样本，也尚未定位长等待原因。'
        '单个格子以及相邻 token 档的下降不能直接当作稳定的扩展规律；需要同时查看第二轮和完整 30 次原始事件。',
        '- 单卡 Expert 计算图中大 token 时不同 Expert 数的差距是否收敛，需要按对应的 token–expert 行数查图。'
        '本通信图本身不能证明 EP/PP 的端到端排名，也不能把通信随 token 增长的代价当成已经被计算重叠隐藏。','',
        '## 简表：同一张图中的原始合计值','',
        '| 模型尺寸 | EP | 8 token (ms) | 128 token (ms) | 1024 token (ms) | 8192 token (ms) | 32768 token (ms) |',
        '|:--|--:|--:|--:|--:|--:|--:|']
    for model in shapes:
        for ep in (2,4,8):
            values=[lookup[(model['id'],1,'graph',ep,n)]['total_ms'] for n in (8,128,1024,8192,32768)]
            lines.append(f'| {model["label"]} | {ep} | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    for repeat in (1,2):
        for mode in ('graph','eager'):
            lines+=['',f'## {mode}，第 {repeat} 轮','']
            for metric,label in (('total_ms','Dispatch + Combine'),('dispatch_ms','Dispatch'),('combine_ms','Combine')):
                stem=f'{metric}_{mode}_round{repeat}'
                lines += [f'![{label}，绝对 ms]({stem}.png)','']
    lines+=['## 如何对应 Expert 计算图','',
        '例如总数为 8192 token 时，EP2/EP4/EP8 每卡分别收到 32768 / 16384 / 8192 个 token–expert 行；驻留 Expert 数分别为全局 Expert 数除以 EP。'
        '需要使用对应的单卡 Expert 行数、驻留/激活 Expert 数、padding、同一分块和同一精度，再讨论省下的 Expert 时间能否抵消通信。'
        '本实验没有直接测得 EP/PP 的端到端盈亏平衡点；不能把两个组件在不同配置下的独立最慢值相加当成实测端到端时间。','',
        '## 原始数据与复现','',
        '- [交互表格](report.html)：可切换轮次、Eager/Graph、Dispatch/Combine/合计；所有格子都是固定事件 1 的原始值。',
        '- [每格原始事件 1](absolute_event01.csv)，[全部 30 次、每次关键 rank](all_critical_events.csv)，[全部 rank 原始事件](all_rank_events.csv)。',
        '- [配置](config.json)，[模型尺寸来源](geometry_sources.json)，[科学验证](comparison.json)，[独立原始事件审计](independent_audit.json)，[清理证据](cleanup.json)，[软件包元数据](environment_metadata.json)，[来源](sources.json)，[归档清单](manifest.json)。',
        f'- 原始运行目录：`{run}`；冻结 harness 在其中 `frozen/harness/`。所有 native、wrapper、配置和 harness SHA256 见 frozen_inputs.json。',
        f'- CPU 重画：`/home/bxb1/conda/bin/python {Path(__file__).resolve()} --run {run} --out OUTPUT_DIR`。GPU 复跑使用新目录执行冻结 prepare.py，再按生成的 launch.json 通过 Guard 启动。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    save(out/'data.json',dict(rows=rows,shapes=shapes,tokens=tokens,global_chunk_limit=chunk_limit,units='ms',selected_event_1based=1,aggregation='none'))
    (out/'report.html').write_text(HTML.replace('__EP8_EXAMPLE_ROUND1__',
        f'{latency("qwen35b",1,8,2048):.6f}').replace('__EP8_EXAMPLE_ROUND2__',
        f'{latency("qwen35b",2,8,2048):.6f}'))
    save(out/'presentation.json',dict(aggregation='none',selected_event_1based=1,rounds_separate=True,
        rank_selection='largest aligned D+C event; components from same rank',
        figures=figures,source_csv_sha256=sha(run/'absolute_event01.csv'),generator_sha256=sha(__file__)))
    return dict(rows=len(rows),figures=len(figures),out=str(out))


HTML=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>H100 EP × token 绝对通信耗时</title>
<style>body{font:16px system-ui;margin:28px;color:#182233;background:#f6f8fb}h1{font-size:26px}select{padding:7px;font:inherit;margin-right:20px}.panel{background:white;border:1px solid #dae0e9;border-radius:10px;padding:18px;margin:20px 0;overflow:auto}table{border-collapse:collapse;font-size:15px;white-space:nowrap}th,td{padding:11px 12px;border:1px solid #e1e6ed;text-align:right}th{background:#edf1f7}p{max-width:1200px;line-height:1.65}.note{color:#5a6575}a{color:#194fa4}</style>
<h1>H100：EP × 总 token 通信耗时（绝对 ms）</h1>
<p>横轴：整个 EP 组的原始 token 总数。每卡 Expert 行数 = 总 token × 8 ÷ EP。每格固定显示原始事件 1/30，取该次最慢 rank 的 D+C；分项取自相同 rank。没有平均、中位数或 relative 指标。</p>
<p class="note">EP2 / EP4：单节点 NVLink；EP8：两节点 NVLink + RDMA。统一全组 2048-token 分块：4096 / 8192 / 16384 / 32768 列实际测 2 / 4 / 8 / 16 次交换的完整时延。四组统一 BF16，不加载模型，不包含 Expert 计算或 PP 调度。</p>
<p class="note">保留原始长等待样本：例如 35B / EP8 / 2048 token 的两轮首个样本分别为 __EP8_EXAMPLE_ROUND1__ 和 __EP8_EXAMPLE_ROUND2__ ms。请同时检查另一轮与全部 30 次事件；单个格子不能代表稳定延迟。</p>
<label>指标 <select id="metric"><option value="total_ms">Dispatch + Combine</option><option value="dispatch_ms">Dispatch</option><option value="combine_ms">Combine</option></select></label>
<label>模式 <select id="mode"><option value="graph">CUDA Graph</option><option value="eager">Eager</option></select></label>
<label>轮次 <select id="round"><option value="1">第一轮</option><option value="2">第二轮（EP2/4 交换节点）</option></select></label>
<div id="panels">正在读取原始数据……</div><p><a href="absolute_event01.csv">每格事件 1 CSV</a> · <a href="all_critical_events.csv">全部 30 次</a> · <a href="all_rank_events.csv">全部 rank</a> · <a href="REPORT.md">完整报告</a></p>
<script>
let data;const byid=id=>document.getElementById(id);function render(){const metric=byid('metric').value,mode=byid('mode').value,round=+byid('round').value;byid('panels').replaceChildren();for(const shape of data.shapes){const panel=document.createElement('div');panel.className='panel';const title=document.createElement('h2');title.textContent=shape.label+' — H='+shape.hidden+', '+shape.experts+' Experts, top-8';panel.append(title);const table=document.createElement('table'),head=document.createElement('tr');for(const text of ['EP / 绝对 ms',...data.tokens]){const th=document.createElement('th');th.textContent=text;head.append(th)}table.append(head);for(const ep of [2,4,8]){const tr=document.createElement('tr'),th=document.createElement('th');th.textContent='EP'+ep+' ('+(ep===8?'2 nodes':'1 node')+')';tr.append(th);for(const n of data.tokens){const r=data.rows.find(r=>r.shape===shape.id&&r.round===round&&r.mode===mode&&r.ep===ep&&r.global_original_tokens===n),td=document.createElement('td');td.textContent=r[metric].toFixed(6);td.title='rank '+r.critical_rank+' / '+r.node+' GPU'+r.physical_gpu+'\nD '+r.dispatch_ms+' ms; C '+r.combine_ms+' ms\n'+r.chunks+' chunk(s), Expert rows/GPU '+r.received_assignments_per_rank;tr.append(td)}table.append(tr)}const calls=document.createElement('tr'),label=document.createElement('th');label.textContent='D+C 调用次数';calls.append(label);for(const n of data.tokens){const td=document.createElement('td');td.textContent=Math.ceil(n/data.global_chunk_limit);calls.append(td)}table.append(calls);panel.append(table);byid('panels').append(panel)}}
for(const id of ['metric','mode','round'])byid(id).addEventListener('change',render);fetch('data.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(d=>{data=d;render()}).catch(e=>{byid('panels').textContent='数据加载失败：'+e.message});
</script></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.run.resolve(),args.out.resolve())))
