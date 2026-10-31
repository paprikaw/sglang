"""Publish a traceable report for explicitly selected completed matched runs."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics as st
import subprocess

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'deliverables'
ID='qwen35b_h100_ep_pp_fair_expert_20261026'
ARCHIVE=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')/ID


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def link(label,p):return f'[{label}]({p})'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--native',required=True);ap.add_argument('--optimized',required=True)
    args=ap.parse_args();selected=[args.native,args.optimized]
    OUT.mkdir(exist_ok=True)
    rows=[r for r in csv.DictReader((ROOT/'analysis/metrics.csv').open()) if r['run'] in selected]
    components=[r for r in csv.DictReader((ROOT/'analysis/components.csv').open()) if r['run'] in selected]
    assert len(rows)==8
    audits=read(ROOT/'analysis/run_audit.json')
    for name in selected:
        item=next(a for a in audits if a['run']==name)
        assert item['stage']=='complete' and item['postcheck']=='passed' and item['gpu_registry_state']=='released'
    cfg=read(ROOT/'runs'/args.optimized/'config.json')
    launch=read(ROOT/'runs'/args.optimized/'launch_manifest.json')
    engine=Path(cfg['engine_root'])
    commit=subprocess.check_output(['git','-C',str(engine),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(engine),'status','--porcelain'],text=True).strip()
    assert commit==cfg['engine_commit'] and not dirty
    spec=read(ROOT/'runs'/args.optimized/'dashboard_spec.json')
    native=next(d for d in spec['frozen_inputs'] if d['path']==cfg['deepep_native'])
    assert sha(native['path'])==native['sha256']
    save(OUT/'selection.json',dict(native=args.native,optimized=args.optimized))
    from verify_ep_probability_control import verify
    ep_probability_control=verify(args.native,args.optimized)
    ep_matches_last_native=all(q['mean_abs_logprob_delta']==0 and q['argmax_agreement']==1
                              for q in ep_probability_control['optimized_vs_last_native'])
    vis_spec=importlib.util.spec_from_file_location('fair_finalize_visibility',
        '/home/bxb1/.codex/skills/llm-experiment-guard/scripts/artifact_visibility.py')
    vis=importlib.util.module_from_spec(vis_spec);vis_spec.loader.exec_module(vis)
    pp_numerical=[]
    for name in selected:
        probes=sorted((ROOT/'runs'/name/'pp2/routing').glob('microbench.*.json'))
        assert len(probes)>=2
        for path in probes:
            data,receipt=vis.wait_for_stable_json(path,timeout_s=15)
            result=data['numerical']['triton']
            assert result['passed'] and result['relative_l2']<.015
            pp_numerical.append(dict(run=name,source=str(path),numerical=result,visibility=receipt))
    save(ROOT/'analysis/pp_numerical_validation.json',pp_numerical)
    pp_max_l2=max(p['numerical']['relative_l2'] for p in pp_numerical)
    def metric(mode,arm,phase):
        result=[r for r in rows if r['mode']==mode and r['arm']==arm and r['phase']==phase]
        assert len(result)==1;return result[0]
    def comp(mode,arm,key):
        result=[r for r in components if r['mode']==mode and r['arm']==arm and r['phase']=='decode']
        assert len(result)==2
        return 1000*st.mean(float(r.get(key) or 0) for r in result)
    configurations=[('native','ep2','EP2 native'),('triton_small','ep2','EP2 small-batch Triton'),
                    ('native','pp2','PP2 native'),('triton_small','pp2','PP2 small-batch Triton')]
    main_table=[];component_table=[]
    for mode,arm,label in configurations:
        p,d=metric(mode,arm,'prefill'),metric(mode,arm,'decode')
        main_table.append(f"| {label} | {float(p['throughput_tok_s']):,.1f} ± {float(p['throughput_stdev']):,.1f} | {float(d['throughput_tok_s']):.1f} ± {float(d['throughput_stdev']):.1f} | {float(d['tpot_ms']):.3f} | {float(d['e2e_ms']):.1f} | {float(d['ttft_ms']):.1f} |")
        values=[comp(mode,arm,k) for k in ['attention.core','expert.routed_gemm','communication.dispatch','communication.combine','expert.layout_adapter']]
        component_table.append('| '+label+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    gain={arm:(float(metric('triton_small',arm,'decode')['throughput_tok_s'])/float(metric('native',arm,'decode')['throughput_tok_s'])-1)*100 for arm in ['ep2','pp2']}
    best={arm:max(['native','triton_small'],key=lambda mode:float(metric(mode,arm,'decode')['throughput_tok_s'])) for arm in ['ep2','pp2']}
    best_rate={arm:float(metric(best[arm],arm,'decode')['throughput_tok_s']) for arm in best}
    advantage=(best_rate['ep2']/best_rate['pp2']-1)*100
    quality=[q for q in read(ROOT/'analysis/probability_validation.json') if any(name in q['comparison'] for name in selected)]
    quality_table='\n'.join(f"| {q['comparison']} | {q['common_history_positions']}/{q['candidate_positions']} | {q['identical_argmax_positions']}/{q['common_history_positions']} | {q['mean_top10_overlap']:.2f} | {q['mean_abs_logprob_delta']:.5f} |" for q in quality)
    routes=[r for r in read(ROOT/'analysis/ep_routing.json') if r['run'] in selected and r['local_input_tokens']==4]
    actual_l2=max(r['numerical']['relative_l2'] for r in routes)
    route_table=[]
    for name in selected:
        values=[r for r in routes if r['run']==name]
        route_table.append(f"| {name} | {len(values)} | {st.mean(r['assignments'] for r in values):.2f} | {st.mean(r['active_experts'] for r in values):.2f} | {max(r['numerical']['relative_l2'] for r in values):.6f} |")
    report=f'''# H100：双方应用小批量 Expert 优化后的 EP2 / PP2 对照

同一节点 {cfg['node']} 的 NVLink GPU0/1，完整 Qwen3.6-35B-A3B BF16，固定全局8请求。EP2采用专家并行2路、Attention数据并行2路且每路Attention TP1；PP2采用两个20层stage、两个流水槽位，每个微批次最多4请求。整节点4GPU租约，实际模型始终使用同一对2GPU；四个配置顺序执行。

## 结果

双方都尝试现有 Triton 小批量 Expert GEMM 家族后，Decode吞吐变化为：EP2 **{gain['ep2']:+.2f}%**，PP2 **{gain['pp2']:+.2f}%**。本轮测试候选中，EP2较快实现为 `{best['ep2']}`（{best_rate['ep2']:.1f} tok/s），PP2为 `{best['pp2']}`（{best_rate['pp2']:.1f} tok/s）；前者相对后者的吞吐变化为 **{advantage:+.2f}%**。这是本轮候选比较，不是穷尽调优后的性能上限。

每个配置、每个阶段5轮正式cohort。吞吐列为cohort均值±样本标准差；延迟为请求均值。Prefill：每请求无缓存、新输入4096 token、输出1 token。Decode：每请求真实GPU缓存4096 token、新输入1 token、输出128 token。吞吐以全部8请求完成的cohort墙钟时间计算。TTFT为请求开始到首token，TPOT为首token之后每个输出token的平均间隔，E2E为请求开始到完成。

| 配置 | Prefill input tok/s ↑ | Decode output tok/s ↑ | Decode TPOT ms ↓ | Decode E2E ms ↓ | Decode TTFT ms ↓ |
|---|---:|---:|---:|---:|---:|
{chr(10).join(main_table)}

Prefill的大形状继续使用DeepGEMM。本轮没有观察到Prefill提速；EP均值波动较大，两个实现的Prefill中位数接近，不能把该差异归因为小批量GEMM优化。

{link('逐轮CSV',ROOT/'analysis/rounds.csv')}；{link('逐请求CSV',ROOT/'analysis/per_request.csv')}；{link('完整指标CSV',ROOT/'analysis/metrics.csv')}。所有轮次均保留；未因快慢剔除cohort。

## Expert、Attention和通信

下面为真实Decode CUDA Graph内的GPU kernel时间，单位为µs/GPU/实际层调用。每rank选择稳定本地B4的16个forward，并排除最初4个。Attention仅core，Expert仅路由GEMM；索引适配kernel单列。Dispatch/Combine保留原有DeepEP路径，GPU事件可包含等待和重叠；这些列不能直接相加解释E2E。Prefill全模块累计数据另存components.csv，计时范围不同。

| 配置 | Attention core | Expert GEMMs | Dispatch | Combine | EP新增索引kernel |
|---|---:|---:|---:|---:|---:|
{chr(10).join(component_table)}

![Matched throughput and component comparison]({OUT/'fair_ep_pp_comparison.png'})

## 什么优化应用到了双方

PP沿用之前的策略：实际本地token数≤4时使用已有BF16 Triton Expert执行，大形状继续DeepGEMM。EP也在DeepEP low-latency、本地输入token数≤4时调用**同一个 `invoke_fused_moe_kernel` GEMM实现**，按自身专家数和逻辑token数选择现有小批量tile配置。

EP新增GPU索引适配器，将DeepEP的专家分组buffer及每专家有效行数转换为共享GEMM所需的索引；输出保持原有专家分组布局，交回原生DeepEP加权combine。没有改变路由、专家选择、BF16权重/激活/KV、FP32 GDN状态、Attention、通信二进制、流水槽位或chunk。索引kernel和全部适配执行成本都包含在EP端到端计时中。EP保留原生masked SwiGLU激活；未增加通信或转回其他通信后端。

因此，这轮小批量内核选择属于EP和PP均可尝试的通用优化。此前PP自身提速的证据继续成立；研究中的PP专有贡献需要进一步利用stage局部性、流水调度或微批次组织，并与两边经过同等优化机会的基线比较。不能只拿PP的目标估算与未优化EP作最终胜负判断。

## 正确性与真实路由

启动时每rank使用真实第一层权重检查：0/1/3/15/16/17/64行边界、全部无活跃专家、全部128个本地专家活跃。索引验证覆盖每一个有效assignment，检查没有遗漏、重复或错误专家映射。真实运行另检查两rank各40层；本轮选定运行的最大相对L2误差为 {actual_l2:.6f}（相同1.5%层误差门限）。以下assignment数是实际到达本GPU的token-expert配对数，active experts是本GPU该次调用实际触及的不同专家数。

| 运行 | 本地B4层样本数 | 平均assignments/GPU/层 | 平均active experts/GPU/层 | 最大相对L2 |
|---|---:|---:|---:|---:|
{chr(10).join(route_table)}

PP在相同真实输入与权重上的原生/Triton层输出对照也通过相同1.5%相对L2门限，本轮最大相对L2为 {pp_max_l2:.6f}。{link('PP数值核验',ROOT/'analysis/pp_numerical_validation.json')}。单层诊断microbenchmark不用于上面的正式吞吐或CUDA Graph分项值。

独立的8-token概率验证使用原子请求批次：EP每个Attention DP副本4请求，PP输入8请求并由原生流水切为B4。每配置3轮，比较相同提示及完全相同生成历史处的top-10自然对数概率；下表同历史位置之外不进行概率差值比较。

| 比较 | 相同历史位置/候选位置 | 相同argmax/相同历史位置 | top-10平均交集 | 平均绝对logprob差 |
|---|---:|---:|---:|---:|
{quality_table}

EP原生自对照的最后一轮已有logprob变化；优化版三轮与原生最后一轮的64个验证位置是否全部token选择及top-10 logprob完全一致：**{ep_matches_last_native}**。因此不能将相对原生第一轮的该项概率变化单独归因于Expert优化。{link('原生自对照补充核验',ROOT/'analysis/probability_native_last_reference.json')}。BF16执行路径不自动等于完整生成逐位一致；本轮没有任务级质量评价。PP之前的精度边界仍按本轮概率结果呈现。

## 测量与来源

模型初始化、JIT、图捕获、缓存准备、路由诊断、数值测试和profiler均在正式计时之外。CPU KV注册表没有该Qwen混合GDN模型的兼容项，使用原生模型生成的GPU radix前缀与GDN状态，没有CPU KV恢复或伪造缓存。权重来自Guard认证的完整模型CPU bank。

引擎 `{commit}`，不可变源tag `{launch['source_tag']}`，引擎工作树当前clean。模型revision `{cfg['model_revision']}`。实际DeepEP native `{native['path']}`，SHA256 `{native['sha256']}`；wrapper `{cfg['deepep_wrapper']}`。每个run的私有代码和输入均独立冻结。native与optimized之间另有一处仅在untimed microbenchmark保留输入storage的修正，正式推理路径不受该修正影响。

主对照运行为 `{args.native}` 和 `{args.optimized}`。最初`v1`仅准备、未启动；`native v2`因适配函数关键字参数名不匹配在启动阶段失败，没有正式性能数据，已修复并在新目录重跑。所有已启动尝试均保留Guard终态与清理证据；{link('审计记录',ROOT/'analysis/run_audit.json')}；{link('概率验证',ROOT/'analysis/probability_validation.json')}；{link('完整EP数值资格检查',ROOT/'analysis/ep_qualifications.json')}；{link('真实EP路由与内核对照',ROOT/'analysis/ep_routing.json')}。

复现：先用 `prepare.py` 创建全新run，再通过原有Guard监督的 `h100_pp_expert_fragmentation_20261026/launch.py` 启动。重新确认booking与缓存；保留新run目录与冻结输入。离线分析运行 `analyze.py`，再运行 `finalize.py --native <native-run> --optimized <optimized-run>`。不修改已完成run。
'''
    (OUT/'REPORT.md').write_text(report)
    save(OUT/'comparison.json',dict(selection=selected,decode_gain_percent=gain,best_mode=best,
        best_decode_rate=best_rate,EP_vs_PP_percent=advantage,source_commit=commit,
        source_tag=launch['source_tag'],native=native,rows=rows,components=components))
    subprocess.run(['/home/bxb1/conda/bin/python',str(ROOT/'plot.py')],check=True)
    (ROOT/'README.md').write_text('# H100 fair EP2 / PP2 Expert optimization\n\n'+
        link('Measured results and methodology',OUT/'REPORT.md')+' · '+link('Figure',OUT/'fair_ep_pp_comparison.png')+'\n\n'+
        f'Selected complete runs: {args.native}, {args.optimized}. Same shared small-batch Triton GEMM family, unchanged model precision and communication.\n')
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    copied=[]
    def copy(src,rel):
        dst=ARCHIVE/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        assert sha(src)==sha(dst);copied.append(dict(path=str(rel),sha256=sha(dst),bytes=dst.stat().st_size))
    copy(OUT/'REPORT.md',Path('summary.md'))
    for folder in ['analysis','deliverables']:
        for src in sorted((ROOT/folder).glob('*')):
            if src.is_file():copy(src,Path(folder)/src.name)
    for src in sorted(ROOT.glob('*.py')):copy(src,Path('code')/src.name)
    for name in selected:
        run=ROOT/'runs'/name
        for filename in ['config.json','launch_manifest.json','kv_checkpoint_discovery.json']:
            copy(run/filename,Path('runs')/name/filename)
        for src in sorted((run/'code').glob('*.py')):copy(src,Path('runs')/name/'code'/src.name)
    manifest=dict(id=ID,title='H100 fair EP2 and PP2 small-batch Expert execution',status='complete',
        created_at=datetime.now(timezone.utc).isoformat(),selection=selected,source_commit=commit,
        source_tag=launch['source_tag'],model_path=cfg['model_path'],model_revision=cfg['model_revision'],
        node=cfg['node'],job_id=cfg['job_id'],actual_gpu_pair=[0,1],node_lease_gpus=4,
        protocol=cfg['comparison_scope'],precision=cfg['precision'],metrics=rows,
        scientific_gate='raw metrics, exact cache/output counts, shared GEMM paths, numerical qualification, private hashes and cleanup verified',
        decode_gain_percent=gain,best_mode=best,EP_vs_PP_percent=advantage,quality_evaluation_completed=False,
        artifacts=copied,original_runs=[str(ROOT/'runs'/name) for name in selected],
        raw_traces='Referenced in analysis/profile_details.json; original large traces remain in run directories',
        excluded='No credentials, model weights, KV payloads or GPU lease authorization copied')
    save(ARCHIVE/'manifest.json',manifest)
    # Extend the user's existing Q12 report instead of replacing its cohort.
    note=f'''\n<!-- begin fair_ep_pp_20261026 -->
## 双方应用同一类小批量Expert优化

新的5轮/配置对照：EP Decode {gain['ep2']:+.2f}%，PP Decode {gain['pp2']:+.2f}%。本轮较优EP实现 {best_rate['ep2']:.1f} tok/s，较优PP实现 {best_rate['pp2']:.1f} tok/s。复用同一Triton GEMM，EP适配成本包含在E2E；模型/通信/Attention不变。原有单侧PP实验保持独立。

{link('公平对照完整报告',OUT/'REPORT.md')} · {link('图表',OUT/'fair_ep_pp_comparison.png')} · {link('数据CSV',ROOT/'analysis/metrics.csv')} · {link('归档',ARCHIVE/'manifest.json')}
<!-- end fair_ep_pp_20261026 -->\n'''
    q12=ROOT.parent/'experiment_summary/component_reports/Q12.md'
    text=q12.read_text();begin='<!-- begin fair_ep_pp_20261026 -->';end='<!-- end fair_ep_pp_20261026 -->'
    if begin in text:text=text[:text.index(begin)].rstrip()+text[text.index(end)+len(end):]
    q12.write_text(text.rstrip()+'\n'+note)
    index=ROOT.parent/'experiment_summary/README.md'
    text=index.read_text();marker='<!-- fair_ep_pp_20261026 -->'
    lines=text.splitlines()
    for i in range(len(lines)-1,-1,-1):
        if marker in lines[i]:
            start=i-1 if i>0 and not lines[i-1] else i
            end=i+2 if i+1<len(lines) and not lines[i+1] else i+1
            del lines[start:end]
    line=f"{marker} Q12扩展：双方应用相同小批量GEMM优化后的{link('公平EP2/PP2对照',OUT/'REPORT.md')}；EP {gain['ep2']:+.2f}%，PP {gain['pp2']:+.2f}%，独立保留历史cohort。"
    pos=next(i for i,s in enumerate(lines) if '| [Q12' in s)+1
    # Put prose after the table, keeping the Markdown table contiguous.
    while pos<len(lines) and lines[pos].startswith('|'):pos+=1
    lines[pos:pos]=['',line,''];index.write_text('\n'.join(lines)+'\n')
    registry=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    text=registry.read_text();start=f'<!-- begin {ID} -->';end=f'<!-- end {ID} -->'
    entry=f'''{start}
### {ID}

Snapshot: {ARCHIVE}

- Matched full Qwen3.6-35B-A3B BF16 on H100178 NVLink GPUs0/1; EP2 Attention DP2/TP1 versus two-slot PP2 local B4. Cold4096->1 Prefill and real GPU-cached4096+new1->128 Decode; five timed cohorts/configuration/phase.
- Same existing small-batch Triton GEMM family applied to both layouts, with EP GPU indexing adapter and original DeepEP weighted combine. All adapter costs included. Decode EP {gain['ep2']:+.2f}%, PP {gain['pp2']:+.2f}%; best tested EP {best_rate['ep2']:.2f} versus PP {best_rate['pp2']:.2f} output tok/s.
- Exact valid-assignment indexing, real first-layer boundary/empty/all-active tests and actual routed layers pass; atomic per-DP B4 and PP B8 probability probes retained. No task-quality evaluation. Complete raw-metric/cache/count/hash/profile audit and owned cleanup/lease release verified. Earlier startup keyword failure has no timed samples and remains preserved.

Artifacts: report, PNG/PDF scientific figure, full/round/request/component CSVs, probability and routed-input evidence, frozen private adapters, original trace descriptors, manifest and SHA256s. No credentials, weights or KV payload copied.

{end}
'''
    if start in text:text=text[:text.index(start)].rstrip()+'\n\n'+text[text.index(end)+len(end):].lstrip('\n')
    head,tail=text.split('## Saved Experiments',1)
    registry.write_text(head+'## Saved Experiments\n\n'+entry.rstrip()+'\n\n'+tail.lstrip('\n'))
    print(json.dumps(dict(report=str(OUT/'REPORT.md'),archive=str(ARCHIVE),archived_files=len(copied),decode_gain_percent=gain,best_rate=best_rate,EP_vs_PP_percent=advantage),indent=2))


if __name__=='__main__':main()
