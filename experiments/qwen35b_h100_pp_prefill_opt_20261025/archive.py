"""Save only small, non-secret summaries and update the existing experiment index."""
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from consolidate import ROOT,PERF,PROFILE,digest,load

ID='qwen35b_h100_pp_prefill_optimization_20261025'
SNAP=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')/ID
INDEX=ROOT.parent/'experiment_summary'
REGISTRY=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')


def replace_block(path,begin,end,block,anchor=None):
    text=path.read_text()
    if begin in text:
        start=text.index(begin);finish=text.index(end,start)+len(end)
        text=text[:start]+block+text[finish:]
    elif anchor:
        assert anchor in text;text=text.replace(anchor,block+'\n\n'+anchor,1)
    else:text=text.rstrip()+'\n\n'+block+'\n'
    path.write_text(text)


def main():
    validation=load(ROOT/'analysis/validation.json');assert validation['passed']
    headlines=load(ROOT/'analysis/headlines.json');SNAP.mkdir(exist_ok=True)
    copied=[]
    for source,target in [(ROOT/'RESULTS.md',SNAP/'summary.md')]+[(ROOT/'analysis'/n,SNAP/n) for n in (
            'full_metrics.csv','request_metrics.csv','stage_metrics.csv','validation.json','headlines.json')]:
        shutil.copyfile(source,target);copied.append(target)
    for name in ('chunk_comparison.png','chunk_comparison.svg','pipeline_timeline.png','pipeline_timeline.svg'):
        target=SNAP/name;shutil.copyfile(ROOT/'runs'/PERF[1]/'analysis'/name,target);copied.append(target)
    sources=SNAP/'code';sources.mkdir(exist_ok=True)
    # No runtime config, access key, lease token, tensor bank, or large native/trace is copied.
    for source in ROOT.glob('*.py'):
        target=sources/source.name;shutil.copyfile(source,target);copied.append(target)
    ready_source=ROOT/'runs'/PROFILE/'code/pp_ready_send.py'
    shutil.copyfile(ready_source,sources/'pp_ready_send.py');copied.append(sources/'pp_ready_send.py')
    cfg=load(ROOT/'runs'/PERF[2]/'config.json');engine=Path(cfg['engine_root'])
    def git(*args):return subprocess.check_output(['git','-C',str(engine),*args],text=True).strip()
    commit=git('rev-parse','HEAD');assert commit==cfg['engine_commit']
    dirty=git('status','--porcelain','--untracked-files=no').splitlines()
    frozen=[]
    for name in (*PERF,PROFILE):
        run=ROOT/'runs'/name;spec=load(run/'dashboard_spec.json')
        frozen.append(dict(run=name,spec=digest(run/'dashboard_spec.json'),
            frozen_inputs=spec['frozen_inputs'],original_run=str(run)))
    manifest=dict(id=ID,title='H100 PP2 Prefill: native chunk budgets, bounded admission, and producer-ready send',
        created_at=datetime.now(timezone.utc).isoformat(),status='complete',
        implementation=dict(engine='SGLang',worktree=str(engine),commit=commit,branch=git('rev-parse','--abbrev-ref','HEAD'),
            source_tag='experiment/sglang/glm47-ep16-canonical-profile-20261004-cpu-validated',
            dirty_tracked_files=dirty,old_implementation=False,
            experiment_private_adapters=True,shared_engine_modified=False,
            deepep_wrapper=cfg['deepep_wrapper'],deepep_native=cfg['deepep_native'],
            qualified_guard_scripts=cfg['guard_scripts'],frozen_runs=frozen),
        model=dict(name='Qwen3.6-35B-A3B',path=cfg['model_path'],revision=cfg['model_revision'],precision='BF16',full_layers=40),
        deployment=dict(node=cfg['node'],slurm_job=cfg['job_id'],gpu_type='H100 80GB',
            registered_gpus=4,pp_gpu_pair=[2,3],ep_reference_pair=[0,1],topology='pair-local NV6',
            gateway_modified=False,ephemeral_owned_servers_stopped=True),
        workload=dict(request_order='concurrent offered cohorts; no pre-timing all-request admission barrier',
            cases=load(ROOT/'runs'/PERF[1]/'config.json')['optimization_cases'],output_tokens=1,
            prefixes='real model KV and GDN states already on GPU',cpu_hicache=False,synthetic_kv=False,
            preparation_and_warmup_excluded=True,scheduling_and_synchronization_included=True,
            formal_cohorts=46,request_measurements=validation['request_rows'],
            pp_abba=True,chunk_order=[4096,2048,1024,8192,8192,1024,2048,4096],samples_per_policy_case=2,
            profile_only_cohorts=2,profile_followup_not_performance=True),
        parameters=dict(attention_backend='FA3',linear_attention_backend='Triton',moe_runner='BF16 DeepGEMM',
            pp=dict(tp=1,ep=1,dp=1,pp=2,layers_per_stage=20,loop_slots=2,async_batch_depth=0,overlap=False),
            ep_reference=dict(tp=2,ep=2,dp=2,pp=1,attention_tp=1,overlap=True),
            decode_graph=True,prefill_graph=False,page_size=64,mem_fraction_static=.94,max_running_requests=32,
            linear_states='PP164 per stage, EP168 total; max states per path1'),
        key_results=headlines,
        validation=dict(formal_guard_original_postchecks_passed=True,profile_followup_guard_postcheck_passed=False,
            profile_followup_failure='controller counted metric files, not two completed profile manifests',owned_cleanup_verified=True,
            leases_released=True,formal_cold_outputs_identical=True,formal_long_ready_outputs_identical=True,
            cpu_tests_passed=11,raw_validation=str(ROOT/'analysis/validation.json')),
        caveats=['Only two formal cohorts per mode; descriptive P99 is not stable service tail.',
            'No Decode or TPOT result in this turn.',
            'No new optimized-EP2 versus optimized-PP2 ranking; EP references are noisy.',
            'Native chunk8192 benefits cold8 only, not the long cached case.',
            'R4 final candidate rank1 trace was truncated by early cleanup; original evidence retained. R5 joins independent complete two-stage profiles, but its original Guard postcheck fails because controller final progress counted zero metrics instead of two profile manifests; cache and cleanup pass. The failed postcheck is preserved.',
            'Prepared chunk_r2 was never launched and is not a result.',
            'Generic Guard prelaunch model-profile rejection was resolved using the unchanged Qwen-qualified frozen Guard, not by relaxing cache requirements.'],
        artifacts=dict(summary=str(SNAP/'summary.md'),metrics=str(SNAP/'full_metrics.csv'),
            stage_metrics=str(SNAP/'stage_metrics.csv'),request_metrics=str(SNAP/'request_metrics.csv'),
            original_root=str(ROOT),plots=[str(SNAP/n) for n in ('chunk_comparison.png','pipeline_timeline.png')]))
    (SNAP/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');copied.append(SNAP/'manifest.json')
    (SNAP/'SHA256SUMS.json').write_text(json.dumps([digest(p) for p in copied],indent=2)+'\n')
    ca=headlines['cold_baseline'];cb=headlines['cold_candidate'];la=headlines['long_baseline'];lb=headlines['long_candidate']
    short=f'''# Q05：H100 PP2 Prefill 局部优化

Qwen3.6-35B-A3B BF16，节点178，PP2固定GPU2／3、各20层、两个原生循环槽位。**已经缓存的历史**在GPU，前缀准备／形状预热不计时，必要调度和同步计时。每条只生成1token，所以端到端延迟等于TTFT，没有TPOT。

| 请求负载 | 已测优化 | 新输入 token/s ↑ | 整批完成，毫秒 ↓ | 平均 TTFT，毫秒 ↓ |
|---|---|---:|---:|---:|
| 8请求；没有旧缓存；每条新增4096token | 原生chunk预算4096→8192 | {ca['throughput_new_tokens_s']:.1f}→**{cb['throughput_new_tokens_s']:.1f}（+{headlines['cold_chunk_gain_percent']:.2f}%）** | {ca['mean_cohort_duration_ms']:.2f}→{cb['mean_cohort_duration_ms']:.2f} | {ca['ttft_ms_mean']:.2f}→{cb['ttft_ms_mean']:.2f} |
| 32请求；每条GPU已缓存89,984token；每条新增512token | chunk仍4096；数据就绪后才发送 | {la['throughput_new_tokens_s']:.1f}→**{lb['throughput_new_tokens_s']:.1f}（+{headlines['long_ready_gain_percent']:.2f}%）** | {la['mean_cohort_duration_ms']:.2f}→{lb['mean_cohort_duration_ms']:.2f} | {la['ttft_ms_mean']:.2f}→{lb['ttft_ms_mean']:.2f} |

`chunk预算`是一次模型forward允许的**全部新输入token总数**，不是请求数或每请求历史长度。发送就绪适配在原生发送前等待生产CUDA事件，等待仍计时；没有改精度、模型计算或物理链路。

两个收益分别验收，不能相乘。长缓存8192反而更慢；2048／1024使forward增多且Expert累计计算上升；最多5ms的输入合批也未获益。冷Prefill八个正式cohort及长缓存四个正式cohort分别逐请求输出一致；每种策略只有两批，不宣称普遍最优或稳定P99，也不据此宣称已胜过充分调优的EP2。

完整报告包含全部23组mean／median／P99、46个正式cohort和逐卡Attention／routed Expert／shared Expert／NCCL数据。R4末阶段最后一份trace清理过早，原样保留；R5只补profile、等待两阶段完整落盘，不替换正式性能。R5原始Guard状态因控制器误计0／2而失败；两阶段导出、缓存和清理单独验证通过，原失败证据保留。

[完整简报与全部表]({ROOT}/RESULTS.md) · [完整指标]({ROOT}/analysis/full_metrics.csv) · [逐卡分阶段]({ROOT}/analysis/stage_metrics.csv) · [归档及版本]({SNAP}/manifest.json)

![四种chunk的吞吐与延迟]({ROOT}/runs/{PERF[1]}/analysis/chunk_comparison.png)
'''
    (INDEX/'component_reports/Q05.md').write_text(short)
    begin='<!-- BEGIN Q05 H100 PREFILL OPT -->';end='<!-- END Q05 H100 PREFILL OPT -->'
    row=f'''{begin}
| [Q05：H100 PP2 Prefill 局部优化]({INDEX}/component_reports/Q05.md) | 同Q04的35B BF16；178节点GPU2／3。8请求无旧缓存、每条新增4096；另有32请求、各GPU已缓存89,984、各新增512；输出1。 | 冷Prefill chunk4096→8192：吞吐+{headlines['cold_chunk_gain_percent']:.2f}%；长缓存发送就绪：+{headlines['long_ready_gain_percent']:.2f}%。短／长缓存预算扫描及有界合批负结果均保留；不外推Decode。 | 46正式cohort／23组全指标，独立两阶段GPU trace、TTFT／端到端mean／median／P99和原始哈希；不与EP8规模混排。 |
{end}'''
    index=INDEX/'README.md';text=index.read_text();new_row=row.splitlines()[1]
    lines=text.splitlines();old=next((line for line in lines if line.startswith('| [Q05：')),None)
    if old:text=text.replace(old,new_row,1)
    else:
        anchor='\n\n<!-- END Q01 QWEN EP2 PP2 -->';assert anchor in text
        text=text.replace(anchor,'\n'+new_row+anchor,1)
    index.write_text(text)
    opt=f'''{begin}
### 5.4 H100 35B：按负载选择 Prefill chunk，并验收发送就绪

同节点178、PP2 GPU2／3，Qwen3.6-35B-A3B BF16。8请求各无旧缓存、新输入4096时，chunk4096→8192使吞吐{ca['throughput_new_tokens_s']:.1f}→{cb['throughput_new_tokens_s']:.1f}（+{headlines['cold_chunk_gain_percent']:.2f}%），整批{ca['mean_cohort_duration_ms']:.2f}→{cb['mean_cohort_duration_ms']:.2f}ms；8个正式cohort输出逐请求一致。不是增大请求数，也没有少算输入。

32请求各GPU已缓存89,984、新输入512时，保留chunk4096，复用A100的“生产端数据就绪后再发布发送通知”：吞吐{la['throughput_new_tokens_s']:.1f}→{lb['throughput_new_tokens_s']:.1f}（+{headlines['long_ready_gain_percent']:.2f}%），TTFT{la['ttft_ms_mean']:.2f}→{lb['ttft_ms_mean']:.2f}ms；四个相同输入cohort输出一致。候选仍为显式启用的实验私有适配。

长缓存不能照搬8192；2048／1024也更慢。独立profile显示2048把forward从5次增至9次，Attention累计量基本不变而Expert成本上升，NCCL驻留更短不等于端到端更快。有界5ms合批也未采用。每策略只有两次，未重测充分调优EP2；本轮没有Decode或TPOT结论。[全部数据、负结果和两阶段profile]({INDEX}/component_reports/Q05.md)。
{end}'''
    replace_block(INDEX/'OPTIMIZATIONS.md',begin,end,opt,anchor='## 6. 权重／KV Bank：缩短准备时间，也修正测量公平性')
    entry=f'''<!-- begin {ID} -->
### {ID}

Snapshot: {SNAP}

- Complete H100178 Qwen3.6-35B-A3B BF16 Prefill optimization; native PP2, GPU2/3,20 layers/stage,two loop slots. Required canonical CPU weight bank reused; real KV/GDN prefixes already GPU-hot. No CPU HiCache, synthetic KV, Decode, or TPOT measurement.
- Cold8 x4096 new input: same-process native chunk4096 to8192 yields {ca['throughput_new_tokens_s']:.1f} to{cb['throughput_new_tokens_s']:.1f} new-token/s (+{headlines['cold_chunk_gain_percent']:.2f}%), cohort{ca['mean_cohort_duration_ms']:.2f} to{cb['mean_cohort_duration_ms']:.2f}ms. All8 formal cold cohorts have identical per-request output IDs.
- Long32 x89984 cached +512 new: native chunk4096 plus unchanged A100 producer-ready metadata adapter yields {la['throughput_new_tokens_s']:.1f} to{lb['throughput_new_tokens_s']:.1f} (+{headlines['long_ready_gain_percent']:.2f}%), TTFT{la['ttft_ms_mean']:.2f} to{lb['ttft_ms_mean']:.2f}ms; all4 formal identical-input outputs agree. Benefits are separate, not multiplied.
- Four chunk budgets and bounded5ms coalescing negatives preserved: smaller chunks increase forwards and Expert cost; long8192 is slower. 46 formal cohorts,23 grouped metric rows,1136 request timings; only2 cohorts per mode, noisy EP2 references are not a best-versus-best ranking.
- Formal Guard postchecks/cache receipts,owned cleanup,lease release pass. R4 last candidate rank1 trace was not joined before cleanup; raw partial evidence retained. Profile-onlyR5 joins both complete GPU exports/CRCs for both policies and does not replace formal throughput; its original postcheck remains failed because final progress incorrectly counts zero metric files instead of two profile manifests, while cache and cleanup pass. 11 CPU tests pass. Shared engine/gateway untouched.

Artifacts: manifest,all mean/median/P99 metrics,per-request and per-stage CSVs,scientific bar/timeline plots,source hashes and lightweight adapters/analyzers/tests. No credentials,weights,KV bytes,large natives or raw traces copied.

<!-- end {ID} -->'''
    replace_block(REGISTRY,f'<!-- begin {ID} -->',f'<!-- end {ID} -->',entry,anchor='<!-- begin single_layer_ep4_ep8_ce_rdma_20261024_r1 -->')
    # Keep the plan as history while making the result entry immediately visible.
    readme=ROOT/'README.md';text=readme.read_text()
    lead=f'''# H100 PP2 Prefill：已完成的局部优化

[完整结果、全部指标和归因](RESULTS.md) · [实验总览中的Q05]({INDEX}/component_reports/Q05.md) · [版本归档]({SNAP}/manifest.json)

- 8请求、无旧缓存、各新增4096token：原生chunk8192相对4096吞吐 **+{headlines['cold_chunk_gain_percent']:.2f}%**，输出一致。
- 32请求、各GPU已缓存89,984token、各新增512token：保留chunk4096，发送就绪适配 **+{headlines['long_ready_gain_percent']:.2f}%**，输出一致。
- 有界合批、缩小chunk及长缓存强行放大chunk均未采用。没有Decode／TPOT提速结论；改动不自动替换共享引擎。

源码：`chunk_budget.py`、冻结运行中的`pp_ready_send.py`、`coalesce.py`；分析由`analyze.py`、`inspect_prefill.py`、`consolidate.py`重算。`profile_ready.py`在清理前等待两阶段gzip／JSON完整写盘。CPU测试11项通过。

## 原始设计记录
'''
    if '## 原始设计记录' in text:text=text.split('## 原始设计记录',1)[1].lstrip()
    readme.write_text(lead+'\n'+text)
    print(json.dumps(dict(snapshot=str(SNAP),summary_entry=str(INDEX/'component_reports/Q05.md'),copied_files=len(copied))))


if __name__=='__main__':main()
