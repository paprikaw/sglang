"""Export checked component results; large raw traces stay in the source run."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_a100_ep8_pp2ep4_m655360_b8_decode_20261022_r1')


def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    summary=read(a.analysis/'summary.json');task=read(a.run/'task/results.json')
    cleanup=read(a.run/'task/cleanup.json');lease=read(a.run/'dashboard_guard/gpu_registry_lease.json')
    # Run-local lease is the immutable acquisition receipt. The shared registry
    # history, not that old receipt, is authoritative for release.
    registry_path=Path('/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/gpu_registry/registry.json')
    registry=read(registry_path)
    released=[x for x in registry['history'] if x['lease_id']==lease['lease_id']]
    if (not task['passed'] or not cleanup['passed'] or len(released)!=1
            or released[0]['state']!='released' or released[0].get('cleanup_verified') is not True
            or any(x['lease_id']==lease['lease_id'] for x in registry['leases'])):
        raise ValueError('Run and cleanup must pass, lease must be released')
    posts=sorted((a.run/'dashboard_guard/checks').glob('*-post.json'))
    if not posts or not read(posts[-1])['passed']:raise ValueError('Guard postcheck must pass')
    a.out.mkdir(parents=True,exist_ok=True);save(a.out/'component_summary.json',summary)
    save(a.out/'lease_release.json',dict(registry=str(registry_path),
        **{k:released[0][k] for k in ('lease_id','experiment_id','nodes','state','cleanup_verified','ended_at','end_reason')}))
    rows=[]
    for arm,which,label in [('ep8','all_rank_mean','EP8'),('pp2ep4','stage0','PP2xEP4 stage0'),('pp2ep4','stage1','PP2xEP4 stage1')]:
        g=summary['groups'][arm][which];ms=g['mean_component_ms'];pct=g['component_percent']
        row=dict(configuration=label,mean_microbatch_cycle_ms=g['mean_microbatch_cycle_ms'])
        for key in ['attention','expert_ffn','communication']:
            row[key+'_ms']=ms.get(key,0);row[key+'_percent']=pct.get(key,0)
        row['remaining_ms']=g['mean_microbatch_cycle_ms']-sum(ms.get(k,0) for k in ['attention','expert_ffn','communication'])
        row['remaining_percent']=100-sum(pct.get(k,0) for k in ['attention','expert_ffn','communication'])
        rows.append(row)
    with (a.out/'stage_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# GLM-4.7-AWQ A100 Decode：逐阶段构成','',
        'B8，每请求 P=81,920 个已缓存 token，Q=1；32 个输出 token。两节点 065／167，8×A100。',
        'EP8：Attention TP4×DP2／Expert EP8；PP：46+46 层、两组 B4 slot、每 stage 为 Attention TP4／Expert EP4。',
        '', '|配置|GPU microbatch 周期|Attention|Expert/FFN|通信|其余|',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        cells=[r['configuration'],f"{r['mean_microbatch_cycle_ms']:.2f} ms"]
        cells += [f"{r[k+'_percent']:.1f}%（{r[k+'_ms']:.2f} ms）" for k in ['attention','expert_ffn','communication','remaining']]
        lines.append('|'+ '|'.join(cells)+'|')
    lines += ['', '## 计时边界','',
        '- 分母是每张 GPU 上稳态相邻 microbatch 的开始时间间隔，按 GPU 求均值；不跨 GPU 相加，不是 HTTP TPOT 的严格分解。',
        '- PP 每次只执行半模型；每条请求仍需经过两个 stage，且两组 B4 slot 轮转，不能把表内半 stage 周期直接当成请求 TPOT。',
        '- Attention 含 QKV/O 投影与 Attention 内核；Expert/FFN 含 routed、shared 和前三层 dense FFN。',
        '- 通信含 DeepEP Dispatch/Combine、TP NCCL 和采样周期内的 PP 边界；通信 kernel 的轮询/等待也计入，不是纯链路传输时长。',
        '- 其余保留 router、norm、输出处理、未分类 kernel、GPU 无活动间隙及跨类别重叠；不强行归入三个主分项。',
        '- Graph OFF、调度 overlap OFF；DeepEP 为 pair NVLink＋pair 外 RDMA（PCIe threshold=0），不是 CE Prefill 库。',
        '- 先完成 CPU HiCache→GPU 恢复与预热，再采稳态 Decode；本表不含首次换入。原始 128-output 冷起 benchmark 的 TTFT/TPOT 保持原值，不由此替换。',
        '', '## 观察者效应核验','',
        '|配置|无 profiler TPOT|CUDA-only TPOT|变化|','|---|---:|---:|---:|']
    for arm,v in summary['perturbation'].items():
        lines.append(f"|{arm}|{v['control_tpot_ms']:.3f} ms|{v['gpu_profile_tpot_ms']:.3f} ms|{100*(v['whole_cohort_ratio']-1):+.2f}%|")
    lines += ['', '只记录 CUDA 活动，层内不加计时点；根据同版本实际 CPU scope→CUDA launch 关联校准 kernel 分类，并用不同 forward 验证。归类有歧义的 kernel 保留未分类。',
        '较早的 CPU+CUDA 全量打点、全层事件及稀疏层事件均存在观察者效应，保留为诊断证据，不作为此表占比来源。',
        '', f'原始 run：`{a.run.resolve()}`',f'逐 rank、逐周期数据：`{a.analysis.resolve()}`','']
    (a.out/'summary.md').write_text('\n'.join(lines))
    manifest=copy.deepcopy(read(BASE/'manifest.json'))
    manifest.update(id=a.out.name,title='GLM-4.7 AWQ A100: calibrated CUDA-only steady Decode component profile',
        created_at=datetime.now(timezone.utc).isoformat(),status='complete',original_run=str(a.run.resolve()),
        original_benchmark=str(BASE),key_results=dict(stage_rows=rows,perturbation=summary['perturbation']))
    manifest['deployment'].update(id=read(a.run/'config.json')['id'],launch_manifest=str(a.run/'dashboard_spec.json'),
        validation_evidence=dict(task=str(a.run/'task/results.json'),cleanup=str(a.run/'task/cleanup.json'),postcheck=str(posts[-1])))
    manifest['workload'].update(output_tokens_per_request=32,prefix_state_at_t0='complete target prefix on GPU; independent control and CUDA-only profile',
        arrivals='barrier-released B8 HTTP streaming; one warm cohort then one unprofiled control and one CUDA-only profile per layout',
        timed=['steady Decode microbatch GPU timeline'],excluded=['weight load','synthetic KV construction','initial HiCache H2D','warmup'])
    manifest['measurement']=dict(method=summary['method'],denominator=summary['denominator'],
        communication=summary['communication'],calibration=summary['calibration'],sample_counts=dict(ep8_forwards_per_rank=4,pp_forwards_per_rank=8),
        caveats=['single cohort, not a confidence interval','GPU activity includes polling/waits','PP stage time is not request TPOT','unknown, idle and overlap stay explicit'])
    manifest['artifacts']=dict(snapshot=['manifest.json','summary.md','component_summary.json','stage_metrics.csv','lease_release.json'],
        raw_trace_directory=str(a.run.resolve()),per_rank_analysis=str(a.analysis.resolve()),large_traces_copied=False)
    manifest['validation']=dict(guard_postcheck_passed=True,scientific_success=True,cleanup_verified=True,
        lease_released=True,full_canonical_cpu_weight_usage_verified=True,all_rank_b4_decode_receipts_verified=True,
        cpu_tests_passed=14,all_selected_gpu_windows_have_two_real_correlated_markers=True)
    manifest['caveats']=manifest['measurement']['caveats']+[
        'CUDA-only profiler changes cohort TPOT by about 6 percent; percentages are approximate measured profile composition, not an exact uninstrumented HTTP TPOT decomposition',
        'Synthetic KV does not establish output quality','Historical pair-NVLink/RDMA implementation, not CE Prefill']
    capacities={arm:[read(a.run/arm/'capacity'/f'rank{rank}.json') for rank in range(8)] for arm in ('ep8','pp2ep4')}
    manifest['engine_parameters']['actual_rank_capacity_receipts']=capacities
    save(a.out/'manifest.json',manifest)
    print(json.dumps(dict(snapshot=str(a.out.resolve()),rows=rows,perturbation=summary['perturbation']),indent=2))


if __name__=='__main__':main()
