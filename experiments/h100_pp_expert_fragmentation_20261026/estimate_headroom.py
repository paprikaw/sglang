"""Offline, explicit-assumption Expert roofline and fixed-schedule PP2 estimate."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'deliverables'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    metrics=list(csv.DictReader((ROOT/'analysis/metrics.csv').open()))
    components=list(csv.DictReader((ROOT/'analysis/components.csv').open()))
    routes=list(csv.DictReader((ROOT/'analysis/routing.csv').open()))
    metric=lambda run,arm:next(r for r in metrics if r['run']==run and r['arm']==arm and r['phase']=='decode')
    expert=lambda run,arm:1000*st.mean(float(r['expert.routed_gemm']) for r in components
        if r['run']==run and r['arm']==arm and r['phase']=='decode')
    current=metric('r5_adaptive_confirm','pp2')
    native=metric('r4_reference_fixed','pp2')
    ep=metric('r4_reference_fixed','ep2')
    t=expert('r5_adaptive_confirm','pp2')
    qual=json.loads((ROOT/'runs/r5_adaptive_confirm/pp2/qualification/pp0.tp0.json').read_text())
    assert qual['device']=='NVIDIA H100 80GB HBM3' and qual['weight_dtype']=='torch.bfloat16'
    resident,gate_up,hidden=qual['expert_shape'];intermediate=gate_up//2
    selected=[r for r in routes if r['run']=='r5_adaptive_confirm' and r['phase']=='D' and r['tokens']=='4']
    assert len(selected)==40
    active=st.mean(int(r['active_experts']) for r in selected)
    assignments=st.mean(int(r['assignments']) for r in selected)
    weight_bytes=3*hidden*intermediate*2
    bytes_per_call=active*weight_bytes
    bandwidth=3.35e12
    dense_bf16_flops=989e12
    useful_flops=6*assignments*hidden*intermediate
    memory_us=bytes_per_call/bandwidth*1e6
    compute_us=useful_flops/dense_bf16_flops*1e6
    calls_per_wave=2*20
    remaining=128-1
    coefficient=remaining*calls_per_wave/1000
    wall=float(current['cohort_ms'])
    tpot=float(current['tpot_ms'])
    rows=[]
    for label,run,arm in [('EP2 current','r4_reference_fixed','ep2'),('PP2 native','r4_reference_fixed','pp2'),('PP2 current','r5_adaptive_confirm','pp2')]:
        m=metric(run,arm)
        rows.append(dict(configuration=label,kind='measured',expert_us=expert(run,arm),
                         tpot_ms=float(m['tpot_ms']),throughput_tok_s=float(m['throughput_tok_s']),
                         cohort_ms=float(m['cohort_ms'])))
    for label,target in [('PP2 target45',45),('PP2 target40',40),('PP2 target35',35),('PP2 ideal HBM model',memory_us)]:
        delta=coefficient*(t-target)
        assert wall>delta
        rows.append(dict(configuration=label,kind='modeled_not_measured',expert_us=target,
            tpot_ms=tpot-calls_per_wave*(t-target)/1000,throughput_tok_s=1024e3/(wall-delta),cohort_ms=wall-delta))
    crossover_throughput=t-(wall-float(ep['cohort_ms']))/coefficient
    crossover_tpot=t-(tpot-float(ep['tpot_ms']))*1000/calls_per_wave
    predicted_native=wall+coefficient*(expert('r4_reference_fixed','pp2')-t)
    backcheck_error=(predicted_native/float(native['cohort_ms'])-1)*100
    result=dict(created_at=datetime.now(timezone.utc).isoformat(),status='offline_estimate_no_new_GPU_measurement',
        interpretation='NVIDIA GEMM optimization guide, not GLM-model deployment practice',
        scope='Qwen3.6-35B-A3B BF16, same two H100 GPUs, B8 global/two B4 slots;cached4096+new1->output128 Decode',
        geometry=dict(hidden=hidden,intermediate=intermediate,resident_experts=resident,mean_active_experts=active,
                      local_tokens=4,assignments=assignments,bytes_per_expert=weight_bytes),
        roofline=dict(peak_HBM_bytes_s=bandwidth,approx_dense_BF16_flops_s=dense_bf16_flops,
                      active_weight_bytes=bytes_per_call,useful_flops=useful_flops,
                      arithmetic_intensity_FLOP_per_byte=useful_flops/bytes_per_call,
                      ideal_HBM_read_us=memory_us,ideal_dense_compute_us=compute_us,
                      engineered_target_assumptions=dict(effective_bandwidth_fraction=[.60,.75],extra_execution_us=[5,10],
                        resulting_us=[memory_us/.75+5,memory_us/.60+10],proposed_target_us=[35,45])),
        fixed_schedule_model=dict(calls_per_gpu_per_decode_wave=calls_per_wave,remaining_steps=remaining,
            formula='cohort_new_ms=cohort_current_ms-127*40*(expert_current_us-expert_target_us)/1000',
            assumptions=['only Expert GEMM changes','TTFT and non-Expert costs unchanged',
                'balanced PP stages;two B4 slots; unchanged critical-path contribution per wave',
                'active expert count is a routed sample, not full-trace DRAM traffic',
                'HBM model reads active weights once;isolated hot-cache benchmarks have a different roofline'],
            throughput_EP_crossover_expert_us=crossover_throughput,tpot_EP_crossover_expert_us=crossover_tpot,
            backcheck_native_cohort_predicted_ms=predicted_native,backcheck_native_cohort_actual_ms=float(native['cohort_ms']),
            backcheck_error_percent=backcheck_error,
            validation='not a validated performance predictor;backcheck includes other kernel/layout/trajectory changes'),
        comparisons=rows,
        sources=[dict(url='https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html',
                      supports='arithmetic intensity;memory-vs-math limit;tile efficiency and parallelism'),
                 dict(url='https://www.nvidia.com/en-us/data-center/h100/',supports='H100 SXM80GB 3.35TB/s;sparse BF161979TF/s, dense approximately989TF/s')],
        source_files=[dict(path=str(ROOT/'analysis'/name),sha256=sha(ROOT/'analysis'/name))
                      for name in ('metrics.csv','components.csv','routing.csv')],
        caution='EP column is the current measured implementation;not an equally optimized EP theoretical limit.35-45us is our engineering hypothesis,not a NVIDIA guarantee.')
    (OUT/'headroom_estimate.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'headroom_comparison.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    comparison='\n'.join(f"| {r['configuration']} | {'实测' if r['kind']=='measured' else '估算'} | {r['expert_us']:.2f} | {r['tpot_ms']:.3f} | {r['throughput_tok_s']:.1f} |" for r in rows)
    report=f'''# PP2 Expert 优化空间估算：NVIDIA GEMM 原则与现有实测

本页是离线计算，没有启动新GPU实验。将用户的“GLM best practice”理解为 NVIDIA GEMM 矩阵乘优化指南。沿用现有 Qwen3.6-35B-A3B BF16；仅讨论本地 B4 的 Decode，不使用 GLM 模型或更低精度的性能替换当前模型。

结论：当前 Expert GEMM 57.62µs/实际层调用，建议下一工程目标40µs，35µs为更进取目标。活跃权重每次从HBM读取一遍、达到峰值带宽的理想模型约{memory_us:.2f}µs。端到端吞吐达到当前EP2附近具有可能性，现有计算不足以保证胜出。

## 计算依据

本模型hidden={hidden}，每个Expert intermediate={intermediate}，SwiGLU含gate/up/down三个矩阵。每个Expert的BF16权重为：

`3 × {hidden} × {intermediate} × 2 = {weight_bytes:,} bytes = 6 MiB`。

最终r5真实路由样本40层，本地4 token、top8，共32次配对，平均{active:.2f}个不同的活跃Expert。活跃权重共`{active:.2f} × 6 MiB = {bytes_per_call/1e6:.2f} MB`。H100 SXM80GB官方峰值HBM带宽3.35TB/s，因此理想读取时间：

`{bytes_per_call/1e6:.2f} MB / 3.35 TB/s = {memory_us:.2f} µs`。

有效计算量为`6 × 32 × {hidden} × {intermediate} = {useful_flops/1e9:.3f} GFLOP`；即使按约989TF/s的dense BF16计算，算力项仅{compute_us:.3f}µs。权重主导的算术强度约{useful_flops/bytes_per_call:.2f}FLOP/byte，远低于GPU的计算/带宽比例。因此不能直接用Tensor Core峰值推算加速倍数。

这是理想HBM模型：不含元数据、激活搬运、tile浪费及其他执行成本；L2驻留情况也会影响读取量，重复单层的热缓存microbenchmark不是同一个HBM条件。它不是对实际kernel时延的严格通用下界。

工程目标采用我们的假设：有效HBM达到峰值的60%–75%，另有5–10µs执行开销，可得{memory_us/.75+5:.1f}–{memory_us/.60+10:.1f}µs，故取35–45µs作为待实测的目标范围。这个效率比例和开销范围不是NVIDIA公布的保证值。已有单层T4小tile测试约33–47µs提供实现线索，不能直接替代完整模型的57.62µs。

## 与当前EP/PP对照

Expert单位µs/GPU/实际层调用；TPOT单位ms；吞吐为当前8请求、缓存4096、新增1、输出128的output token/s。测量和估算明确分列。

| 配置 | 类型 | Expert GEMM ↓ | TPOT ↓ | 吞吐 ↑ |
|---|---|---:|---:|---:|
{comparison}

## 端到端换算假设

PP2每个stage20层，两个B4槽位；每个全局Decode wave，每卡执行约40次Expert调用。在流水调度、TTFT、Attention和其他成本不变且这些Expert收益均能作用于流水关键路径的假设下：

`T_cohort(target) = 1271.16ms − 127 × 40 × (57.6156 − target_Expert_us)/1000`。

`吞吐 = 8 × 128 × 1000 / T_cohort(target)`。

这不是把不同GPU的profile时间直接相加。它是固定调度的反事实模型，40来自每卡两个槽位各20层；重叠、等待、负载不平衡改变时，换算也会改变。回算原生PP时，该模型得到{predicted_native:.2f}ms，实测为{float(native['cohort_ms']):.2f}ms，误差{backcheck_error:.2f}%；原生到优化同时改变了其他kernel/布局/轨迹，说明不能把这个模型当作已验证预测器。40µs目标对应的吞吐优势小于该误差量级。

在该模型中，当前128输出测试的吞吐追平EP2需要Expert约≤{crossover_throughput:.2f}µs；TPOT追平EP2则需要≤{crossover_tpot:.2f}µs。两者不同，是因为PP这组数据的TTFT更低；延长生成后，首token差异会逐渐摊薄。

因此，把40µs作为下一目标比较合理：对应Expert比当前再快约{(1-40/t)*100:.1f}%，完整模型粗算约867 tok/s，接近当前EP2的863。达到35µs粗算886 tok/s；即使达到21.4µs的理想HBM模型，完整模型也只是约942 tok/s，并不是整体获得2.7倍加速。EP自身也有优化空间，此处只与已测EP对比。现有BF16不同执行路径的生成差异与质量边界继续适用。

## 来源

- [NVIDIA GEMM指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)：算术强度、内存/算力限制、tile大小与并行度取舍。
- [NVIDIA H100官方规格](https://www.nvidia.com/en-us/data-center/h100/)：SXM80GB、3.35TB/s；BF161979TF/s带稀疏标记，dense约989TF/s。
- [原始完整报告]({OUT/'REPORT.md'})；[计算参数和来源哈希]({OUT/'headroom_estimate.json'})；[对照CSV]({OUT/'headroom_comparison.csv'})。
'''
    (OUT/'HEADROOM_ESTIMATE.md').write_text(report)
    print(json.dumps(dict(roofline=result['roofline'],comparisons=rows,crossover_throughput_us=crossover_throughput,
                         crossover_tpot_us=crossover_tpot,backcheck_error_percent=backcheck_error),indent=2))


if __name__=='__main__':
    main()
