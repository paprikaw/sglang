"""Inspect temporal kernel overlap in saved Prefill traces; no GPU execution."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re
import statistics as st

import analyze as a

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/qwen35b_h100_prefill_optimized_r2_20261027'


def merged(spans):
    result=[]
    for lo,hi in sorted(spans):
        if not result or lo>result[-1][1]: result.append([lo,hi])
        else:result[-1][1]=max(result[-1][1],hi)
    return result


def intersection(left,right):
    i=j=0;total=0.
    while i<len(left) and j<len(right):
        total+=max(0.,min(left[i][1],right[j][1])-max(left[i][0],right[j][0]))
        if left[i][1]<=right[j][1]:i+=1
        else:j+=1
    return total


def inspect(desc,tools):
    path=Path(desc['path']);assert a.sha(path)==desc['sha256']
    with gzip.open(path,'rt') as f:
        events=[e for e in json.load(f)['traceEvents'] if e.get('ph')=='X' and e.get('dur',0)>0]
    anns=[e for e in events if e.get('cat')=='user_annotation']
    si=tools.IntervalIndex([e for e in anns if e['name'].startswith('step[')])
    layers=[]
    for e in anns:
        if re.fullmatch(r'component\.layer\.\d+',e['name']):
            s=si.find(e['ts'],e.get('tid'))
            if s and ('EXTEND' in s['name'] or 'PREFILL' in s['name']):layers.append(e)
    assert len(layers)==640
    li=tools.IntervalIndex(layers)
    ci=tools.IntervalIndex([e for e in anns if e['name'].startswith('component.') and not e['name'].startswith('component.layer.')])
    runtime={e.get('args',{}).get('correlation'):e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and e.get('args',{}).get('correlation') is not None}
    spans=defaultdict(list);streams=defaultdict(set)
    for e in events:
        if e.get('cat')!='kernel':continue
        launch=runtime.get(e.get('args',{}).get('correlation'))
        if not launch or not li.find(launch['ts'],launch.get('tid')):continue
        scope=ci.find(launch['ts'],launch.get('tid'))
        label=tools.label(scope) if scope else 'other'
        if label=='expert.routed' and (('deep_gemm::' in e['name'] and 'gemm' in e['name'].lower()) or 'fused_moe_kernel' in e['name']):bucket='routed_gemm'
        elif label in ('communication.dispatch','communication.combine'):bucket=label.split('.')[-1]
        elif label in ('attention.full','attention.linear'):bucket='attention'
        elif label=='expert.shared':bucket='shared_expert'
        else:bucket='other'
        spans[bucket].append((e['ts'],e['ts']+e['dur']))
        streams[bucket].add(e.get('args',{}).get('stream'))
    spans['dispatch_combine']=spans['dispatch']+spans['combine']
    unions={k:merged(v) for k,v in spans.items()}
    totals={k:sum(hi-lo for lo,hi in v)/len(layers) for k,v in unions.items()}
    overlaps={k:intersection(unions['dispatch_combine'],unions[k])/len(layers)
        for k in ('routed_gemm','shared_expert','attention')}
    assert totals['routed_gemm']>0 and totals['dispatch_combine']>0
    return dict(trace=desc,actual_prefill_layer_calls=len(layers),
        kernel_union_us_per_actual_layer=totals,
        communication_intersection_us_per_actual_layer=overlaps,
        communication_overlap_pct={k:100*v/totals['dispatch_combine'] for k,v in overlaps.items()},
        cuda_streams={k:sorted(v,key=str) for k,v in streams.items()},
        method='Trace timestamp intersections on one GPU, across all attributed Prefill layers; host-launch attribution identical to main analyzer. This is temporal overlap, not inferred critical-path savings or an overlap-enabled benchmark.')


def main():
    tools=a.load('overlap_gpu_tools',ROOT.parent/'pp_ep_steady_decode_20261026/analyze.py')
    manifest=RUN/'ep2/workloads/b32_q4096_repeated/traces/p0/manifest.json'
    rows=[inspect(d,tools) for d in a.read(manifest)['files']]
    average={k:st.mean(r['kernel_union_us_per_actual_layer'][k] for r in rows) for k in rows[0]['kernel_union_us_per_actual_layer']}
    reference=next(r for r in a.read(ROOT/'analysis'/RUN.name/'results.json')['components'] if r['arm']=='ep2' and r['workload']=='b32_q4096_repeated')
    assert abs(average['routed_gemm']-reference['expert_gemm_us_per_layer'])<.001
    assert abs(average['dispatch_combine']-reference['dispatch_combine_us_per_layer'])<.001
    value=dict(run=str(RUN),workload='b32_q4096_repeated',new_gpu_measurements=False,
        same_trace_main_component_means_verified=True,
        mean_kernel_union_us_per_actual_layer=average,
        mean_communication_intersection_us_per_actual_layer={k:st.mean(r['communication_intersection_us_per_actual_layer'][k] for r in rows) for k in rows[0]['communication_intersection_us_per_actual_layer']},
        ranks=rows)
    engine=Path(a.read(RUN/'config.json')['engine_root'])
    value['inspected_sources']=[dict(path=str(p),sha256=a.sha(p)) for p in [
        engine/'python/sglang/srt/models/qwen2_moe.py',
        engine/'python/sglang/srt/models/qwen3_5.py',
        engine/'python/sglang/srt/batch_overlap/operations_strategy.py',
        engine/'python/sglang/srt/layers/moe/token_dispatcher/deepep.py']]
    a.save(ROOT/'analysis'/RUN.name/'overlap_results.json',value)
    a.save(ROOT/'report/overlap_results.json',value)
    note='''## Existing-trace check: communication/computation overlap

复查B32、P=0、Q4096重复文本的两张GPU独立Prefill trace，没有启动新GPU实验。每张卡640次实际Prefill层调用，重新归因的GEMM与通信均值与主报告吻合。

| GPU kernels / card / actual Prefill layer | µs |
|---|---:|
| Dispatch | 324.82 |
| Routed Expert GEMM | 484.73 |
| Combine | 259.22 |
| Dispatch+combine 与 routed GEMM 的时间交集 | 0.00 |
| Dispatch+combine 与 Attention 的时间交集 | 0.00 |
| Dispatch+combine 与 shared expert 的时间交集 | 3.98 |

Shared Expert本身约46.74 µs/card/layer；现有shared-expert overlap只与约0.68%的通信GPU时间重叠。这是GPU时间轴交集，不等于已经确认的端到端可节省时间。通信口径继承主报告的dispatch/combine作用域内kernel，并非纯NVLink传输线时间。

同一批token有 Dispatch(A) → GEMM(A) → Combine(A) 的依赖。可利用独立请求微批：Dispatch(B)与GEMM(A)重叠，Combine(A)与GEMM(B)重叠；也可用另一批的Attention覆盖通信。优先保留足够大的Expert输入形状，避免为重叠切得过细而损失GEMM效率。

忽略启动/排空、所有竞争及分块开销，仅把通信和GEMM看作理想独立流水资源，稳态工作间隔的乐观模型为 max(584.04,484.73)=584.04 µs，相比两项串行和1068.77 µs有484.73 µs的理想覆盖空间。这不是单批延迟、实际two-batch预测或E2E加速估计。Normal DeepEP通信使用SM，需与GEMM分配SM/带宽；分块后的两项时间也须重新测量。

当前实验two-batch overlap关闭。源码的Qwen shared-expert overlap与TBO是两种机制；现有TBO操作策略列出了Qwen3Moe等支持路径，当前Qwen3.5/3.6混合Full/GDN Attention的forward不在该Qwen3Moe操作分解路径中，不能据一个开关认定适配完成。此前暂停的TBO适配本次没有恢复。

因此主报告的PP吞吐优势是当前运行配置的结果，尚不能当作充分优化过通信/计算重叠的EP上限。完整两rank时间交集、CUDA stream与源文件SHA256见[overlap_results.json](./overlap_results.json)。
'''
    (ROOT/'report/overlap.md').write_text(note)
    report=ROOT/'report/REPORT.md';text=report.read_text();marker='## Existing-trace check: communication/computation overlap'
    if marker in text:text=text[:text.index(marker)]
    report.write_text(text.rstrip()+'\n\n'+note)
    combined=a.read(ROOT/'report/results.json');combined['overlap_inspection']=value
    a.save(ROOT/'report/results.json',combined)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
