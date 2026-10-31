"""Archive the controlled scheduling measurements without promoting a candidate."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    metrics=list(csv.DictReader((HERE/'analysis/metrics.csv').open()))
    audit=json.loads((HERE/'analysis/audit.json').read_text())
    rows=[];headline=[]
    for model in ('35b','122b'):
        selected=[r for r in metrics if r['model']==model]
        for r in selected:
            rows.append([model,r['arm'],r['mode'],'确认组3轮' if r['confirmation']=='True' else '初筛2轮',
                f"{float(r['output_tokens_per_s']):.2f}",f"{float(r['ttft_ms']):.2f}",
                f"{float(r['tpot_ms']):.3f}",f"{float(r['e2e_ms']):.2f}"])
        baseline=next(r for r in selected if r['arm']=='pp2' and r['mode']=='baseline')
        early=next(r for r in selected if r['mode']=='early_activation')
        headline.append(dict(model=model,pp_rate_gain_pct=100*(float(early['output_tokens_per_s'])/float(baseline['output_tokens_per_s'])-1),
            pp_tpot_change_ms=float(early['tpot_ms'])-float(baseline['tpot_ms'])))
    text='''# H100 双卡 Decode：提前发布 PP 激活的局部对照

## 布局、请求与计时

EP2：两张卡各一份不做 TP 的 Attention，每层路由专家跨卡分配，每卡128个。PP2：两卡各负责一半层，每层256个专家本地执行；两个等大流水槽位，每槽4请求。两个模型分别为 Qwen3.6-35B-A3B BF16（40层）和 Qwen3.5-122B-A10B FP8（48层），KV均为BF16，线性Attention状态为FP32。

同一已预约 H100 178 节点，实际 GPU0/1（NVLink）顺序运行，Guard 登记整节点4卡；没有新增GPU预约。复用完整CPU权重bank。每组8请求，每请求有4096个真实模型产生、已经驻留GPU的缓存前缀；输入1个新token，生成128个token。不是CPU HiCache换入、合成KV或MTP。原生混合缓存初始化／页对齐可能在启动Extend中产生额外回放；稳态Graph诊断丢弃前4轮，仅保留实际本地batch4的24轮。

上下文准备、完整形状预热、模式切换后的额外整批预热及独立profile均不计入正式指标。吞吐是整批输出token／整批完成时间；TTFT为提交至首token；TPOT为首至末token间隔／127；端到端为提交至末token。表中延迟为请求均值。

## 改了什么

- `baseline`：原生发布顺序，先处理上一槽反馈，再发送本槽激活。
- `early_activation`：计算提交后、处理上一槽反馈前，调用**同一个**原生激活发送方法；保留生产事件、stream、Tensor、输出FIFO和等待对象，不改变模型计算或输出反馈协议。
- `producer_ready`：先等生产事件，再发布原生消息；等待计时。
- `early_and_ready`：合并上述两项。

改动均为冻结的实验私有Python适配；共享引擎没有修改。EP原生overlap／PP不支持overlap、CUDA Graph、原生Hopper DeepGEMM、精度和本地batch均保留。六项CPU顺序／单次发送／模式切换测试通过。

PP四模式先做平衡顺序的8批初筛；根据初筛均值选择候选，再做独立 A/B/B/A/A/B 确认。确认组不混入初筛，两种模型都选中 `early_activation`。EP参考各3轮。没有按结果删除不利模式。

## 全部速度结果

'''+table(['模型','布局','模式','样本范围','输出tok/s','TTFT ms','TPOT ms','端到端 ms'],rows)+'''

## 结论与尚未通过的检查

提前发送激活的确认组中，35B吞吐改善约3.61%、TPOT减少约0.415 ms；122B改善约1.85%、TPOT减少约0.285 ms。额外同步生产事件的两项方案更慢，不作为候选。两个模型的PP仍慢于同卡EP；本轮没有制造“更重模型一定更偏EP”的趋势。

**候选未提升为正式优化版本。** 所有请求长度、缓存命中、完整模型资格、原始Guard postcheck、实际rank缓存加载及退出清理通过；但原调度自身A/A重复输出也不完全相同，跨模式输出同样不完全一致。原始逐请求token都保留在受控运行目录。本轮没有证明这种差异仅是正常数值波动，也没有证明候选损害语义；在厘清之前，速度只能作为实验级结果，不能称为确定性正确性验证通过。

## 为什么参数量不是唯一控制量

模型配置和实际资格证据显示，35B专家权重为BF16、隐藏维度2048、专家中间维度512；122B为FP8、隐藏维度3072、中间维度1024。忽略缩放元数据，一个完整路由专家三矩阵的存储约为6 MiB／9 MiB，增长50%；相同token数、BF16传输的隐藏激活字节也增长50%。因此Expert权重成本和通信负载同时增长，不能只用总参数量推导净收益。两模型层数、FP8／BF16计算内核也不同，不是只改“模型大小”的控制实验。

PP还保留模型Graph之外的阶段传输、反馈和调度成本；即使绝对额外成本不变，在更慢的模型上它的百分比也会减小。历史绝对分项已单独完整列在 [Q07报告](../h100_ep2_pp2_model_weight_analysis_20261026/REPORT.md)，本轮新样本未混入该历史图。

## Profile 与证据

[10行速度汇总](analysis/metrics.csv)、[12份逐卡稳态Graph统计](analysis/profiles.csv)、[完整性／输出审查及来源哈希](analysis/audit.json)。每份profile取24个实际Graph：Attention仅核心，Expert仅路由grouped GEMM；其他保留投影、共享专家、路由等。Graph各分项用GPU区间并集，分类之间仍可重叠，不能直接求和当TPOT。GPU内首核至下一Graph首核的间隔仅是该卡周期，不能直接解释为单个请求TPOT；Graph间隔还包含通信及依赖，不能全称空闲。

正式延迟不含profiler；本轮独立profile只能定位可能路径，不能把其绝对值直接加到未profile的TPOT上。
'''
    (HERE/'REPORT.md').write_text(text)
    manifest=dict(id=HERE.name,title='H100 two-GPU PP Decode activation scheduling A/B',
        created_at='2026-10-26',status='measurements_complete_correctness_investigation_pending',
        implementation=dict(engine='SGLang',commit='870269702285d94bdd926a50cc29ad1a20bcf60c',
            source_tag='experiment/sglang/glm47-ep16-canonical-profile-20261004-cpu-validated',
            worktree=str(HERE.parent.parent/'.engine-worktrees/glm47-ep16-canonical-profile-20261004'),
            shared_engine_changed=False,private_adapters=True),
        deployment=dict(node='spartan-gpgpu178',job_id='33646188',gpus='H100 80GB',
            physical_pair=[0,1],whole_node_lease_gpus=4,layouts=['EP2/AttentionDP2','PP2/AttentionTP1/ExpertEP1'],
            endpoint_retired=True,new_allocations=False),
        workload=dict(requests=8,gpu_cached_tokens_per_request=4096,fresh_input_tokens=1,output_tokens=128,
            real_semantic_prefix=True,hicache=False,mtp=False,pp_slots=2,requests_per_slot=4),
        artifacts=dict(report='REPORT.md',metrics='analysis/metrics.csv',profiles='analysis/profiles.csv',audit='analysis/audit.json'),
        key_results=headline,candidate_promoted=False,
        caveats=['Native A/A output variability not resolved; no deterministic acceptance.',
            'Different model precision/shape; not size-only causal evidence.'])
    (HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(report=str(HERE/'REPORT.md'),results=headline,production_candidate=False)))


if __name__=='__main__':main()
