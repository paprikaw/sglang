"""Aggregate the two validated optimization steps; historical EP8 stays labeled."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'glm47_pp_local_feedback_20261023'
spec=importlib.util.spec_from_file_location('local_feedback_analysis',PREVIOUS/'analyze.py')
LOCAL=importlib.util.module_from_spec(spec);spec.loader.exec_module(LOCAL)


def read(path):return json.loads(Path(path).read_text())
def source(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def components(path,partition):
    data=read(path);assert data['passed'] and not data['production_tpot_partition']
    groups={}
    for stage in (0,1):
        ranks=[r for r in data['ranks'] if r['rank']//4==stage]
        assert len(ranks)==4 and all(r['passed'] for r in ranks)
        values={k:st.fmean(r['target'][k] for r in ranks) for k in
                ('graph_span_ms','attention_ms','expert_ms','communication_ms','other_graph_ms','overlap_ms')}
        values['dispatch_combine_ms']=st.fmean(r['target']['exclusive_ms']['deepep_dispatch']+
                                             r['target']['exclusive_ms']['deepep_combine'] for r in ranks)
        values['moe_layers']=partition[stage]-(3 if stage==0 else 0)
        phases={}
        for phase in sorted(set().union(*(r['phases'] for r in ranks))):
            chosen=[r['phases'][phase] for r in ranks if phase in r['phases']]
            phases[phase]={k:st.fmean(r[k] for r in chosen) for k in
                           ('graph_span_ms','attention_ms','expert_ms','communication_ms','other_graph_ms')}
        groups[f'stage{stage}']=dict(target=values,phases=phases)
    return dict(groups=groups,source=source(path),production_tpot_partition=False)


def candidate(run):
    task=read(run/'task/results.json');cleanup=read(run/'task/cleanup.json')
    assert task['passed'] and cleanup['passed'] and task['candidate_baseline_output_parity']['passed']
    arm=run/'pp2ep4_balanced';owned=read(arm/'arm_result.json')
    assert owned['passed'] and owned['performance_valid'] and owned['profile_valid'] and owned['cleanup_verified']
    cohorts=[read(arm/name) for name in ('p4096_decode_results.json','p4096_decode_repeat_results.json')]
    performance={k:LOCAL.stats([r[k] for r in cohorts]) for k in ('throughput_tok_s','wall_s')}
    for k in ('ttft_s','tpot_s','end_to_end_s','spec_accept_length','spec_accept_rate','spec_verify_ct'):
        performance[k]=LOCAL.stats([r[k]['mean'] for r in cohorts])
    traces=[LOCAL.flow(arm/f'unfenced_flow_rank{rank}.json') for rank in range(8)]
    for row in traces:
        # This diagnostic fixes phase propagation for direct draft-backend replay.
        assert row['draft_replays_observed']==(row['rank']>=4)
        row['caveat']='Actual target and tail draft Graph intervals use unique outer events without per-replay fences. Current-stream gaps are not device idle and PP host API spans are not pure NIC latency.'
    return dict(performance=performance,flow=traces,
        components=components(arm/'stage_activity_summary.json',[50,42]),
        flow_overhead=read(arm/'unfenced_flow_results.json')['trace_overhead_ratio'],
        candidate_baseline_output_parity=task['candidate_baseline_output_parity'],
        sources=[source(arm/name) for name in ('p4096_decode_results.json','p4096_decode_repeat_results.json',
                                             'stage_activity_summary.json','unfenced_flow_results.json')])


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    run=parser.parse_args().run.resolve()
    first=read(PREVIOUS/'summary.json');balanced=candidate(run)
    result=dict(run=str(run),workload='SemanticB8, logicalP4096 / nativeCPU4032, Q1 ->64, MTP topk1/steps2/VerifyQ3; CPU HiCache H2D timed',
        engine_commit='9e48f876a8eb2d7460317bc713bfa8680d78aea5',nodes=['spartan-gpgpu065','spartan-gpgpu167'],
        gpu='8xA10080GB',transport='NVLink pairs / RDMA outside pairs, PCIe threshold0, no CE',
        sources=[source(PREVIOUS/'summary.json'),source(run/'config.json'),source(run/'task/results.json')],
        previous_optimization=first,balanced=balanced,cleanup_passed=True,
        no_lossless_mtp_off_claim=True)
    original=first['arms']['pp2ep4_baseline']['performance']
    local=first['arms']['pp2ep4_local_tail']['performance'];b=balanced['performance']
    ep=first['historical_same_bank_ep8']
    result['ratios']=dict(
        balanced_tpot_vs_original=b['tpot_s']['mean']/original['tpot_s']['mean'],
        balanced_throughput_vs_original=b['throughput_tok_s']['mean']/original['throughput_tok_s']['mean'],
        balanced_tpot_vs_local_tail=b['tpot_s']['mean']/local['tpot_s']['mean'],
        balanced_tpot_vs_historical_ep8=1000*b['tpot_s']['mean']/ep['tpot_ms'],
        balanced_throughput_vs_historical_ep8=b['throughput_tok_s']['mean']/ep['throughput_tok_s'])
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    table=['# PP2×EP4 MTP Decode 优化','',result['workload'],'',
        '| 配置 | 输出 tok/s ↑ | TTFT mean ms ↓ | TPOT mean ms ↓ | E2E mean ms ↓ |',
        '|---|---:|---:|---:|---:|']
    for name,p in [('原 PP 46/46',original),('PP 46/46，本地反馈退休',local),('PP 50/42，本地反馈退休',b)]:
        table.append(f"| {name} | {p['throughput_tok_s']['mean']:.3f} | {1000*p['ttft_s']['mean']:.3f} | {1000*p['tpot_s']['mean']:.3f} | {1000*p['end_to_end_s']['mean']:.3f} |")
    table.extend(['',f"EP8 同 Bank 历史参照：{ep['throughput_tok_s']:.3f} tok/s，TPOT {ep['tpot_ms']:.3f} ms；不是本次重新运行的控制组，跨布局严格输出等价仍未通过。",'',
        '候选与前一轮固定 Verify / 本地退休基线逐 token 和文本完全一致；各臂两个生产 cohort 自重复通过。MTP-off 的已有数值差异没有在本轮修复。',
        '', '两次优化的实际分段、无逐次同步的调度诊断、每 cohort 指标和原始来源哈希保存在 summary.json。组件 Graph 时长不等于 TPOT，不能将 stage 时长相加解释为完整流水线周期。',
        '', '第一步只删除已由尾 stage 持有的结果从首 stage 回传的冗余环路。第二步把4个目标层移到首 stage，补偿尾 stage 的草稿工作；总层数、Expert 数、请求、MTP 参数及输出预算均未减少。'])
    table.extend([
        '', '## 控制条件与测量边界', '',
        '- 同一对已预约节点 065／167、8×A100 80GB、同型号 ConnectX-6 Dx；没有新增预约。',
        '- GLM-4.7-AWQ / Humming INT4，BF16 KV；Attention TP4，Expert EP4，PP2。固定两个 B4 slot，不减少全局 B8。',
        '- 真实语义前缀 P4096，原生 CPU HiCache 实际持有4032 token，GPU 起始为空；计时内重算64 token 尾部并新增 Q1，生成64 token／请求。',
        '- CPU 权重预载及源 Bank 导入不计时，原生 HiCache→GPU 换入计时。两种分层方式复用同一个不可变语义 Bank，按逻辑层映射，不重建前缀。',
        '- MTP top-k1、steps2、Verify Q3，完整 CUDA Graph；每个配置两个无 profiler 的生产 cohort，之后独立做分段与调度诊断。样本数小，不给出显著性或泛化结论。',
        '- DeepEP 原生库、Expert 内核、通信容量不变；节点内 NVLink pair，pair 外均走 RDMA，PCIe threshold=0。本轮没有启用或修改 CE 路径。',
        '', '## 归因', '',
        '第一步移除冗余反馈：尾 stage 已持有结果，但原来仍需等待结果经首 stage 回传。现在保留该 slot 的独占结果，等生产事件就绪后在尾 stage 本地消费；尾→首反馈和 stage 激活传输均保留。slot 生命周期、请求归属和退出排空均有检查。',
        '',
        '无逐 replay 同步的诊断中，尾 stage 活跃反馈 API 平均耗时由3.193降至1.048 ms。该值包含主机调度、序列化与等待，不能当作纯 NIC 延迟。目标 Graph 计算时长基本不变，生产 TPOT 降低3.49%。',
        '',
        '第二步按实际工作而非目标层数平衡：模型是92个目标层加1个物理 MTP 层，steps2并不代表额外的三层权重。原来46／46分层没有扣除尾 stage 的草稿工作。移动4个目标层到首 stage 后，目标计算总量没有减少。',
        '',
        '| 诊断项：每卡／每次相应 Graph，ms | 本地反馈，46／46 | 本地反馈，50／42 |',
        '|---|---:|---:|'])
    before=first['arms']['pp2ep4_local_tail']['components']['groups']
    after=balanced['components']['groups']
    # The old analyzer groups contain the target metrics directly; keep the
    # original schema distinct from the balanced per-phase representation.
    b0,b1=(before[f'stage{s}']['phases']['target_verify'] for s in (0,1))
    a0,a1=after['stage0']['target'],after['stage1']['target']
    # Use saved per-phase summaries for draft work, not stream gaps.
    before_draft=sum(before['stage1']['phases'][p]['graph_span_ms'] for p in ('draft','draft_extend'))
    after_draft=sum(after['stage1']['phases'][p]['graph_span_ms'] for p in ('draft','draft_extend'))
    for label,x,y in [
        ('首 stage 目标 Graph',b0['graph_span_ms'],a0['graph_span_ms']),
        ('尾 stage 目标 Graph',b1['graph_span_ms'],a1['graph_span_ms']),
        ('尾 stage 草稿 + draft extend',before_draft,after_draft),
        ('尾 stage 目标 + 草稿工作',b1['graph_span_ms']+before_draft,a1['graph_span_ms']+after_draft),
        ('两段目标 Expert 合计',b0['expert_ms']+b1['expert_ms'],a0['expert_ms']+a1['expert_ms'])]:
        table.append(f'| {label} | {x:.3f} | {y:.3f} |')
    table.extend([
        '', '这张表只用于比较分段工作量，不把其和解释为生产 TPOT。无逐次同步的实际同 slot 启动周期，rank0／rank4由121.39／121.57降至112.28／112.49 ms，和独立生产测试的改善方向一致。间隙包含其他计算及依赖，不等于 GPU 空闲时间。',
        '', '## 验证与限制', '',
        '- 两步候选均与相应固定 Verify PP 基线的8个请求逐 token ID、文本完全相同，自重复也完全一致；平均推进2.4653 token／验证周期、接受率0.73034和每请求验证次数26均未改变。',
        '- CPU 协议与 Bank 分层映射测试共18项通过，真实 GPU 性能、profile、Guard 后验检查、退出清理均通过；两次 GPU 租约均已验证释放。',
        '- 同 Bank 旧 EP8 参照的 Expert 目标计算约25.06 ms，而 PP 两段约26.05 ms；DeepEP dispatch+combine约26.71对17.93 ms。这里的 Expert 小幅增加不足以抵消通信差额，但这些 Graph 组件不能直接相加推导生产速度。',
        '- 相对旧 EP8，优化 PP 的吞吐高6.95%、TPOT仍高1.41%；EP8不是本次新跑的控制组，不能据此宣称严格胜出。',
        '- 固定 Verify 的 MTP 与 MTP-off 原有数值不一致仍未修复，EP／PP之间也未通过严格输出等价。本轮证明的是保持 PP 基线输出不变的调度优化，不是无损 MTP 的完整正确性证明。',
        '- 50／42只在本次 B8、P4096、MTP 参数下验证，不是所有 context／batch 的通用最优分层。代码目前是冻结 runtime 的显式启用适配，不是默认全局开启。'])
    (ROOT/'RESULTS.md').write_text('\n'.join(table)+'\n')
    print(json.dumps(result['ratios'],indent=2))


if __name__=='__main__':main()
