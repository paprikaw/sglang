"""Publish the fixed-group confirmation separately from historical profiles."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
        '| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    out=HERE/'analysis/fixed_groups'
    metrics=list(csv.DictReader((out/'metrics.csv').open()))
    profiles=list(csv.DictReader((out/'profiles.csv').open()))
    audit=json.loads((out/'audit.json').read_text())
    if len(metrics)!=6 or len(profiles)!=12 or len(audit['runs'])!=2:
        raise RuntimeError('Both complete, verified model comparisons required')
    index={(r['model'],r['arm'],r['mode']):r for r in metrics}
    rows=[];headlines=[];comparisons=[]
    for model in ('35b','122b'):
        ep=index[model,'ep2','baseline'];old=index[model,'pp2','baseline'];new=index[model,'pp2','early_activation']
        for layout,row in [('EP2',ep),('PP2 原调度',old),('PP2 提前发送',new)]:
            rows.append([model,layout,*[f'{float(row[k]):.3f}' for k in
                ('output_tokens_per_s','ttft_ms','tpot_ms','e2e_ms')]])
        headline=dict(model=model,
            throughput_gain_pct=100*(float(new['output_tokens_per_s'])/float(old['output_tokens_per_s'])-1),
            tpot_reduction_ms=float(old['tpot_ms'])-float(new['tpot_ms']),
            pp_ep_tpot_gap_ms=float(new['tpot_ms'])-float(ep['tpot_ms']),
            pp_ep_tpot_penalty_pct=100*(float(new['tpot_ms'])/float(ep['tpot_ms'])-1))
        headlines.append(headline)
        comparisons.append([model,f"{headline['throughput_gain_pct']:.2f}%",
            f"{headline['tpot_reduction_ms']:.3f}",f"{headline['pp_ep_tpot_gap_ms']:.3f}",
            f"{headline['pp_ep_tpot_penalty_pct']:.2f}%"])
    groups=defaultdict(list)
    for row in profiles:groups[row['model'],row['arm'],row['mode']].append(row)
    def pp(model,mode,rank):
        return next(r for r in groups[model,'pp2',mode] if f'PP-{rank}.' in r['rank_trace'])
    gpu_rows=[]
    keys=['attention_core_union_ms','routed_gemm_union_ms','dispatch_union_ms','combine_union_ms',
          'other_union_ms','graph_span_ms','between_graph_spans_ms','routed_gemm_nccl_overlap_ms']
    for model in ('35b','122b'):
        selected=[('EP2 卡均值',{k:statistics.mean(float(r[k]) for r in groups[model,'ep2','baseline']) for k in keys})]
        selected += [(f'PP2 {label} stage {rank}',pp(model,mode,rank))
            for mode,label in [('baseline','原调度'),('early_activation','提前发送')] for rank in (0,1)]
        for label,row in selected:
            gpu_rows.append([model,label,*[f'{float(row[k]):.3f}' for k in keys]])
    overlap_rows=[]
    for model in ('35b','122b'):
        for mode,label in [('baseline','原调度'),('early_activation','提前发送')]:
            p=pp(model,mode,0)
            overlap_rows.append([model,label,f"{float(p['routed_gemm_union_ms']):.3f}",
                f"{float(p['routed_gemm_nccl_overlap_ms']):.3f}",
                f"{float(p['nccl_during_graph_span_ms']):.3f}"])
    payload_count=sum(a['matched_messages'] for run in audit['runs'] for a in run['payload_audits'])
    if not all(run['pp_output_agreement']['all_modes_equal'] and
               run['pp_output_agreement']['baseline_repeat_equal'] and
               all(a['passed'] and a['exact_stage_batch_sequence_equal'] for a in run['payload_audits'])
               for run in audit['runs']):
        raise RuntimeError('Tested PP candidate did not pass the stated acceptance scope')
    text='''# H100 Decode：固定请求分组后的 PP 调度优化

## 测量口径

两种模型是 **Qwen3.6-35B-A3B BF16**（40层）和 **Qwen3.5-122B-A10B FP8**（48层），均使用原生 Hopper DeepGEMM、BF16 KV 和 FP32 线性 Attention 状态。它们的精度、层数、矩阵尺寸均不同，不能把这组比较解释成只改变模型大小的因果实验。

同一已预约 H100 178、同一对 GPU0/1（NVLink），两种布局顺序运行；Guard 登记整节点4卡，没有申请新节点。EP2 每卡一份 Attention，路由专家跨卡分配；Attention 不做 TP。PP2 每卡负责一半模型层，本地保存该层全部专家，两个流水槽位各处理4请求。没有 MTP、HiCache 换入或合成 KV。完整权重从已存在的 CPU bank 复用，CPU 权重缓存没有替换。

每批8请求，各有 **4096个真实模型产生且已驻留GPU的缓存前缀token**，新增输入1个token，生成128个token。冷前缀构建、完整形状预热、模式切换后的整批预热、传输校验及独立profile均不计入正式指标。该点容量充足，不是KV容量极限实验。TTFT包含这次请求启动处理，但不包含前缀构建；TPOT是首token至末token的时间除以127；端到端是提交至末token；吞吐是1024个输出token除以整批完成时间。延迟均为请求均值，每个模式3轮。

本轮固定 PP 两槽成员及顺序：文档0/2/4/6一组，1/3/5/7一组，与EP两个Attention实例的请求成员相同。两个PP stage执行相同准入规则；首个槽等待8请求到齐的时间也计入客户端延迟。这是针对声明的同时到达8请求的实验入口，不是建议任意线上流量必须等待凑满8请求。原生EP仍保留其组内调度和overlap，没有降低EP性能来匹配PP。

## 改动与未改动

提前发送方案把**同一个原生激活发送调用**移到上一槽反馈处理之前，保留生产事件、stream、Tensor、输出FIFO、等待对象及模型计算。两种PP模式使用完全相同的固定分组入口。EP参考3轮；PP按原调度／提前／提前／原调度／原调度／提前顺序测6轮。没有按结果挑删轮次。代码均为实验私有冻结适配，没有修改共享引擎默认配置。

## 正式速度结果（不含 profiler 或张量校验）

'''+table(['模型','布局','输出tok/s','TTFT ms','TPOT ms','端到端 ms'],rows)+'''

'''+table(['模型','PP吞吐提升','PP减少TPOT ms','优化PP仍比EP多出的TPOT ms','优化PP的TPOT相对EP增幅'],comparisons)+'''

![正式TPOT与独立GPU诊断的绝对数值](analysis/fixed_groups/scheduling_absolute.png)

## 发现的实际开销

原调度首stage的Expert GEMM与Graph外的 **NCCL SendRecv内核驻留**发生重叠；提前发送后，该重叠在本轮24个稳态Graph样本中消失。首stage的GEMM变快，尾stage基本不变，Attention核心也基本不变。它说明原先被计作Expert执行时间的部分成本受PP通信调度影响，不能全部归因于专家权重更大。

下表单位均为 **ms／卡／实际本地4请求Decode Graph**，驻留包含通信及等待，不是链路线速延迟：

'''+table(['模型','首stage策略','路由Expert GEMM','其中与NCCL驻留重叠','Graph时段内NCCL驻留'],overlap_rows)+'''

这组A/B只有发布顺序变化，固定请求下输出完全一致，支持调度改动改善运行效率。区间重叠仍不是SM占用、内存带宽或因果硬件计数器；本轮没有直接采这些计数器。不能把“重叠消失”宣传为已证明大量SM原先被占满。

重模型首stage变快后，Graph间依赖等待增加，尾stage仍更慢，因此首stage GEMM的改善不能等比例兑现成TPOT。接近下一Graph的间隔包含阶段传输、反馈及调度依赖，不全是GPU空闲。表里的Graph采样与正式TPOT也不能混合相减来做严格延迟守恒。

## 全部GPU绝对分项

每个逐卡profile选取24个预热后的实际Graph；EP列是两卡均值，PP按stage分开。Attention只含核心内核，路由Expert只含grouped GEMM；共享专家、投影、路由等在其他GPU内核内。区间可重叠，不可把这些列或两个PP stage直接加成请求TPOT；Graph内没有NCCL也不代表整个PP没有通信。

'''+table(['模型','配置／阶段','Attention核心 ms','路由Expert ms','Dispatch ms','Combine ms',
           '其他GPU核 ms','Graph跨度 ms','相邻Graph间隔 ms','Expert与NCCL重叠 ms'],gpu_rows)+'''

## 如何理解“重模型更偏向EP”

本轮优化后，PP相对EP的TPOT**绝对差额**由约2.683 ms增加到3.040 ms，支持“重模型的绝对额外代价可能更大”。但相对增幅由约33.77%降到27.29%，所以不能据此说重模型一定具有更大的EP百分比收益。

模型35B→122B不只增大专家成本：一个路由专家三矩阵的实存字节约6 MiB→9 MiB，增长50%；BF16隐藏激活的通信字节也因隐藏维度2048→3072增长50%。还有40→48层、BF16→FP8及原生内核差别。真正决定部署的是**PP额外Expert与调度成本相对于它省掉的通信成本**，不是模型总参数量一项。

历史Prefill与Decode的绝对分项继续保留在 [Q07图表](../h100_ep2_pp2_model_weight_analysis_20261026/absolute_components.png) 和 [完整表格](../h100_ep2_pp2_model_weight_analysis_20261026/absolute_components.csv)。本轮没有重跑Prefill，也不把不同prompt／分组／kernel边界的历史分项与本轮新profile混成一个可加总的延迟模型。

## 正确性、失败记录与采用范围

'''+f'''- 两种模型的PP原调度重复三轮、提前发送重复三轮，8请求各128个输出token逐项完全相同，共核对每模型6144个生成token；所有请求有正确的4096缓存命中与输出长度。
- 两种模型、两种PP策略的独立诊断共匹配 **{payload_count}条阶段消息**；发送与接收的类型、次序、形状、dtype、原始字节SHA256完全一致。两stage的完整请求批次序列也逐批相同，稳态槽位成员经过实际采样核实。
'''+'''- 之前原生随机分组诊断另有4626条消息匹配通过；原调度A/A同样会改变微批成员并出现输出差异。固定PP分组后，本轮该现象消失。原生EP自身仍有A/A输出差异，保留原始记录，不据此虚称任意部署都逐位确定。
- 两个最终配对run的原始Guard postcheck、每rank权重缓存加载及退出清理全部通过，GPU已释放。21项CPU顺序／边界测试通过。
- `35b_fixed_groups_r1`在准入入口错误返回空值，未测到PP正式结果；`r2`为修正控制返回数量而主动中止，保留原始记录和实际EP清理证据，未拼入最终性能表。最终比较仅来自`35b_fixed_groups_r3`、`122b_fixed_groups_r3`。准备但未启动的其他目录不算测量。
- **候选在本轮限定模型、缓存、固定8请求及两槽配置下通过验证**，仍是私有实验适配，未提升为全局默认版本，不声称所有workload或线上生成质量均已覆盖。

## 可复核来源

[6行正式指标](analysis/fixed_groups/metrics.csv)、[12份逐卡Graph绝对分项](analysis/fixed_groups/profiles.csv)、[传输／输出／准入／清理审查](analysis/fixed_groups/audit.json)。每份逐卡profile带原始路径和SHA256；正式逐请求数据与原始trace保留在各run目录。重生成：先运行`analyze_controlled.py`，再运行`report_controlled.py`。缓存解析结果以原trace与分析代码哈希绑定，不改变输入数据。
'''
    (HERE/'FIXED_GROUPS_REPORT.md').write_text(text)
    summary=dict(status='tested_candidate_validated',models=headlines,
        pp_output_equality_passed=True,matched_messages=payload_count,
        hardware='H100 178, same GPUs0/1; full-node Guard registration',
        scope='Real GPUcached4096/new1/output128/B8; two fixed PP slots of four requests',
        shared_engine_changed=False,production_promoted=False,
        causal_limit='Model precision, shapes, layer count and communication bytes differ')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    original=HERE/'REPORT.md';start='\n## 后续：固定请求分组的确认\n'
    body=original.read_text().split(start)[0]
    original.write_text(body+start+'\n'+
        '已进一步定位并修正实验微批组成未固定的问题。两个模型的PP原调度与提前发送均通过本轮逐token一致性及真实传输校验；35B吞吐改善约3.98%、122B约1.35%。原始初筛数值保留，不与后续合并。'+
        '\n\n[固定分组的6行指标、完整绝对分项及NCCL重叠归因](FIXED_GROUPS_REPORT.md)。本页早期“待诊断”状态只描述原随机分组队列；限定验证结果以后续报告为准。\n')
    manifest=json.loads((HERE/'manifest.json').read_text())
    manifest.update(status='fixed_group_confirmation_complete',fixed_group_confirmation=summary)
    manifest['artifacts']['fixed_groups_report']='FIXED_GROUPS_REPORT.md'
    manifest['artifacts']['fixed_groups_summary']='analysis/fixed_groups/summary.json'
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
