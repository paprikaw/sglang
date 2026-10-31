"""Validate saved measurements and phase-specific profiles; CPU only."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

import prefill_attribution
import trace_analysis
from validate_metrics import analyze as validate_metrics


def save_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def decode_bucket(name):
    if 'flash::FlashAttnFwd' in name or 'flash::prepare_varlen_num_blocks_kernel' in name:
        return 'attention_ms'
    if 'fused_recurrent_gated_delta_rule' in name:
        return 'attention_ms'
    if 'deep_gemm::' in name and '(deep_gemm::GemmType)2' in name:
        return 'routed_gemm_ms'
    if 'deep_ep::' in name:
        return 'dispatch_ms' if 'dispatch' in name else 'combine_ms' if 'combine' in name else 'deepep_other_ms'
    if name.startswith('ncclDevKernel'):
        return 'nccl_ms'
    return 'other_ms'


def analyze(run, followup=None):
    evidence = validate_metrics(run)
    rows = evidence['rows']
    out = run / 'analysis'
    cfg = json.loads((run / 'config.json').read_text())
    complete = json.loads((run / 'completed.json').read_text())
    trace_analysis.summarize(run,followup)
    if followup:
        follow_post=next((followup/'dashboard_guard/checks').glob('*-post.json'))
        evidence['decode_followup']=dict(run=str(followup),guard_postcheck=str(follow_post),
            passed=json.loads(follow_post.read_text())['passed'],profile_only=True)
        (out/'validated_metrics.json').write_text(json.dumps(evidence,indent=2)+'\n')
    decode = []
    inventory = json.loads((out / 'decode_graph_inventory.json').read_text())['rows']
    for item in inventory:
        path = Path(item['trace'])
        arm = path.parents[2].name
        stage = '' if arm == 'ep2' else int(path.name.split('-PP-')[1].split('.')[0])
        costs = Counter({k: 0. for k in ('attention_ms','routed_gemm_ms','dispatch_ms',
            'combine_ms','deepep_other_ms','nccl_ms','other_ms')})
        n = len(item['samples'])
        for name, value in item['graph_kernel_duration_ms'].items():
            costs[decode_bucket(name)] += value / n
        assert costs['attention_ms'] > 0 and costs['routed_gemm_ms'] > 0, path
        counts = item['graph_kernel_counts']
        layers = 40 if arm == 'ep2' else 20
        full = sum(v for k,v in counts.items() if 'flash::FlashAttnFwdSm90<' in k)
        linear = sum(v for k,v in counts.items() if 'fused_recurrent_gated_delta_rule' in k)
        assert full == layers // 4 * n and linear == layers * 3 // 4 * n, path
        decode.append(dict(case=path.parent.name, arm=arm, stage=stage,
            graph_replays=n, graph_span_ms=statistics.mean(s['graph_span_ms'] for s in item['samples']),
            **dict(costs)))
    prefill, proofs = [], []
    for path in sorted(run.glob('*/traces/*_prefill/manifest.json')):
        manifest = json.loads(path.read_text())
        assert manifest['passed'] and manifest['phase'] == 'prefill'
        arm = path.parents[2].name
        for item in manifest['files']:
            p = Path(item['path'])
            assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256']
            row, proof = prefill_attribution.inspect_trace(p, manifest, arm)
            row['case'] = row['case'].removesuffix('_prefill')
            prefill.append(row)
            proofs.append(proof)
    assert len(prefill) == 12
    save_csv(out / 'prefill_components_per_rank.csv', prefill)
    save_csv(out / 'decode_components_per_rank.csv', decode)
    (out / 'prefill_validation.json').write_text(json.dumps(dict(passed=True,
        traces=proofs, full_model_layers=40), indent=2) + '\n')
    groups = {}
    for phase, data in [('prefill', prefill), ('decode', decode)]:
        fields = (('attention_ms','routed_expert_ms','shared_expert_ms','dispatch_ms','combine_ms',
                   'nccl_residence_ms','other_kernel_ms') if phase == 'prefill' else
                  ('attention_ms','routed_gemm_ms','dispatch_ms','combine_ms','nccl_ms',
                   'other_ms','graph_span_ms'))
        grouped = []
        for case in sorted({r['case'] for r in data}):
            for arm, stage in [('ep2',''), ('pp2',0), ('pp2',1)]:
                selected = [r for r in data if (r['case'],r['arm'],r['stage']) == (case,arm,stage)]
                if not selected:
                    continue
                assert len(selected) == (2 if arm == 'ep2' else 1)
                grouped.append(dict(case=case, arm=arm, stage=stage,
                    **{k:statistics.mean(r[k] for r in selected) for k in fields}))
        save_csv(out / (phase + '_components.csv'), grouped)
        groups[phase] = grouped
    capacities = evidence['logical_capacities']
    single_capacity = evidence['single_gpu_fit']['allocator_tokens']
    lines = ['# 单卡大小模型：H100 双卡 EP2 与 PP2', '',
        'Qwen3.6-35B-A3B，BF16 检查点约 67 GiB；本轮只启用完整语言模型，不加载视觉编码器。'
        '语言模型 40 层；每层 256 专家，激活 8 个。Attention 为全 Attention 与线性 Attention 一比三混合。', '',
        f'已实际验证单张 H100 加载全部 40 个语言模型层（加载后 GPU allocated 约 64.7 GiB），保留 **{single_capacity:,} token** KV 空间，'
        '并完成 4096 输入／4 输出请求；资格验证只允许 2 个并发请求、12 个线性状态，'
        '不同于正式配置的 32 请求运行预算，不能把此容量当作相同 batch 下的单卡性能上限。', '',
        '同节点 178：EP2 使用 GPU0/1，PP2 使用 GPU2/3，每组都是两张 H100 80GB、NV6。'
        'EP 每卡 Attention TP1、请求按 DP2 均分、每层 128 专家；PP 每阶段 20 层、每层 256 专家、'
        'Attention TP1，Decode 请求平均分两个槽位。Prefill 每个 Attention 实例的实际新增 token '
        'chunk 预算均为 4096，较大的 Prefill cohort 会拆成更多 forward，不能把槽位请求上限当成实际 Prefill batch。'
        '均为原生 BF16 DeepGEMM、FA3／Triton，Decode Graph 开启。', '',
        f"实测逻辑 KV 容量：EP2 **{capacities['ep2']:,} token**；PP2 **{capacities['pp2']:,} token**。"
        '容量已扣除通信、Graph 和线性 Attention 状态池；PP 的两个分层 KV 池不相加。', '',
        '## 正式性能', '',
        '旧 context 已在 GPU；构建和预热不计时。每点 3 轮。延迟为 **均值／中位数／P99**，'
        'P99 仅描述本次样本，不代表稳定服务尾延迟。Prefill 吞吐只计新增输入，Decode 只计生成 token。', '',
        '| 负载 | 阶段 | 配置 | 请求数 | 每条已缓存 token | 每条新输入／输出 | 吞吐 token/s | TTFT 秒 | TPOT 毫秒 | 端到端 秒 |',
        '|---|---|---|---:|---:|---:|---:|---|---|---|']
    def triple(r, k):
        return '—' if r[k+'_mean'] is None else ' / '.join(f'{r[k+"_"+s]:.3f}' for s in ('mean','median','p99'))
    for r in sorted(rows, key=lambda r:(r['case'],r['phase'],r['arm'])):
        label = '中等' if r['case']=='moderate' else '长缓存共同负载' if r['case'].startswith('near_') else '仅 PP 自身边界'
        lines.append(f"| {label} | {r['phase']} | {r['arm'].upper()} | {r['requests']} | "
            f"{r['cached_context_per_request']:,} | {r['fresh_input_per_request']} / {r['output_per_request']} | "
            f"{r['tokens_per_s']:.1f} | {triple(r,'ttft_s')} | {triple(r,'tpot_ms')} | {triple(r,'e2e_s')} |")
    lines += ['', '## 同工作量的吞吐差异', '', '| 请求组成 | 阶段 | PP2 相对 EP2 |', '|---|---|---:|']
    for case in complete['cases']:
        for phase in ('prefill','decode'):
            pair = {r['arm']:r for r in rows if r['case']==case['id'] and r['phase']==phase}
            delta = 100*(pair['pp2']['tokens_per_s']/pair['ep2']['tokens_per_s']-1)
            lines.append(f"| {case['requests']} 请求，每条旧缓存 {case['prefill_cached_context_tokens'] if phase=='prefill' else case['cached_context_tokens']:,} token | {phase} | {delta:+.2f}% |")
    lines += ['', '## 分阶段 GPU 诊断', '',
        'Prefill 单位为毫秒／每卡／整批请求，包含模块内投影与重排；Decode 为毫秒／每卡／一次实际 Graph，'
        'Attention 只计核心、Expert 只计路由 grouped GEMM。两种范围不同，不能直接互比占比。'
        '内核可重叠，累计值不能相加当 TPOT。PP 的 Dispatch／Combine 为零只表示没有专家交换，'
        '不代表阶段边界通信为零。NCCL 驻留可能含等待，不等于纯载荷传输。']
    for phase, fields in [('prefill',('attention_ms','routed_expert_ms','shared_expert_ms','dispatch_ms','combine_ms','nccl_residence_ms')),
                          ('decode',('attention_ms','routed_gemm_ms','dispatch_ms','combine_ms','other_ms','graph_span_ms'))]:
        names=dict(attention_ms='Attention',routed_expert_ms='路由 Expert 模块',shared_expert_ms='共享 Expert 模块',
            routed_gemm_ms='路由 Expert GEMM',dispatch_ms='Dispatch',combine_ms='Combine',
            nccl_residence_ms='NCCL 驻留',graph_span_ms='Graph GPU 跨度',other_ms='其他 GPU 内核')
        lines += ['', '### '+phase, '', '| 负载 | 配置／阶段 | '+' | '.join(names[k] for k in fields)+' |',
                  '|---|---|'+ '|'.join('---:' for _ in fields)+'|']
        for r in groups[phase]:
            case = next(c for c in complete['cases']+ [x['case'] for x in complete.get('extra_cases',[])] if c['id']==r['case'])
            desc = f"{case['requests']} 请求，每条旧缓存 {case['prefill_cached_context_tokens'] if phase=='prefill' else case['cached_context_tokens']:,} token"
            title = 'EP2 卡均值' if r['arm']=='ep2' else 'PP2 stage '+str(r['stage'])
            lines.append('| '+desc+' | '+title+' | '+' | '.join(f'{r[k]:.3f}' for k in fields)+' |')
    lines += ['', '**Decode 的其他 GPU 内核不是空闲时间**：其中仍包含投影、共享专家、'
        '激活／重排、归一化、router 和 KV 状态操作。Graph 内的狭义路由 GEMM 不是完整 Expert 模块，'
        '不能仅凭这个分项就断言全部剩余开销来自 Python 或通信。', '',
        '### Decode 每层路由 GEMM：避免把 40 层和 20 层直接比较', '',
        '| 请求组成 | 配置／阶段 | 每次 Graph 执行层数 | 每层路由 GEMM 毫秒 |',
        '|---|---|---:|---:|']
    for r in groups['decode']:
        case=next(c for c in complete['cases']+[x['case'] for x in complete.get('extra_cases',[])] if c['id']==r['case'])
        title='EP2 卡均值' if r['arm']=='ep2' else 'PP2 stage '+str(r['stage'])
        layers=40 if r['arm']=='ep2' else 20
        lines.append(f"| {case['requests']} 请求，每条旧缓存 {case['cached_context_tokens']:,} token | "
                     f"{title} | {layers} | {r['routed_gemm_ms']/layers:.4f} |")
    d16={('EP' if r['arm']=='ep2' else 'PP'+str(r['stage'])):r for r in groups['decode']
         if r['case']=='near_capacity_b16'}
    p32={r['arm']:r for r in rows if r['phase']=='prefill' and r['case']=='near_capacity_b32'}
    lines += ['', '## 本轮能说明什么', '',
        '- 长缓存时 Attention 成为很大的执行项：16 请求点，EP 每次 Graph 的 Attention 核累计 '
        f"{d16['EP']['attention_ms']:.2f} ms，路由 Expert GEMM {d16['EP']['routed_gemm_ms']:.2f} ms，"
        f"Dispatch＋Combine {d16['EP']['dispatch_ms']+d16['EP']['combine_ms']:.2f} ms。"
        '这是 GPU 内核累计量级，不是互斥时间占比，也不足以单独区分计算或显存带宽瓶颈。',
        '- 同一点 PP 每阶段只执行 20 层，但每层专家数从 128 增到 256。路由 GEMM 的每层耗时 '
        f"从 EP 的 {d16['EP']['routed_gemm_ms']/40:.4f} ms 变为 PP 两阶段的 "
        f"{d16['PP0']['routed_gemm_ms']/20:.4f}／{d16['PP1']['routed_gemm_ms']/20:.4f} ms。"
        '不能因为每阶段层数减半就认为计算一定减半；活跃专家、权重读取和矩阵形状都可能影响。'
        '本轮没有单独隔离这几个成因。省掉的专家通信还要支付 Expert 与流水依赖成本。',
        '- 固定总缓存 2,879,488 token，从 16 请求改成 32 请求意味着每条旧 context 减半。'
        'Decode TPOT 只小幅增加而总吞吐接近翻倍；Prefill Attention 累计耗时也接近，'
        '但新增输入量翻倍、Expert 开销增加。这是请求颗粒度改变的结果，不是固定单请求 context 的纯 batch sweep。',
        '- 32 请求 Prefill 的 PP 平均 TTFT 仍更低，但批次最后几个请求更晚：'
        f"平均 TTFT 为 EP {p32['ep2']['ttft_s_mean']:.3f}／PP {p32['pp2']['ttft_s_mean']:.3f} 秒，"
        f"样本 P99 为 EP {p32['ep2']['ttft_s_p99']:.3f}／PP {p32['pp2']['ttft_s_p99']:.3f} 秒。"
        '吞吐按整批 makespan 计算，因此平均响应更快与整批吞吐略低可以同时成立。',
        '- 所有实际 Decode Graph 都已核对每卡／每槽请求数，依次为 4、8、16。'
        'PP 每槽仍需依次通过两个 stage；槽位均衡不消除单请求的阶段依赖。'
        '精确拆分 Graph 外调度和反馈成本需要端到端时间线，本轮未把剩余差额全部归为网络或气泡。']
    lines += ['', '## 验证和边界', '',
        '- 使用既有完整 CPU 权重 Bank；逐卡层数、BF16 dtype、真实后端与 H100 架构通过资格检查。'
        '未启用 HiCache、CPU KV 换入、合成 KV 或 MTP。',
        '- 原始请求逐条核验新增／输出 token 数及旧缓存命中。缓存 Prefill 每轮从相同基础前缀开始，'
        '旧前缀重新准备在计时外；Decode 单独准备。实际 Graph 重放、逐日志 Bank 使用与清理已核验。',
        '- EP 支持 overlap，PP 为引擎支持的流水路径；不是只改通信的单变量实验。两对 GPU 并发仍共享节点资源。',
        '- 首轮诊断发现原生跳过默认预热时，EP 的自动 GC 冻结回调早于 HTTP socket 监听。'
        'r1 已清理并排除；r2 在解析该管理端点的纯文本应答时失败，已修复并通过 CPU 回归测试。'
        '最终运行在各服务就绪后显式执行同一个原生 `/freeze_gc`，不计入正式耗时。',
        '- 原始 Guard 收尾为失败，唯一原因是本实验单卡日志契约未适配 TP1 不带 TP0 标签的原生格式；'
        '原始记录与策略没有改动。另做只读复核：校验所有冻结输入、日志 inode／新增字节／哈希、'
        '原生单卡资格与唯一 canonical 加载记录，EP／PP 仍使用原始精确 rank 检查。'
        '这份补充证据不等于 Guard 原始检查通过，见 cache_usage_supplemental_review.json。',
        '- 长 context 与较大请求数组合触及实测 KV 边界；没有完整 batch sweep，不能宣称找到绝对最优 batch。'
        '单次短 profile 不能证明稳定通信尾延迟或精确独占瓶颈比例。',
        '- PP 专家集中与 EP 通信节省应结合实际阶段跨度和请求 TPOT 判断；不能只比较 Dispatch／Combine。', '',
        '- 长缓存 PP 的原始 4 步 profile 窗口只采到首 token Extend，没有实际 Decode Graph。'
        '后用同模型、同 GPU2/3、同运行预算和请求组成，单独补 16 步 Decode 诊断；'
        '该补采没有与 EP 并发，不能将 profiler 的绝对差额当作正式 TPOT 的精确归因。'
        '原始 trace 和正式性能未替换，逐 trace 清单保留对应关系。', '',
        '[CSV](full_metrics.csv) · [逐请求重算](validated_metrics.json) · '
        '[Prefill 逐卡](prefill_components_per_rank.csv) · [Decode 逐卡](decode_components_per_rank.csv) · '
        '[配置](../config.json) · [容量与请求组成](../capacity_matrix.json)', '',
        '![正式性能](performance.png)', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(passed=True, rows=len(rows), report=str(out/'REPORT.md'))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--decode-followup',type=Path)
    a=p.parse_args()
    analyze(a.run.resolve(),a.decode_followup.resolve() if a.decode_followup else None)
