"""Audit the independent recheck and join untimed, cross-run numerical probes."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st

import analyze

ROOT=Path(__file__).resolve().parent
MAIN=ROOT/'runs/qwen35b_h100_prefill_optimized_r2_20261027'
FOCUS=ROOT/'runs/qwen35b_h100_prefill_focus_r3_20261027'
REPLAY=ROOT/'runs/qwen35b_h100_prefill_numerical_r4_20261027'
OUT=ROOT/'report'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def source(p,kind): return dict(path=str(p),sha256=sha(p),kind=kind)


def audit(run,complete):
    spec=read(run/'dashboard_spec.json')
    for d in spec['frozen_inputs']: assert sha(Path(d['path']))==d['sha256'],d
    assert read(run/'cpu_qualification.json')['passed']
    assert read(run/'task/cleanup.json')['passed']
    posts=[read(p) for p in (run/'dashboard_guard/checks').glob('*-post.json')]
    assert len(posts)==1
    post=posts[0]
    assert post['cleanup_verified'] and post['model_cache']['passed']
    records=[]
    for line in (run/'supervisor.log').read_text().splitlines():
        try: v=json.loads(line)
        except ValueError: continue
        if isinstance(v,dict) and 'gpu_registry_state' in v: records.append(v)
    assert records[-1]['gpu_registry_state']=='released'
    progress=read(run/'task/progress.json')
    if complete:
        assert post['passed'] and progress['stage']=='complete'
        assert progress['valid_points']==read(run/'config.json')['expected_points']
    else:
        assert progress['stage']=='failed' and not post['passed']
    return dict(run=str(run),status=progress['stage'],guard_postcheck=post['passed'],
        complete_wrapper_required=complete,frozen_hashes_verified=True,cpu_qualification=True,
        cache_usage_verified=True,cleanup_verified=True,gpu_registry_state='released',
        completed_points=progress['valid_points'],expected_points=read(run/'config.json')['expected_points'])


def top(row):
    values=row['validation_logits']['output_top_logprobs']
    assert len(values)==1 and len(values[0])==10
    result={int(v[1]):float(v[0]) for v in values[0]}
    assert len(result)==10
    return result


def compare(case,label,left,right):
    assert len(left)==len(right)==2
    details=[]; deltas=[]; margins=[]; changes=[]
    for repeat,(a,b) in enumerate(zip(left,right)):
        assert len(a['requests'])==len(b['requests'])==32
        for i,(x,y) in enumerate(zip(a['requests'],b['requests'])):
            ta,tb=top(x),top(y)
            aa,bb=x['output_ids'][0],y['output_ids'][0]
            shared=set(ta)&set(tb)
            diff=[abs(ta[t]-tb[t]) for t in shared];deltas+=diff
            assert shared
            row=dict(workload=case,comparison=label,repeat=repeat,request=i,
                reference_token=aa,candidate_token=bb,changed=aa!=bb,
                shared_top10_tokens=len(shared),max_shared_logprob_delta=max(diff),
                both_predictions_in_both_top10=aa in tb and bb in ta)
            if aa!=bb:
                if row['both_predictions_in_both_top10']:
                    row.update(reference_logprob_margin=ta[aa]-ta[bb],
                        candidate_logprob_margin=tb[bb]-tb[aa],
                        reference_probability_margin=math.exp(ta[aa])-math.exp(ta[bb]),
                        candidate_probability_margin=math.exp(tb[bb])-math.exp(tb[aa]))
                    margins += [row['reference_logprob_margin'],row['candidate_logprob_margin']]
                changes.append(row)
            details.append(row)
    return dict(workload=case,comparison=label,request_comparisons=len(details),
        changed_predictions=len(changes),all_changes_within_both_top10=all(r['both_predictions_in_both_top10'] for r in changes),
        max_changed_prediction_logprob_margin=max(margins) if margins else None,
        max_abs_shared_top10_logprob_delta=max(deltas),mean_abs_shared_top10_logprob_delta=st.mean(deltas)),details


def main():
    focused=read(ROOT/'analysis'/FOCUS.name/'results.json')
    fa=audit(FOCUS,False);ra=audit(REPLAY,True)
    assert len(focused['comparisons'])==1
    c=focused['comparisons'][0];assert c['workload']=='b8_q4096_repeated'
    metrics=[r for r in focused['metrics'] if r['workload']==c['workload']]
    assert len(metrics)==3 and all(r['repeats']==10 for r in metrics)
    assert read(FOCUS/'pp2/workloads'/c['workload']/'output_agreement.json')['passed']
    for arm in ('ep2','pp2'):
        assert read(FOCUS/arm/'cleanup.json')['cleanup_verified']
    summary=[];details=[];sources=[];formal_replay=[];output_checks=[]
    main_cfg=read(MAIN/'config.json')
    for case in main_cfg['workloads']:
        rows=[]
        for arm,policy in [('ep2',0),('pp2',0),('pp2',1)]:
            rounds=read(MAIN/arm/'workloads'/case['id']/f'metrics.prefill.p{policy}.json')['rounds']
            first=[r['output_ids'] for r in rounds[0]['requests']]
            assert all(first==[r['output_ids'] for r in g['requests']] for g in rounds)
            rows.append(first)
        assert rows[1]==rows[2]
        changed=[i for i,(x,y) in enumerate(zip(rows[0],rows[2])) if x!=y]
        output_checks.append(dict(workload=case['id'],requests=case['requests'],
            cross_layout_changed_first_tokens=len(changed),changed_request_indices=changed,
            pp_native_early_equal=True,stable_across_formal_repetitions=True))
    for case,q in [('b32_q128_repeated',128),('b32_q512_repeated',512)]:
        e=FOCUS/'ep2/workloads'/case;p=REPLAY/'pp2/workloads'/case
        ep=read(e/'validation.prefill_logprobs.json');pp=read(p/'validation.prefill_logprobs.json')
        assert not ep['performance_sample'] and not pp['performance_sample']
        assert pp['validation_mem_fraction']==.60
        assert read(e/'protocol.json')['input_sha256']==read(p/'protocol.json')['input_sha256']
        assert read(e/'protocol.json')['input_sha256']==read(MAIN/'pp2/workloads'/case/'protocol.json')['input_sha256']
        for value in (ep,pp):
            for groups in value['by_policy'].values():
                assert len(groups)==2
                for group in groups:
                    assert len(group['requests'])==32
                    for row in group['requests']:
                        assert (row.get('cached_tokens') or 0)==0 and row['prompt_tokens']==q and row['output_tokens']==1
        for label,left,right in [('EP_vs_PP_native',ep['by_policy']['0'],pp['by_policy']['0']),
                                  ('PP_native_vs_early',pp['by_policy']['0'],pp['by_policy']['1'])]:
            row,ds=compare(case,label,left,right);summary.append(row);details+=ds
        for arm,values,base in [('ep2',ep,FOCUS),('pp2',pp,MAIN)]:
            for policy,groups in values['by_policy'].items():
                original=read(base/arm/'workloads'/case/f'metrics.prefill.p{policy}.json')['rounds'][0]['requests']
                mismatches=[sum(x['output_ids']!=y['output_ids'] for x,y in zip(original,g['requests'])) for g in groups]
                formal_replay.append(dict(workload=case,arm=arm,policy=int(policy),reference_formal_run=str(base),
                    mismatched_requests_by_replay=mismatches,reference_requests=len(original)))
        for folder in (e,p):
            for name in ('protocol.json','validation.prefill_logprobs.json'):
                sources.append(source(folder/name,'untimed_numerical_validation'))
    analyze.table(OUT/'numerical_summary.csv',summary)
    analyze.table(OUT/'numerical_details.csv',details)
    analyze.table(OUT/'focused_recheck.csv',metrics)
    analyze.table(OUT/'cross_layout_outputs.csv',output_checks)
    value=dict(focused_recheck=dict(comparison=c,metrics=metrics,audit=fa,
        timing_validity='All ten formal B8Q4096 cohorts/config completed and were independently audited before the untimed PP logprob OOM. Wrapper status remains failed; no missing workloads are imputed.',
        raw_analysis=str(ROOT/'analysis'/FOCUS.name/'results.json')),
        numerical_validation=dict(audit=ra,performance_samples=False,ep_mem_fraction=.94,pp_mem_fraction=.60,
            summary=summary,details=details,formal_replay=formal_replay,
            source_descriptors=sources),
        cross_layout_output_checks=output_checks,
        quality_boundary='Matching PP native/early outputs validates the early-send change. Cross-layout EP/PP native short-input numerical equivalence is unresolved: changed predictions need not appear in the other layout top10, so do not dismiss this as a near-tie or attribute it to BF16 without further evidence.')
    save(OUT/'followups.json',value)
    result=read(OUT/'results.json');result['followups']=value;save(OUT/'results.json',result)
    lines=['## 独立定点复测与数值诊断','',
        'B8/Q4096主矩阵的5轮样本波动较大，独立补测每配置10轮。以下均为清缓存、profiler关闭的正式样本，未与主矩阵合并。','',
        '| Layout | Input tok/s ↑ | Sample standard deviation |','|---|---:|---:|']
    for m in metrics:
        label={('ep2',0):'EP2',('pp2',0):'PP2 native',('pp2',1):'PP2 early'}[(m['arm'],m['policy'])]
        lines.append(f"| {label} | {m['throughput_tok_s']:,.1f} | {m['throughput_stdev']:,.1f} |")
    lines += ['',f"提前提交相对原生PP仅 {c['early_gain_pct']:+.3f}%；相对EP为 {c['pp_early_vs_ep_pct']:+.2f}%。"
        '结合主矩阵其他点，未观察到可确认的Prefill提前发送收益。',
        '', '该补测wrapper随后在**非计时**的Q4096原生PP logprob诊断发生CUDA OOM，故保留failed与5/9完成状态。'
        '上表三组完整计时均在失败前完成，已独立重算吞吐并验证零缓存、输入/输出长度、提前发送计数、'
        '冻结源、模型缓存使用及清理；没有补填未完成点。', '',
        '短输入数值检查改为单独的PP诊断：静态显存比例0.60，为logprob临时张量保留空间；'
        '正式性能实验仍为0.94。本诊断没有任何性能样本，使用同权重、精度、输入及原生计算路径，与此前EP诊断比较。', '',
        '| Workload | Comparison | Changed first tokens / comparisons | Max shared-top10 logprob delta | All changed predictions present in both top10 |',
        '|---|---|---:|---:|---|']
    for s in summary:
        lines.append(f"| {s['workload']} | {s['comparison']} | {s['changed_predictions']}/{s['request_comparisons']} | {s['max_abs_shared_top10_logprob_delta']:.6f} | {s['all_changes_within_both_top10']} |")
    lines+=['','B32的Q128、Q512分别有2/32、3/32个请求的首token跨布局不同，两次冷重放均重现；'
        '共享top10候选的最大logprob差为5.08、6.16，且不同预测并不都出现在另一布局的top10中。'
        '因此这不是已有证据可忽略的近似并列，不能直接归因于BF16。PP原生/提前发送的top10 logprob则逐项相同，'
        '单独数值诊断的所有首token也都复现了各自正式性能实验。','',
        '这是原生布局之间的数值差异检查，不能仅凭同一模型名或BF16精度认定等价。'
        '短输入EP/PP的逐请求差异、对应候选概率及诊断输出能否重现正式性能输出，均保存在followups.json和numerical_details.csv；'
        '在原因明确前，短输入的端到端差距应视为该实现的观测值。'
        'Q4096的重复及多主题主实验首token跨布局一致，B8/Q4096独立复测也一致；本次仅验证第一个输出token。', '',
        '## 本轮可以得出的结论','',
        '- 正式计时中PP吞吐均高于EP；Q4096重复/多主题输入的首token跨布局一致，B32吞吐分别高17.6%/19.0%。短输入数值等价尚未成立，不能据此宣布胜负切换边界。',
        '- PP提前activation发送在Prefill上未见稳定收益；Decode专用下一槽位lookahead不适用于本轮Prefill。',
        '- PP Expert GEMM仍更慢，Attention累计时间接近；EP每层的dispatch/combine及完整执行/调度路径共同影响端到端结果。',
        '- 部署判断应分阶段：本轮长Prefill可优先评估原生PP；短输入还需先解决原生EP/PP数值差异，不能把本轮结果写成质量已完全等价的普遍结论。', '',
        f'- [独立补测与数值审计]({OUT/"followups.json"})',
        f'- [数值汇总]({OUT/"numerical_summary.csv"})',
        f'- [数值明细]({OUT/"numerical_details.csv"})','']
    report=OUT/'REPORT.md';text=report.read_text();marker='## 独立定点复测与数值诊断'
    if marker in text:text=text[:text.index(marker)]
    headline=('Q4096长Prefill中，PP early相对EP的B32重复/多主题吞吐高17.6%/19.0%，首token跨布局一致；'
        'B8独立10轮复测高20.3%。提前发送相对原生PP仅约0.15%，未见稳定Prefill收益。'
        '**Q128/Q512的原生EP/PP存在显著数值差异，其吞吐只能作为实现观测，暂不作为等质量部署结论。**')
    title,body=text.split('\n',1)
    if headline not in body:body='\n'+headline+'\n'+body
    report.write_text(title+'\n'+body.rstrip()+'\n\n'+'\n'.join(lines))
    print(json.dumps(dict(focused_comparison=c,numerical_summary=summary,formal_replay=formal_replay),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
