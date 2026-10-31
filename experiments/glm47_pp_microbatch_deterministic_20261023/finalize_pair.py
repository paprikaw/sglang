"""Read-only paired report; never accept incomplete/synthetic/changed cohorts."""
import argparse
import json
from pathlib import Path


def load_arm(run,mode):
    cfg=json.loads((run/'config.json').read_text())
    result=json.loads((run/'task/results.json').read_text())
    cleanup=json.loads((run/'task/cleanup.json').read_text())
    if cfg['mode']!=mode or not result['passed'] or not cleanup['passed'] or result['valid_points']!=1:
        raise ValueError('Actual single-arm result and owned cleanup must pass')
    if len(result['arms'])!=1:raise ValueError('Unexpected arm count')
    arm=result['arms'][0]
    if not arm['passed'] or not arm['cleanup_verified'] or len(arm['results'])!=1:
        raise ValueError('Incomplete arm')
    point=arm['results'][0]
    if (point['synthetic_kv'] or not point['old_context_constructed_by_model']
        or point['request_count']!=8 or point['microbatch_mode']!=mode
        or not point['actual_microbatch_scheduler_proof']['passed']
        or cfg.get('deterministic_inference') is not True
        or not point['actual_microbatch_scheduler_proof'].get('deterministic_runtime',{}).get('passed')):
        raise ValueError('Actual semantic microbatch proof missing')
    return cfg,point


def compare(baseline,candidate):
    left,a=load_arm(baseline,'baseline');right,b=load_arm(candidate,'candidate')
    for key in ('engine_commit','engine_root','nodes','layouts','prefix_lengths',
                'production_output_tokens','profile_output_tokens','semantic_inputs','deterministic_inference'):
        if left[key]!=right[key]:raise ValueError('Unmatched config field '+key)
    for key in ('prefix_tokens_per_request','actual_cached_tokens_per_request',
                'first_forward_new_tokens_per_request','output_tokens_per_request'):
        if a[key]!=b[key]:raise ValueError('Unmatched workload '+key)
    digests=lambda p:[row['sha256'] for row in p['semantic_input_manifest']['requests']]
    if digests(a)!=digests(b):raise ValueError('Different semantic prompts')
    by_id=lambda p:{row['request_index']:row for row in p['requests']}
    aa,bb=by_id(a),by_id(b)
    if set(aa)!=set(range(8)) or set(bb)!=set(range(8)):raise ValueError('Incomplete request-index matching')
    parity=[]
    for index in range(8):
        x,y=aa[index],bb[index]
        tokens_equal=x.get('output_ids')==y.get('output_ids') and bool(x.get('output_ids'))
        text_equal=x.get('output_text')==y.get('output_text') and isinstance(x.get('output_text'),str)
        parity.append(dict(request_index=index,token_ids_equal=tokens_equal,text_equal=text_equal,
            baseline_output_ids=x.get('output_ids'),candidate_output_ids=y.get('output_ids'),
            baseline_text=x.get('output_text'),candidate_text=y.get('output_text')))
    pair_equal=all(row['token_ids_equal'] and row['text_equal'] for row in parity)
    repeatable=all(point.get('repeatability',{}).get('passed') is True for point in (a,b))
    correct=pair_equal and repeatable
    value=dict(protocol='same_engine_semantic_pp_microbatch_flag_ab',
        source_runs=dict(baseline=str(baseline.resolve()),candidate=str(candidate.resolve())),
        engine_commit=left['engine_commit'],baseline=a,candidate=b,
        candidate_tpot_speedup=a['tpot_s']['mean']/b['tpot_s']['mean'],
        candidate_throughput_speedup=b['throughput_tok_s']/a['throughput_tok_s'],
        synthetic_kv=False,numerical_parity_with_non_speculative_not_claimed=True,
        correctness_status='greedy_pair_and_repeat_parity_passed' if correct else 'correctness_unconfirmed',
        repeatability_status=dict(baseline=a.get('repeatability'),candidate=b.get('repeatability')),
        greedy_pair_outputs_equal=pair_equal,
        performance_comparison_claim_valid=correct,greedy_pair_parity=parity,
        actual_slot_composition=dict(baseline=a['actual_microbatch_scheduler_proof'],candidate=b['actual_microbatch_scheduler_proof']))
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();value=compare(args.baseline,args.candidate)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'comparison.json').write_text(json.dumps(value,indent=2)+'\n')
    a,b=value['baseline'],value['candidate']
    lines=['# PP2 × EP4 semantic microbatch A/B','',
        'Same source revision, full canonical CPU weight banks, real semantic Prefill KV.',
        'HiCache CPU-to-GPU load is timed; model loading, Prefill and warmup are not.',
        '', '| Metric | Baseline | Candidate |','|---|---:|---:|']
    for label,key,scale in [('TTFT (s)','ttft_s',1),('TPOT (ms)','tpot_s',1000),
        ('End-to-end (s)','end_to_end_s',1),('Draft acceptance (%)','spec_accept_rate',100)]:
        lines.append(f'| {label} | {a[key]["mean"]*scale:.3f} | {b[key]["mean"]*scale:.3f} |')
    lines.append(f'| Global output tok/s | {a["throughput_tok_s"]:.3f} | {b["throughput_tok_s"]:.3f} |')
    lines+=['','Profile cohorts are separate. P99 from eight requests is the sample maximum.','']
    lines+=['Correctness: '+value['correctness_status'],
        'Performance data are retained even on mismatch; no validated speedup claim until greedy parity passes.','']
    (args.out/'RESULTS.md').write_text('\n'.join(lines));print('\n'.join(lines))
    return 0 if value['performance_comparison_claim_valid'] else 2


if __name__=='__main__':raise SystemExit(main())
