"""Local post-run analysis. Does not launch, modify, or time an experiment."""
import argparse
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def latency(rows):
    gaps=[(r['token_events'][1][1]-r['token_events'][0][1])*1000 for r in rows]
    by_slot=[]
    for i in range(2):
        slot=[g for g,r in zip(gaps,rows) if r['request_index']//4==i]
        by_slot.append(statistics.mean(slot))
    steady=[(b[1]-a[1])*1000 for r in rows for a,b in zip(r['token_events'][2:-1],r['token_events'][3:])]
    return dict(first_gap_ms=statistics.mean(gaps),first_gap_by_slot_ms=by_slot,
                steady_chunk_gap_median_ms=statistics.median(steady),
                first_chunk_counts=sorted({r['token_events'][0][0] for r in rows}),
                second_chunk_counts=sorted({r['token_events'][1][0] for r in rows}))


def summarize(point):
    return dict(cohort=point['cohort'],tpot_ms=point['tpot_s']['mean']*1000,
        ttft_ms=point['ttft_s']['mean']*1000,e2e_ms=point['end_to_end_s']['mean']*1000,
        throughput_tok_s=point['throughput_tok_s'],
        accept_length=point['spec_accept_length']['mean'],verify_count=point['spec_verify_ct']['mean'],
        **latency(point['requests']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='prefill_graph_ab_r1');a=p.parse_args()
    run=ROOT/'runs'/a.name;cfg=json.loads((run/'config.json').read_text());arm=run/cfg['arm_order'][0]
    complete=json.loads((arm/'arm_result.json').read_text())
    if not all(complete.get(k) for k in ('passed','performance_valid','profile_valid','cleanup_verified')):
        raise RuntimeError('Requires qualified performance, profile and cleanup; failed attempts are not results')
    names=['control_a_decode_results.json','p4096_decode_results.json','p4096_decode_repeat_results.json','control_b_decode_results.json']
    points=[json.loads((arm/n).read_text()) for n in names]
    controls=json.loads((arm/'same_process_controls.json').read_text())
    repeats=json.loads((arm/'p4096_decode_repeatability.json').read_text())
    if not controls['passed'] or not repeats['passed']:raise RuntimeError('Missing exact token/text parity')
    data=[summarize(p) for p in points]
    average=lambda rows:{k:statistics.mean(x[k] for x in rows) for k in ('tpot_ms','ttft_ms','e2e_ms','throughput_tok_s','first_gap_ms','accept_length','verify_count')}
    before=average([data[0],data[3]]);after=average([data[1],data[2]])
    flow=[]
    for rank in (0,4):
        trace=json.loads((arm/f'unfenced_flow_rank{rank}.json').read_text())
        target=[f for f in trace['forwards'] if f['phase'] in ('target_prefill','target_verify')]
        flow.append(dict(rank=rank,prefill_ms=[f['elapsed_ms'] for f in target if f['phase']=='target_prefill'],
            first_target_forwards=[{k:f[k] for k in ('phase','start_ms','end_ms','elapsed_ms')} for f in target[:6]],
            per_replay_fences=trace['per_replay_fences'],same_gpu_clock_only=trace['same_gpu_clock_only']))
    report=dict(passed=True,run=str(run),cohorts=data,control_mean=before,candidate_mean=after,
        change_percent={k:100*(after[k]/before[k]-1) for k in before},candidate_flow=flow,
        correctness='Exact greedy token IDs and text, ABBA same process and candidate repeat',
        first_gap_boundary='Client first token chunk to next token-bearing chunk; first/second counts retained',
        claim_scope='B8, two B4 slots, native CPU4032/GPU0 start, Q65 residual target Prefill, output64, fixed real semantic corpus',
        startup_solved_not_inferred_from_small_steady_state_gain=True)
    (ROOT/'analysis/summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# PP residual Prefill Graph transition A/B/B/A','',report['claim_scope'],'',
        '| Cohort | TTFT ms | TPOT ms | E2E ms | output tok/s | First gap ms |',
        '|---|---:|---:|---:|---:|---:|']
    for row in data:lines.append('| {cohort} | {ttft_ms:.2f} | {tpot_ms:.3f} | {e2e_ms:.2f} | {throughput_tok_s:.2f} | {first_gap_ms:.2f} |'.format(**row))
    lines+=['',report['correctness'],'','Actual per-rank Prefill Graph counts are in each result receipt.',
        'Timing includes native HiCache→GPU; excludes immutable bank import and model loading.',
        'Graph flow uses GPU-local clocks and a single final fence; no all-GPU-idle inference.','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(before=before,after=after,change_percent=report['change_percent'],flow=flow),indent=2))


if __name__=='__main__':main()
