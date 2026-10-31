"""CPU-only audit of the four saved EP4/PP4 decode measurements."""
import csv
import hashlib
import json
import statistics as st
from pathlib import Path

ROOT = Path('/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_five_layouts_hicache_20261028')
OUT = Path(__file__).resolve().parent
groups = []
cohorts = []
receipts = []
for run, layout in [('ep_178_r4', 'ep4'), ('ep_178_pp4_r1', 'pp4')]:
    for mode in ['natural_replay', 'cpu_replay']:
        src = ROOT / 'runs' / run / 'results' / f'{layout}_p130304' / f'metrics.{mode}.decode.json'
        raw = src.read_bytes()
        data = json.loads(raw)
        receipts.append({'path': str(src), 'sha256': hashlib.sha256(raw).hexdigest()})
        requests = []
        for repeat, cohort in enumerate(data['rounds']):
            qs = cohort['requests']
            assert len(qs) == 64 and all(q['output_tokens'] == 128 for q in qs)
            span = max(q['finished'] for q in qs) - min(q['started'] for q in qs)
            assert abs(span - cohort['makespan_s']) < 1e-7
            assert abs(8192 / span - cohort['output_tokens_per_s']) < 1e-7
            longest = max(range(len(qs)), key=lambda i: qs[i]['finished'])
            q = qs[longest]
            others = [x for i, x in enumerate(qs) if i != longest]
            cohorts.append(dict(layout=layout, cache=mode, repeat=repeat,
                makespan_s=span, output_tokens_per_s=8192/span,
                request_mean_e2e_s=st.mean(x['e2e_s'] for x in qs),
                ttft_mean_s=st.mean(x['ttft_s'] for x in qs),
                tpot_mean_ms=st.mean(x['tpot_ms'] for x in qs),
                last_request_index=longest, last_request_ttft_s=q['ttft_s'],
                last_request_e2e_s=q['e2e_s'],
                seconds_after_other_63_finished=q['finished']-max(x['finished'] for x in others),
                client_start_spread_s=max(x['started'] for x in qs)-min(x['started'] for x in qs)))
            requests.extend(qs)
        rates = [x['output_tokens_per_s'] for x in data['rounds']]
        groups.append(dict(layout=layout, cache=mode,
            round_rates=rates, round_mean_tokens_per_s=st.mean(rates),
            round_median_tokens_per_s=st.median(rates),
            request_mean_ttft_s=st.mean(q['ttft_s'] for q in requests),
            request_mean_tpot_ms=st.mean(q['tpot_ms'] for q in requests),
            request_mean_e2e_s=st.mean(q['e2e_s'] for q in requests),
            selected_optimization=data.get('selected_optimization')))

pp = [g for g in groups if g['layout'] == 'pp4']
assert all(g['selected_optimization']['policy'] == 0 and g['selected_optimization']['microbatch'] == 16 for g in pp)
cpu = {g['layout']: g for g in groups if g['cache'] == 'cpu_replay'}
result = dict(scope='Saved 4x H100, Qwen3.6-35B-A3B BF16, B64/P130304/Q1/O128; no new GPU run',
    groups=groups, cohorts=cohorts, sources=receipts,
    cpu_pp_advantage_from_mean=cpu['pp4']['round_mean_tokens_per_s']/cpu['ep4']['round_mean_tokens_per_s']-1,
    cpu_pp_advantage_from_median=cpu['pp4']['round_median_tokens_per_s']/cpu['ep4']['round_median_tokens_per_s']-1,
    slow_round_kept=True,
    causal_status='Request-tail localization and matched PP settings verified; origin of delayed admission and cache-only causal effect remain unresolved')
(OUT/'cache_reversal_analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
with (OUT/'cache_reversal_cohorts.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(cohorts[0]))
    writer.writeheader()
    writer.writerows(cohorts)
print(json.dumps({'groups':len(groups), 'cohorts':len(cohorts),
    'cpu_mean_advantage':result['cpu_pp_advantage_from_mean'],
    'cpu_median_advantage':result['cpu_pp_advantage_from_median'],
    'pp_policy':'both native, microbatch=16'}, ensure_ascii=False))
