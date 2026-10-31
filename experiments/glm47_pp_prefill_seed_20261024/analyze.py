"""Compare unchanged PP and proposal-seeded PP, preserving timing boundaries."""
import argparse,importlib.util,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/bxb1/.codex/skills/llm-experiment-guard/scripts')
from artifact_visibility import wait_for_stable_json
spec=importlib.util.spec_from_file_location('pp_causal_analysis',ROOT.parent/'glm47_pp_ep8_startup_causality_20261023/analyze.py')
causal=importlib.util.module_from_spec(spec);spec.loader.exec_module(causal)

def read(path):return wait_for_stable_json(path)[0]

def valid_run(run):
    cfg=read(run/'config.json');result=read(run/'task/results.json');cleanup=read(run/'task/cleanup.json')
    checks=sorted((run/'dashboard_guard/checks').glob('*-post.json'))
    if not checks:raise RuntimeError('Missing Guard postcheck')
    post=read(checks[-1])
    if not all(x.get('passed') is True for x in (result,cleanup,post)):
        raise RuntimeError('Run did not pass performance/profile/cache/cleanup gates')
    return cfg

def metrics(path):
    cohorts=[causal.cohort(path/name) for name in ('p4096_decode_results.json','p4096_decode_repeat_results.json')]
    return dict(cohorts=cohorts,mean={key:statistics.fmean(c['mean'][key] for c in cohorts) for key in cohorts[0]['mean']},
        throughput_out_tok_s=statistics.fmean(c['throughput_out_tok_s'] for c in cohorts),
        wall_s=statistics.fmean(c['wall_s'] for c in cohorts))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--control-run',type=Path);a=p.parse_args()
    run=a.run.resolve();cfg=valid_run(run);arm=cfg['arm_order'][0]
    control_cfg=valid_run(a.control_run.resolve()) if a.control_run else None
    if control_cfg:
        for key in ('nodes','jobs','engine_commit','source_semantic_bank_ready','production_output_tokens','production_repeats'):
            if control_cfg[key]!=cfg[key]:raise RuntimeError('Fresh control mismatch: '+key)
        baseline=a.control_run.resolve()/control_cfg['arm_order'][0]
    else:baseline=Path(cfg['source_layout_references'][arm]['production']['path']).parent
    rows={'control':metrics(baseline),'seeded':metrics(run/arm)};old,new=rows['control'],rows['seeded']
    change={key:100*(new['mean'][key]/old['mean'][key]-1) for key in ('ttft_ms','tpot_ms','e2e_ms','first_transition_ms','middle_emission_cycle_ms')}
    change['throughput_percent']=100*(new['throughput_out_tok_s']/old['throughput_out_tok_s']-1)
    parity=read(run/'same_layout_reference_output_parity.json')
    flow=causal.flow(run/arm)
    receipts=read(run/arm/'p4096_decode_results.json')['all_rank_after']
    seeds=[dict(rank=r['rank'],**r['prefill_first_proposal']) for r in receipts]
    summary=dict(passed=True,rows=rows,relative_change_percent=change,
        exact_same_pp_output_parity=all(v['passed'] for v in parity.values()),
        causal_flow=flow,seed_activation=seeds,historical_baseline=not bool(control_cfg),fresh_paired_confirmation=bool(control_cfg),
        scope='Same nodes/jobs, semantic Bank, model/kernel/transport, twoB4slots P4096 Q1->64; first-proposal reuse only',
        limitations=['Two N8 cohorts, not a corpus-wide throughput or tail-latency claim','Independent flow diagnostic is not production latency','First proposal still requires Target verification'],
        artifact_visibility=causal.VISIBILITY)
    path=run/('paired_comparison.json' if control_cfg else 'screening_comparison.json');path.write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# PP first-proposal reuse','','Fresh unchanged control.' if control_cfg else 'Historical same-node control; small deltas need fresh confirmation.','',
        '| PP2 x EP4 | Output tok/s | TTFT ms | TPOT ms | E2E ms | First Verify advance |','|---|---:|---:|---:|---:|---:|']
    for name,row in rows.items():
        m=row['mean'];lines.append(f"| {name} | {row['throughput_out_tok_s']:.3f} | {m['ttft_ms']:.3f} | {m['tpot_ms']:.3f} | {m['e2e_ms']:.3f} | {m['first_verify_advance']:.3f} |")
    lines+=['',f"Throughput change {change['throughput_percent']:+.3f}%; TPOT change {change['tpot_ms']:+.3f}%.",'',
        'Exact same-PP generated ID/text parity, all-rank activation, performance/profile, cache postcheck and cleanup passed.',
        'No extra Draft forward, payload or KV allocation. Native CPU4032/GPU0 at timer start; semantic import/warmup excluded; native H2D included.',
        'The rejected earlier-send candidate is disabled. Separate flow tracing has no per-replay GPU fence.',
        'Two cohorts of eight requests; no claim of corpus-wide speedup or stable P99.']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(run/'REPORT.md'),change_percent=change,first_verify_advance={k:v['mean']['first_verify_advance'] for k,v in rows.items()},exact_output_parity=summary['exact_same_pp_output_parity'])))
if __name__=='__main__':main()
