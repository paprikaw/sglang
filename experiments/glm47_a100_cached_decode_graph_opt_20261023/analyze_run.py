"""Summarize paired official SSE metrics and fixed-token steady cadence."""
import argparse,json,statistics
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def arm_summary(run,arm):
    result=read(run/arm/'arm_result.json')
    if not result['passed'] or not result['cleanup_verified']:
        raise RuntimeError('Arm did not pass and clean: '+arm)
    row=read(run/arm/'p81920_decode_results.json')
    if not row['actual_graph_proof']['passed']:
        raise RuntimeError('Missing actual replay evidence')
    steady=[];intervals=[]
    for req in row['requests']:
        if req['completion_tokens']!=128 or req['meta'].get('num_retractions')!=0:
            raise RuntimeError('Incomplete output / retraction')
        ts=dict(req['token_events'])
        if not all(k in ts for k in range(1,129)):
            raise RuntimeError('Token-level stream incomplete')
        steady.append((ts[112]-ts[16])/96*1000)
        intervals.extend((ts[k]-ts[k-1])*1000 for k in range(17,113))
    caps=[read(run/arm/'capacity'/f'rank{r}.json') for r in range(8)]
    return dict(layout=arm,request_count=8,prefix_tokens=81920,total_context_tokens=655360,
        new_input_tokens=1,output_tokens=128,throughput_output_tok_s=row['throughput_tok_s'],
        ttft_mean_s=row['ttft_s']['mean'],tpot_mean_ms=row['tpot_s']['mean']*1000,
        tpot_median_ms=row['tpot_s']['median']*1000,tpot_p99_ms=row['tpot_s']['p99']*1000,
        e2e_mean_s=row['end_to_end_s']['mean'],steady_tokens_16_to_112_mean_ms=statistics.mean(steady),
        steady_all_request_interval_median_ms=statistics.median(intervals),
        actual_graph_replays=[r['actual_decode_replays'] for r in row['actual_graph_proof']['rank_proofs']],
        capacity_tokens=[c['token_capacity'] for c in caps],all_request_retractions_zero=True,
        source_result=str((run/arm/'p81920_decode_results.json').resolve()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    total=read(a.run/'task/results.json')
    if not total['passed']:raise RuntimeError('Paired run incomplete')
    arms=[arm_summary(a.run,x) for x in ('ep8','pp2ep4')]
    ep,pp=arms
    out=dict(schema_version=1,passed=True,arms=arms,
        pp_tpot_change_percent=(pp['tpot_mean_ms']/ep['tpot_mean_ms']-1)*100,
        pp_steady_cadence_change_percent=(pp['steady_tokens_16_to_112_mean_ms']/ep['steady_tokens_16_to_112_mean_ms']-1)*100,
        pp_output_throughput_change_percent=(pp['throughput_output_tok_s']/ep['throughput_output_tok_s']-1)*100,
        no_profiler=True,decode_graph_enabled_for_both=True,synthetic_native_cpu_hicache=True,
        timing='CPU HiCache ready / target GPU prefix empty; H2D and admission included; weights and warmup excluded',
        caveats=['One paired cohort; eight requests are not eight independent experiment repetitions.',
                 'Official mean TPOT retains startup admission gaps after first token.',
                 'Steady token16..112 interval is secondary, not a replacement for official TPOT.',
                 'Graph receipts prove actual replay submission, not a new independent numerical kernel validation.'])
    (a.run/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
