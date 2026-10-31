"""Read-only paired analysis; never turn stream gaps into device-idle time."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'glm47_mtp_verify_fixed_ep8_pp2ep4_20261023'
spec = importlib.util.spec_from_file_location('previous_pair_reporting', PREVIOUS / 'report_pair.py')
REPORT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(REPORT)
ARMS = ('pp2ep4_baseline', 'pp2ep4_local_tail')


def read(path):
    return json.loads(Path(path).read_text())


def source(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def stats(values):
    values = sorted(values)
    return dict(count=len(values), mean=st.fmean(values), median=st.median(values),
                sample_max=values[-1]) if values else None


def flow(path):
    data = read(path)
    assert data['passed'] and data['per_replay_fences'] == 0
    replays = data['replays']
    target = [r for r in replays if r['phase'] == 'target_verify']
    by_request = defaultdict(list)
    for row in target:
        by_request[tuple(row['request_ids'])].append(row)
    cycles, steady = [], []
    for rows in by_request.values():
        local = [b['start_ms']-a['start_ms'] for a,b in zip(rows,rows[1:])]
        cycles.extend(local)
        steady.extend(local[2:-2])
    calls = defaultdict(list)
    active_steps={r['pp_microstep'] for r in target}
    for row in data['host_calls']:
        # Idle event-loop turns must not dilute the active-path API duration.
        if row['microstep'] not in active_steps:
            continue
        calls[row['name']].append((row['cpu_end_ns']-row['cpu_start_ns'])/1e6)
    phases = defaultdict(list)
    for row in replays:
        phases[row['phase']].append(row['elapsed_ms'])
    gaps = [b['start_ms']-a['end_ms'] for a,b in zip(replays,replays[1:])]
    return dict(rank=data['rank'], pp_rank=data['pp_rank'],
                request_sets=[list(k) for k in by_request],
                same_slot_cycle_ms=stats(cycles), steady_same_slot_cycle_ms=stats(steady),
                graph_phase_ms={k:stats(v) for k,v in phases.items()},
                between_model_graphs_current_stream_gap_ms=stats(gaps),
                host_api_ms={k:stats(v) for k,v in calls.items()},
                source=source(path),
                draft_replays_observed=any(r['phase'].startswith('draft') for r in replays),
                caveat='GPU intervals are same-device current-stream intervals including dependencies. This version captures target replays; direct draft-backend calls lack a phase and are not individually captured. Gaps include draft work and are not GPU idle; host API includes serialization, scheduling, waits and enqueue, not just NIC time.')


def analyze(run):
    run=Path(run).resolve()
    task=read(run/'task/results.json')
    cleanup=read(run/'task/cleanup.json')
    assert task['passed'] and cleanup['passed'] and task['candidate_baseline_output_parity']['passed']
    result=dict(run=str(run), sources=[source(run/'config.json'),source(run/'task/results.json')],
                same_semantic_bank=True, unprofiled_performance=True, arms={},
                candidate_exact_matches_baseline=True,
                lossless_against_mtp_off_qualified=False,
                known_caveat='The pre-existing experimental fixed-Verify path did not exactly match MTP-off. This protocol optimization must exactly preserve the fixed-Verify baseline; it does not repair that separate numerical issue.')
    for arm in ARMS:
        base=run/arm
        owned=read(base/'arm_result.json')
        assert owned['performance_valid'] and owned['profile_valid'] and owned['cleanup_verified']
        cohorts=[read(base/name) for name in ('p4096_decode_results.json','p4096_decode_repeat_results.json')]
        perf={key:stats([row[key] for row in cohorts]) for key in ('throughput_tok_s','wall_s')}
        for key in ('ttft_s','tpot_s','end_to_end_s','spec_accept_length','spec_accept_rate','spec_verify_ct'):
            perf[key]=stats([row[key]['mean'] for row in cohorts])
        after=cohorts[-1]['all_rank_after']
        result['arms'][arm]=dict(performance=perf,
            flow=[flow(base/f'unfenced_flow_rank{rank}.json') for rank in range(8)],
            flow_overhead=read(base/'unfenced_flow_results.json')['trace_overhead_ratio'],
            components=REPORT.aggregate_activity(base/'stage_activity_summary.json'),
            local_feedback=[dict(rank=r['rank'],**r['local_tail_feedback']) for r in after],
            sources=[source(base/name) for name in ('p4096_decode_results.json','p4096_decode_repeat_results.json',
                                                  'stage_activity_summary.json','unfenced_flow_results.json')])
    a,b=[result['arms'][arm]['performance'] for arm in ARMS]
    result['candidate_vs_baseline']=dict(tpot_ratio=b['tpot_s']['mean']/a['tpot_s']['mean'],
        throughput_ratio=b['throughput_tok_s']['mean']/a['throughput_tok_s']['mean'])
    old=read(PREVIOUS/'summary.json')
    ep=old['arms']['ep8_mtp_fixed']['production_aggregate']
    result['historical_same_bank_ep8']=dict(source=source(PREVIOUS/'summary.json'),
        tpot_ms=ep['tpot_ms']['mean'],throughput_tok_s=ep['throughput_tok_s']['mean'],
        fresh_control=False,exact_cross_topology_output_equivalence_qualified=False)
    result['candidate_vs_historical_ep8']=dict(
        tpot_ratio=1000*b['tpot_s']['mean']/ep['tpot_ms']['mean'],
        throughput_ratio=b['throughput_tok_s']['mean']/ep['throughput_tok_s']['mean'])
    return result


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);args=p.parse_args()
    result=analyze(args.run)
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ('candidate_vs_baseline','candidate_vs_historical_ep8')},indent=2))


if __name__=='__main__':main()
