"""Read original requests and unfenced timelines; never launch GPU work."""
import argparse, hashlib, json, statistics as st, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0, "/home/bxb1/.codex/skills/llm-experiment-guard/scripts")
from artifact_visibility import wait_for_stable_json
VISIBILITY=[]
def read_json(path):
    value, receipt = wait_for_stable_json(path)
    VISIBILITY.append(receipt)
    return value
SOURCES={
 'ep8':ROOT.parent/'glm47_mtp_verify_fixed_ep8_pp2ep4_20261023/runs/paired_r1/ep8_mtp_fixed',
 'pp2ep4':ROOT.parent/'glm47_pp_stage_balance_20261023/runs/balance_r1/pp2ep4_balanced'}

def descriptor(path):
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def cohort(path):
    data=read_json(path); rows=[]
    for request in data['requests']:
        events=request['token_events']
        assert events[0][0]==1 and events[-1][0]==64 and len(events)>5
        first_ms=(events[1][1]-events[0][1])*1000
        rest_ms=(events[-1][1]-events[1][1])*1000
        tpot_ms=(first_ms+rest_ms)/63
        assert abs(tpot_ms-request['tpot_s']*1000)<1e-7
        intervals=[(b[1]-a[1])*1000 for a,b in zip(events,events[1:])]
        rows.append(dict(request_index=request['request_index'],ttft_ms=events[0][1]*1000,
            tpot_ms=tpot_ms,e2e_ms=events[-1][1]*1000,first_transition_ms=first_ms,
            first_transition_tpot_contribution_ms=first_ms/63,
            remaining_tpot_contribution_ms=rest_ms/63,
            first_verify_advance=events[1][0]-events[0][0],
            middle_emission_cycle_ms=st.fmean(intervals[2:-2])))
    assert len(rows)==8
    keys=[k for k in rows[0] if k!='request_index']
    means={k:st.fmean(r[k] for r in rows) for k in keys}
    return dict(source=descriptor(path),mean=means,requests=rows,wall_s=data['wall_s'],throughput_out_tok_s=512/data['wall_s'],
                first_group_transition_ms=st.fmean(r['first_transition_ms'] for r in rows[:4]),
                second_group_transition_ms=st.fmean(r['first_transition_ms'] for r in rows[4:]))

def compare(arms):
    result={}
    for name,path in arms.items():
        cohorts=[cohort(path/f) for f in ['p4096_decode_results.json','p4096_decode_repeat_results.json']]
        means={k:st.fmean(c['mean'][k] for c in cohorts) for k in cohorts[0]['mean']}
        result[name]=dict(cohorts=cohorts,mean=means,mean_wall_s=st.fmean(c['wall_s'] for c in cohorts),mean_throughput_out_tok_s=st.fmean(c['throughput_out_tok_s'] for c in cohorts))
    a,b=result['ep8']['mean'],result['pp2ep4']['mean']
    delta={k:b[k]-a[k] for k in a}
    assert abs(delta['tpot_ms']-delta['first_transition_tpot_contribution_ms']-delta['remaining_tpot_contribution_ms'])<1e-8
    return dict(arms=result,pp_minus_ep8=delta,exact_tpot_identity='TPOT=(first_SSE_gap+remaining_elapsed_time)/63',
        middle_cycle_scope='Per-request SSE interval mean, omitting first2/last2; not per-token latency or GPU idle',
        cross_layout_output_equivalence_qualified=False)

def flow(path):
    data=read_json(path/'unfenced_flow_results.json')
    requests={r['rid']:r['request_index'] for r in data['requests']}
    output=[]
    for rank in range(8):
        p=path/f'unfenced_flow_rank{rank}.json';trace=read_json(p)
        forwards=trace['forwards'];target=[r for r in forwards if r['phase'].startswith('target_')]
        origin=target[0]['start_ms'];first_cpu=min(r['cpu_start_ns'] for r in trace['batches'])
        def compact(row):
            return {k:row.get(k) for k in ['index','phase','pp_microstep','batch_size','input_shape','elapsed_ms']}|dict(
                request_indices=[requests.get(r,-1) for r in row.get('request_ids',[])],
                start_relative_ms=row['start_ms']-origin,end_relative_ms=row['end_ms']-origin,
                cpu_start_relative_ms=(row['cpu_start_ns']-first_cpu)/1e6,
                cpu_end_relative_ms=(row['cpu_end_ns']-first_cpu)/1e6)
        graphs=trace['replays'];groups={}
        for row in graphs:
            if row['phase']!='target_verify':continue
            key=tuple(sorted(requests.get(r,-1) for r in row['request_ids']))
            groups.setdefault(key,[]).append(row)
        cycles=[]
        for key,rows in groups.items():
            durations=[]
            for a,b in list(zip(rows,rows[1:]))[2:-2]:
                inside=[r for r in graphs if r['start_ms']>=a['start_ms']-.001 and r['end_ms']<=b['start_ms']+.001]
                # Replay intervals on the same local stream must not overlap.
                ordered=sorted(inside,key=lambda r:r['start_ms'])
                if any(y['start_ms']<x['end_ms']-.01 for x,y in zip(ordered,ordered[1:])):
                    raise RuntimeError('Cannot sum overlapping replay intervals')
                cycle=b['start_ms']-a['start_ms'];covered=sum(r['elapsed_ms'] for r in inside)
                durations.append(dict(cycle_ms=cycle,actual_graph_covered_ms=covered,
                    uncovered_current_stream_ms=cycle-covered,
                    target_graph_ms=sum(r['elapsed_ms'] for r in inside if r['phase']=='target_verify'),
                    draft_graph_ms=sum(r['elapsed_ms'] for r in inside if not r['phase'].startswith('target_'))))
            if durations:cycles.append(dict(request_indices=list(key),mean={k:st.fmean(r[k] for r in durations) for k in durations[0]},median={k:st.median(r[k] for r in durations) for k in durations[0]},samples=durations,cycles=len(durations)))
        host=[]
        for r in trace['host_calls']:
            if (r['cpu_end_ns']>=first_cpu-2_000_000 and r['cpu_start_ns']<=first_cpu+1_000_000_000 and len(host)<400):
                host.append(dict(name=r['name'],microstep=r.get('microstep'),
                    request_indices=[requests.get(x,-1) for x in r.get('request_ids',[])],
                    cpu_start_relative_ms=(r['cpu_start_ns']-first_cpu)/1e6,
                    cpu_end_relative_ms=(r['cpu_end_ns']-first_cpu)/1e6,
                    elapsed_ms=(r['cpu_end_ns']-r['cpu_start_ns'])/1e6))
        output.append(dict(rank=rank,source=descriptor(p),initial_forward_order=[compact(r) for r in forwards[:16]],
            initial_batch_order=[compact(r) for r in trace['batches'][:8]],host_startup=host,
            same_request_group_cycles=cycles,all_model_forwards=trace['all_model_forwards']))
    return dict(ranks=output,source=descriptor(path/'unfenced_flow_results.json'),
        diagnostic_output_parity=dict(passed=data['output_parity'].get('passed'),request_count=len(data['output_parity'].get('requests',[]))),trace_overhead_ratio=data['trace_overhead_ratio'],
        same_gpu_clock_only=True,no_gpu_idle_fraction_claim=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);parser.add_argument('--ep8-run',type=Path);parser.add_argument('--pp-only',type=Path);args=parser.parse_args()
    result=dict(historical=compare(SOURCES),new_gpu_data_used=False)
    if args.pp_only:
        result.update(new_gpu_data_used=True,pp_flow=flow(args.pp_only/'pp2ep4_flow'))
    if args.run:
        ep_run=args.ep8_run or args.run
        result.update(new_gpu_data_used=True,current=compare(dict(ep8=ep_run/'ep8_flow',pp2ep4=args.run/'pp2ep4_flow')),
            flow=dict(ep8=flow(ep_run/'ep8_flow'),pp2ep4=flow(args.run/'pp2ep4_flow')))
    result['artifact_visibility']=VISIBILITY
    (ROOT/'causal_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(report=str(ROOT/'causal_evidence.json'),historical_pp_minus_ep8=result['historical']['pp_minus_ep8'],
        fresh_pp_minus_ep8=result.get('current',{}).get('pp_minus_ep8'))))
if __name__=='__main__':main()
