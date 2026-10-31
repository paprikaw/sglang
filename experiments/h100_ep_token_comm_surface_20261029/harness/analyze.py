"""Raw absolute events, aligned critical rank; no mean or median."""
import argparse
import csv
import json
from pathlib import Path
from common import save
from guard_authorization import visible_json


def aligned_critical(samples):
    assert len({s['iteration'] for s in samples})==1
    critical=max(range(len(samples)),key=lambda r:samples[r]['pair_cuda_ms'])
    sample=samples[critical]
    assert abs(sample['dispatch_cuda_ms']+sample['combine_cuda_ms']-sample['pair_cuda_ms'])<1e-3
    return critical,sample


def write_csv(path,rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(run):
    cfg=json.loads((run/'config.json').read_text())
    receipts=run/'artifact_visibility/analysis'
    selected,critical_events,rank_events=[],[],[]
    route_signatures={}
    rank_records=0
    for phase in cfg['phases']:
        ranks=[visible_json(run/phase['id']/f'result_rank{r}.json',receipts) for r in range(phase['world'])]
        rank_records+=len(ranks)
        assert all(r['passed'] and r['buffer_explicitly_destroyed'] and r['model_loaded'] is False for r in ranks)
        assert all(len(r['points'])==len(phase['workloads']) for r in ranks)
        for wi,work in enumerate(phase['workloads']):
            points=[r['points'][wi] for r in ranks]
            assert all(p['id']==work['id'] and p['passed'] and len(p['samples'])==cfg['iterations'] for p in points)
            assert all(p['correctness_before']['passed'] and p['correctness_after_changed_payload']['passed'] for p in points)
            assert all(p['received_assignments_per_rank']==work['tokens']*8//phase['world'] for p in points)
            if work['cuda_graph']:
                assert all(p['graph_correctness']['original_payload']['passed'] and p['graph_correctness']['changed_payload']['passed'] for p in points)
            if work['tokens'] in (8,32768):
                assert all(p['distinct_expert_validation']['passed'] and len(p['skewed_changed_payload_checks'])==2 for p in points)
                assert all(all(c['passed'] for c in p['skewed_changed_payload_checks']) for p in points)
            signatures=[[(c['global_route_sha256'],c['global_token_ids_sha256']) for c in p['chunk_shapes']] for p in points]
            assert all(s==signatures[0] for s in signatures)
            key=(phase['model'],work['tokens'])
            if key in route_signatures:
                assert route_signatures[key]==signatures[0]
            route_signatures[key]=signatures[0]
            base=dict(phase=phase['id'],round=phase['round'],shape=phase['model'],hidden=phase['hidden'],
                experts=phase['experts'],topk=8,ep=phase['world'],mode='graph' if work['cuda_graph'] else 'eager',
                global_original_tokens=work['tokens'],source_tokens_per_rank_total=work['tokens']//phase['world'],
                received_assignments_per_rank=work['tokens']*8//phase['world'],chunks=points[0]['chunks'],
                global_chunk_limit=cfg['global_chunk_limit'],capacity_per_rank=phase['capacity'],transport=phase['transport'],
                global_input_bytes=work['tokens']*phase['hidden']*2,
                global_remote_assignments=points[0]['global_remote_assignments'])
            for iteration in range(cfg['iterations']):
                samples=[p['samples'][iteration] for p in points]
                assert all(s['iteration']==iteration and s['event_1based']==iteration+1 for s in samples)
                critical,sample=aligned_critical(samples)
                place=phase['placement'][critical]
                row=dict(**base,event_1based=iteration+1,critical_rank=critical,node=place['node'],physical_gpu=place['gpu'],
                    dispatch_ms=sample['dispatch_cuda_ms'],combine_ms=sample['combine_cuda_ms'],total_ms=sample['pair_cuda_ms'])
                critical_events.append(row)
                if iteration==0:
                    selected.append(row)
                for rank,s in enumerate(samples):
                    rank_events.append(dict(**base,event_1based=iteration+1,rank=rank,
                        node=phase['placement'][rank]['node'],physical_gpu=phase['placement'][rank]['gpu'],critical_rank=critical,
                        dispatch_ms=s['dispatch_cuda_ms'],combine_ms=s['combine_cuda_ms'],total_ms=s['pair_cuda_ms'],
                        cpu_wall_ms=s['pair_cpu_wall_ms'],per_chunk_events_json=json.dumps(s['chunks'],separators=(',',':'))))
    assert len(selected)==len(cfg['phases'])*26==624
    assert len(rank_events)==87360 and len(critical_events)==18720
    write_csv(run/'absolute_event01.csv',selected)
    write_csv(run/'all_critical_events.csv',critical_events)
    write_csv(run/'all_rank_events.csv',rank_events)
    save(run/'comparison.json',dict(passed=True,formal_points=len(selected),rank_records=rank_records,
        raw_rank_events=len(rank_events),critical_events=len(critical_events),selected_event_1based=1,
        aggregation='none',rounds_separate=True,global_original_tokens=cfg['token_sweep'],
        global_chunk_limit=cfg['global_chunk_limit'],routes_and_chunks_equal_across_EP=True,
        timing='per-event largest aligned rank D+C span; D/C from that same rank; never average samples or rounds',
        model_loaded=False,expert_compute=False,attention_compute=False,
        event_scope='includes D/C completion hooks, receive waits and eager host feed; excludes preparation/barriers/verification',
        caveat='N4096/8192/16384/32768 are measured 2/4/8/16 global chunks on every EP degree, not a single unchunked LL call',
        correctness='exact IDs/counts/BF16 payload/identity, changed Graph input, distinct expert outputs, unequal weights and skewed arrivals'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    analyze(parser.parse_args().run.resolve())
