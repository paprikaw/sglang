"""Independently compare every exported event with the immutable rank records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return (row['phase'],row['mode'],int(row['global_original_tokens']),int(row['event_1based']))


def audit(run):
    cfg=json.loads((run/'config.json').read_text())
    spec=json.loads((run/'dashboard_spec.json').read_text())
    wrapper=str(Path(spec['sources'][0]['worktree'])/'deep_ep/__init__.py')
    verdict=json.loads((run/'comparison.json').read_text())
    assert verdict['passed'] and verdict['aggregation']=='none'
    expected={}
    chosen={}
    for phase in cfg['phases']:
        rank_records=[json.loads((run/phase['id']/f'result_rank{rank}.json').read_text()) for rank in range(phase['world'])]
        for rank,record in enumerate(rank_records):
            route=record['route']
            assert record['passed'] and record['buffer_explicitly_destroyed']
            assert route['native']['sha256']==cfg['native']['sha256']
            assert route['wrapper_path']==wrapper
            assert route['hidden']==phase['hidden'] and route['experts']==phase['experts'] and route['topk']==8
            assert len(route['cpu_affinity'])==64 and route['cpus_per_node_step']==64
            assert route['nvshmem_qp_depth']==4096
            assert route['dtype']=='bfloat16' and not any(route[k] for k in ('use_fp8','use_logfmt','ce_enabled','pair_only'))
            assert route['actual_nic_backend']['actual_backend']=='cpu_host_memory'
            assert route['physical_gpu']==phase['placement'][rank]['gpu']
            assert route['node']==phase['placement'][rank]['node']
            assert len(record['points'])==26
            for point in record['points']:
                assert point['passed'] and point['correctness_before']['passed'] and point['correctness_after_changed_payload']['passed']
                assert point['global_chunk_limit']==2048
                n=point['global_original_tokens']
                assert point['chunks']==(n+2047)//2048
                assert point['received_assignments_per_rank']==8*n//phase['world']
                assert len(point['samples'])==30
                for event in point['samples']:
                    k=(phase['id'],'graph' if point['cuda_graph'] else 'eager',n,event['event_1based'])
                    expected[k+(rank,)]=event
        for work in phase['workloads']:
            mode='graph' if work['cuda_graph'] else 'eager'
            for event in range(1,31):
                k=(phase['id'],mode,work['tokens'],event)
                selected=max(range(phase['world']),key=lambda r:expected[k+(r,)]['pair_cuda_ms'])
                chosen[k]=(selected,expected[k+(selected,)])
    assert len(expected)==87360 and len(chosen)==18720
    with (run/'all_rank_events.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    seen=set()
    for row in rows:
        k=key(row)+(int(row['rank']),)
        assert k not in seen
        seen.add(k)
        sample=expected[k]
        for exported,raw in (('dispatch_ms','dispatch_cuda_ms'),('combine_ms','combine_cuda_ms'),('total_ms','pair_cuda_ms'),('cpu_wall_ms','pair_cpu_wall_ms')):
            assert float(row[exported])==sample[raw]
        assert json.loads(row['per_chunk_events_json'])==sample['chunks']
        assert int(row['critical_rank'])==chosen[key(row)][0]
    assert seen==set(expected)
    for filename,count,first_only in (('all_critical_events.csv',18720,False),('absolute_event01.csv',624,True)):
        seen=set()
        with (run/filename).open() as stream:
            rows=list(csv.DictReader(stream))
        assert len(rows)==count
        for row in rows:
            k=key(row)
            assert k not in seen
            seen.add(k)
            if first_only:
                assert k[-1]==1
            critical,sample=chosen[k]
            assert int(row['critical_rank'])==critical
            for exported,raw in (('dispatch_ms','dispatch_cuda_ms'),('combine_ms','combine_cuda_ms'),('total_ms','pair_cuda_ms')):
                assert float(row[exported])==sample[raw]
            assert abs(float(row['dispatch_ms'])+float(row['combine_ms'])-float(row['total_ms']))<1e-3
    frozen=json.loads((run/'frozen_inputs.json').read_text())['files']
    for item in frozen:
        assert sha(item['path'])==item['sha256']
    cleanup=json.loads((run/'task/cleanup.json').read_text())
    assert cleanup['passed'] and cleanup['gpu_idle'] and cleanup['model_processes_exited']
    assert len(cleanup['evidence'])==32
    assert all(p['returncode']==0 and p['gpu_query']['passed'] and p['gpu_query']['post_exit_only'] for p in cleanup['evidence'])
    result=dict(passed=True,raw_rank_events_exactly_recomputed=87360,aligned_critical_events_recomputed=18720,
        displayed_events_exactly_recomputed=624,aggregation='none',selected_event_1based=1,
        cpu_affinity_count_all_ranks=64,per_node_cleanups=32,all_frozen_hashes_unchanged=True,
        native_sha256=cfg['native']['sha256'],csv_sha256={name:sha(run/name) for name in ('absolute_event01.csv','all_critical_events.csv','all_rank_events.csv')})
    (run/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    print(json.dumps(audit(parser.parse_args().run.resolve())))
