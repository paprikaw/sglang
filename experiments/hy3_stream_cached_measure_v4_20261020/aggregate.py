"""Validate completed native-HiCache cohorts and export comparable metrics."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ('ep8_perf_r5', 'pp2ep4_perf_r1', 'ep8_capacity_r6', 'pp2ep4_capacity_r2')


def aggregate():
    rows, sources, capacities = [], [], {}
    for name in RUNS:
        run = ROOT / 'runs' / name
        result_path = run / 'task/results.json'
        result = json.loads(result_path.read_text())
        cleanup = json.loads((run / 'task/cleanup.json').read_text())
        post_path = sorted((run / 'dashboard_guard/checks').glob('*-post.json'))[-1]
        post = json.loads(post_path.read_text())
        if not (result['passed'] and cleanup['passed'] and post['passed']):
            raise ValueError(f'Incomplete or failed run: {name}')
        if len(result['results']) != 2:
            raise ValueError('Both independently prepared phases required')
        cfg = json.loads((run / 'config.json').read_text())
        if cfg['nodes'] != ['spartan-gpgpu065', 'spartan-gpgpu167']:
            raise ValueError('NIC-matched node pair changed')
        layout = cfg['layout']
        caps = [json.loads(p.read_text())['token_capacity']
                for p in sorted((run / layout / 'capacity').glob('rank*.json'))]
        if len(caps) != 8 or len(set(caps)) != 1:
            raise ValueError('All ranks need consistent native KV capacity')
        if layout in capacities and capacities[layout]['per_attention_replica_tokens'] != caps[0]:
            raise ValueError('Capacity changed between the paired context points')
        capacities[layout] = dict(per_attention_replica_tokens=caps[0],
            attention_replicas=2 if layout == 'ep8' else 1,
            logical_global_tokens=caps[0] * (2 if layout == 'ep8' else 1))
        sources.append(dict(run=name, results=str(result_path), postcheck=str(post_path)))
        for case in result['results']:
            p = case['prefix_tokens_per_request']
            if not (len(case['requests']) == 8 and case['hicache_h2d_timed']
                    and not case['weights_loading_timed']
                    and not case['old_context_constructed_by_model']):
                raise ValueError('Timing/cohort boundary differs')
            proofs = case['native_cpu_ready_gpu_empty']
            if len(proofs) != 8 or any(m['gpu_tokens'] != 0 or m['cpu_tokens'] != p
                for proof in proofs for m in proof['matches']):
                raise ValueError('Prefixes were not fully CPU-resident and GPU-cold')
            if any(r['meta']['cached_tokens_details'] != {'device': 0, 'host': p}
                   for r in case['requests']):
                raise ValueError('Timed requests recomputed old context or hit GPU prefix')
            row = dict(run=name, layout=layout, workload=case['workload'], requests=8,
                cached_context_per_request=p, total_cached_context=8*p,
                new_input_per_request=case['query_tokens_per_request'],
                output_per_request=case['output_tokens_per_request'],
                throughput_tok_s=case['throughput_tok_s'],
                throughput_kind=case['throughput_kind'], cohort_wall_s=case['wall_s'])
            for source, label, scale in [('ttft_s', 'ttft_ms', 1000),
                    ('tpot_s', 'tpot_ms', 1000), ('end_to_end_s', 'end_to_end_s', 1)]:
                for statistic in ('mean', 'median', 'p99'):
                    row[f'{label}_{statistic}'] = (case[source][statistic] * scale
                                                   if case[source] is not None else None)
            rows.append(row)
    rows.sort(key=lambda r: (r['cached_context_per_request'], r['workload'], r['layout']))
    summary = dict(model='tencent/Hy3 BF16 full80', hardware=json.loads((ROOT/'hardware.json').read_text()),
        attention_tp=4, expert_layouts=['EP8 attention DP2', 'PP2 x EP4 attention DP1'],
        graph='disabled both phases', local_prefill_chunk=512,
        max_source_dispatch_tokens_per_rank=128, mem_fraction_static=.975,
        weight_loading='bounded streaming, complete before measurement',
        transport='NVLink within pairs; cross-pair and cross-node RDMA; CE disabled',
        synthetic_context='finite BF16, directly constructed in native HiCache host pool before timing',
        timed='native HiCache H2D, scheduler wait, model forward and token streaming',
        p99_note='Eight-request sample maximum, not a population tail estimate',
        repetitions=1, capacities=capacities, sources=sources, metrics=rows)
    (ROOT/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (ROOT/'comparison.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(rows=len(rows), capacities=capacities, metrics=rows), indent=2))


if __name__ == '__main__':
    aggregate()
