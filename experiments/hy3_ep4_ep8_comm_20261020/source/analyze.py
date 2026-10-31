"""Recompute aligned communication timings from every rank, without a model."""
import argparse
import csv
import json
from pathlib import Path
from common import save, NATIVE_SHA
from helpers import aggregate
from guard_authorization import visible_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    cfg = json.loads((run/'config.json').read_text())
    flat, points = [], []
    for phase in cfg['phases']:
        records = [visible_json(run/phase['id']/f'result_rank{r}.json',
                               run/'artifact_visibility/analysis', 30)
                   for r in range(phase['world'])]
        assert all(r['passed'] and r['native_sha256'] == NATIVE_SHA
                   and r['buffer_explicitly_destroyed'] for r in records)
        for i, work in enumerate(cfg['workloads']):
            rows = [r['points'][i] for r in records]
            source = work['source_tokens_per_rank']
            total = source*phase['world']
            for row in rows:
                assert row['passed'] and not row['cuda_graph']
                assert row['global_tokens'] == total and row['source_tokens_per_rank'] == source
                assert row['shape']['local_tokens'] == source
                assert row['shape']['per_owner_assignment_counts'] == [source*8]*phase['world']
                assert len(row['samples']) == cfg['iterations']
                assert row['correctness_before']['passed'] and row['correctness_after_changed_payload']['passed']
                for sample in row['samples']:
                    assert abs(sample['pair_cuda_ms'] - sample['dispatch_cuda_ms']
                               - sample['combine_cuda_ms']) < .01
            samples = [r['samples'] for r in rows]
            metrics = {key: aggregate(samples, key) for key in
                       ('pair_cuda_ms', 'pair_cpu_wall_ms', 'dispatch_cuda_ms', 'combine_cuda_ms')}
            points.append(dict(phase=phase['id'], workload=work['id'], world=phase['world'],
                               nodes=len(phase['nodes']), source_tokens_per_rank=source,
                               global_tokens=total, metrics=metrics, shape=rows[0]['shape']))
            for iteration in range(cfg['iterations']):
                flat.append(dict(phase=phase['id'], workload=work['id'], world=phase['world'],
                                 source_tokens_per_rank=source, global_tokens=total, iteration=iteration,
                                 **{k: v['per_iteration_rank_max_ms'][iteration] for k,v in metrics.items()}))
    save(run/'comparison.json', dict(passed=True, points=points, native_sha256=NATIVE_SHA,
         aggregation='Aligned maximum across all ranks per iteration; then descriptive mean/median over three samples.',
         caveats=['Segments have independently selected critical ranks and must not be added.',
                  'Matches source and received assignments per rank, not fixed global token count.',
                  'Balanced synthetic routes; includes native packing/control and CPU submission gaps; excludes model, GEMM and HiCache.',
                  'This does not isolate physical wire latency or prove a full-model PP speedup.']))
    with (run/'full_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    print(json.dumps(dict(passed=True, points=len(points), native_sha256=NATIVE_SHA)))

if __name__ == '__main__':
    main()
