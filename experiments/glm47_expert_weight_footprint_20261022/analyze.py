"""Archive measured expert footprint curves; no GPU access."""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import statistics
import tarfile

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/r1'
ID = 'glm47_expert_weight_footprint_20261022_r1'
SNAPSHOTS = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')
OUT = SNAPSHOTS / ID
HISTORY = SNAPSHOTS / 'glm47_h100_ae_plateau_tp4_ep4_20261009'


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str) + '\n')


def table(name, rows):
    with (OUT / name).open('w', newline='') as f:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Immutable snapshot already exists')
    if (OUT / 'raw_result.json').exists():
        assert (OUT / 'raw_result.json').read_bytes() == (RUN / 'profile/result.json').read_bytes()
    result = json.loads((RUN / 'profile/result.json').read_text())
    spec = json.loads((RUN / 'dashboard_spec.json').read_text())
    cleanup = json.loads((RUN / 'profile/task/cleanup.json').read_text())
    posts = list((RUN / 'dashboard_guard/checks').glob('*-post.json'))
    assert result['passed'] and len(result['rows']) == 36 and cleanup['passed']
    assert len(posts) == 1
    post = json.loads(posts[0].read_text())
    assert post['passed']
    assert len(result['numerical']) == 36 and all(v['passed'] for v in result['numerical'])
    rows, primary, plateau = [], [], []
    for r in result['rows']:
        assert len(r['cuda_event_ms']) == 30 and len(r['block_median_ms']) == 3
        assert all(math.isfinite(x) and x > 0 for x in r['cuda_event_ms'])
        assert math.isclose(statistics.median(r['cuda_event_ms']), r['timing']['median_ms'])
        assert math.isclose(statistics.mean(r['cuda_event_ms']), r['timing']['mean_ms'])
        counts = r['counts']
        assert sum(counts) == r['total_assignments']
        active = [c for c in counts if c]
        assert len(active) == r['active_experts'] and max(active) - min(active) <= 1
        assert len(counts) == 40 and max(counts) <= 4096
        median = r['timing']['median_ms']
        row = dict(total_local_assignments=r['total_assignments'],
                   active_experts=r['active_experts'], resident_experts=40,
                   active_checkpoint_MiB=r['active_checkpoint_bytes'] / 2**20,
                   rows_per_active_expert=r['mean_rows_per_active_expert'],
                   gemm_flops=r['gemm_flops'], mode=r['mode'], **r['timing'],
                   block_min_median_ms=min(r['block_median_ms']),
                   block_max_median_ms=max(r['block_median_ms']),
                   block_relative_range=(max(r['block_median_ms']) - min(r['block_median_ms'])) / median,
                   effective_tflops=r['effective_tflops'])
        rows.append(row)
        if r['mode'] == 'cuda_graph':
            primary.append(row)
    for total in (128, 512, 8192):
        points = [r for r in primary if r['total_local_assignments'] == total]
        best = min(r['median_ms'] for r in points)
        baseline = points[0]['median_ms']
        within = [r for r in points if r['median_ms'] <= 1.10 * best]
        plateau.append(dict(total_local_assignments=total, threshold='<= 110% of sampled minimum',
                            sampled_best_ms=best,
                            admitted_active_experts=[r['active_experts'] for r in within],
                            largest_sampled_within_10pct=max(r['active_experts'] for r in within),
                            e40_over_e2_pct=(points[-1]['median_ms'] / baseline - 1) * 100))
        for r in points:
            r['over_e2_pct'] = (r['median_ms'] / baseline - 1) * 100
            r['over_sampled_best_pct'] = (r['median_ms'] / best - 1) * 100
    history = []
    for r in csv.DictReader((HISTORY / 'full_metrics.csv').open()):
        if (r['stage'] == 'expert' and r['timing_mode'] == 'cuda_graph'
                and r['point_id'] in ('t64_e2', 't64_e10', 't64_e40',
                                     't1024_e2', 't1024_e10', 't1024_e40')):
            history.append(dict(point_id=r['point_id'],
                                historical_T_unique=int(r['T_unique']),
                                total_local_assignments=int(r['local_routed_assignments']),
                                active_experts=int(r['active_experts_local']),
                                median_ms=float(r['median_ms']), samples=int(r['n']),
                                scope='H100 four-rank aligned max; D4 source convention; isolated expert core',
                                source=str(HISTORY / 'full_metrics.csv')))
    assert len(history) == 6
    OUT.mkdir(exist_ok=True)
    # Do not copy model tensors, cache bytes, credentials, or private lease receipts.
    for src, dst in [(RUN / 'profile/result.json', 'raw_result.json'),
                     (RUN / 'selection.json', 'weight_selection.json'),
                     (RUN / 'dashboard_spec.json', 'frozen_spec.json')]:
        shutil.copyfile(src, OUT / dst)
    table('full_metrics.csv', rows)
    table('graph_metrics.csv', primary)
    table('historical_h100.csv', history)
    save('plateau.json', plateau)
    save('validation.json', dict(passed=True, numerical=result['numerical'],
                                 cleanup=cleanup, guard_postcheck=post))
    with tarfile.open(OUT / 'frozen_sources.tar.gz', 'w:gz') as archive:
        for p in sorted(ROOT.glob('*.py')):
            if p.name != 'analyze.py':
                archive.add(p, arcname=p.name)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, total in zip(axes, (128, 512, 8192)):
        ps = [r for r in primary if r['total_local_assignments'] == total]
        xs = [r['active_experts'] for r in ps]
        ys = [r['median_ms'] for r in ps]
        lo = [max(0, r['median_ms'] - r['block_min_median_ms']) for r in ps]
        hi = [max(0, r['block_max_median_ms'] - r['median_ms']) for r in ps]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker='o', capsize=3, color='#166e95')
        best = min(ys)
        ax.axhspan(best, 1.1 * best, color='#66b58a', alpha=.18,
                   label='Within 10% of sampled minimum')
        ax.set_title(f'Fixed {total:,} local token-expert rows')
        ax.set_xlabel('Active experts (40 resident)')
        ax.set_ylabel('Expert-core latency (ms)')
        ax.set_xticks(xs)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.18)
        ax.legend(fontsize=7, loc='upper left')
    fig.suptitle('A100 / GLM-4.7 AWQ / CUDA Graph / balanced routing\n'
                 'Weight load and JIT excluded; error bars = range of three block medians', fontsize=11)
    fig.savefig(OUT / 'expert_footprint.png', dpi=180)
    fig.savefig(OUT / 'expert_footprint.pdf')
    plt.close(fig)
    lines = ['# GLM expert weight-footprint sweep', '',
             'Completed one booked A100 node (065), actual GPU0. Real GLM-4.7-AWQ layer3 experts0..39.',
             'Expert core is W13 GEMM + masked SiLU/multiply + W2 GEMM. No router, shared expert, Attention or communication.',
             '',
             'Fixed actual local token-expert rows T, balanced across E active experts. '
             'Resident expert count40, fixed capacity4096/expert, and tuning estimate ceil(T/40) '
             'stay unchanged within each T sweep. Increasing E changes the active weight working set, '
             'not loaded/resident bytes. Counts of unused experts are zero. No D4 source convention is used.',
             '',
             '| Local rows | E2 | E4 | E8 | E10 | E20 | E40 |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for total in (128, 512, 8192):
        ps = [r for r in primary if r['total_local_assignments'] == total]
        lines.append('| ' + str(total) + ' | ' + ' | '.join(f"{r['median_ms']:.4f}" for r in ps) + ' |')
    lines += ['', 'Times are CUDA Graph medians in ms; eager controls and all statistics are in full_metrics.csv.', '',
              '## Observed plateau (an explicit tolerance, not a hardware limit)', '']
    for r in plateau:
        lines.append(f"- T={r['total_local_assignments']}: active E={r['admitted_active_experts']} "
                     'are within10% of the sampled minimum. The interval between the final admitted '
                     'point and next sample brackets a knee only at this resolution.')
    lines += ['', '## Historical H100 evidence', '',
              'historical_h100.csv recovers the six accepted fixed-work expert-core rows. '
              'That cohort uses four-rank aligned-max latency,24 samples/point, a D4 source convention, '
              'and a different engine revision. It is not a direct A100/H100 speedup comparison.', '',
              '## Interpretation and limits', '',
              'At fixed T, ideal GEMM FLOPs6*H*I*T do not change. Increasing E increases the distinct '
              'weight working set and reduces rows per expert, changing weight reuse, padding and tile scheduling. '
              'Timing establishes the net cost; HBM/L2 traffic and occupancy counters were not measured. '
              'Checkpoint bytes are not measured DRAM traffic. No universal expert-count limit, '
              'pure bandwidth attribution, full-model latency, PP/EP win, or SLO is inferred.', '',
              'Both modes use five setup warmups and three immediate warmups before each block, '
              'three alternating/rotated blocks of ten CUDA-event samples. These are within-process '
              'blocks, not three independent launches. Block medians drift by up to about12% '
              'for some points, so sub-10% rankings should not be treated as stable wins. '
              'P99 is descriptive. Warm replay favors reuse '
              'and does not simulate interleaved layers evicting the weights.', '',
              'All36 output/reference checks passed. Guard postcheck and owned GPU cleanup passed; lease released.', '']
    (OUT / 'summary.md').write_text('\n'.join(lines))
    save('manifest.json', dict(
        id=ID, title='Fixed-token expert active-weight overhead on A100, with recovered H100 controls',
        created_at=datetime.now(timezone.utc).isoformat(), status='complete_verified',
        implementation=dict(engine='SGLang', sources=spec['sources'],
                            frozen_spec_sha256=hashlib.sha256((RUN / 'dashboard_spec.json').read_bytes()).hexdigest(),
                            backend=result['backend'], old_implementation=False),
        model=dict(path=spec['model_cache']['model_path'], layer=3, dtype='W4A16 BF16 activations',
                   selected_experts=list(range(40))),
        testbed=dict(node='spartan-gpgpu065', job_id='33463294', actual_gpu=0,
                     registered_whole_node_gpus=4, gpu=result['gpu']),
        workload=dict(total_local_assignments=[128,512,8192], active_experts=[2,4,8,10,20,40],
                      balanced=True, source_duplication='not used; actual received rows fixed',
                      request_context='not applicable; expert core only',
                      active_payload_MiB_per_expert=12_257_280/2**20),
        engine_parameters=dict(fixed_resident_experts=40, capacity_per_expert=4096,
                               timing_modes=['eager','cuda_graph'], warmups=5,
                               immediate_warmups_per_block=3, blocks=3, samples_per_block=10,
                               attention=False, communication=False, cache_protocol='warm replay'),
        artifacts=dict(run=str(RUN), historical=str(HISTORY), plot='expert_footprint.png'),
        validation=dict(scientific=True, numerical_checks=36, guard_postcheck=True, cleanup=True,
                        lease_id=result['authorization']['lease_id'], hardware_counters=False),
        key_results=plateau,
    ))
    checksums = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in OUT.iterdir() if p.is_file()}
    save('checksums.json', checksums)
    assert all(hashlib.sha256((OUT / n).read_bytes()).hexdigest() == h for n,h in checksums.items())
    print(json.dumps(dict(snapshot=str(OUT), checksums_verified=len(checksums), plateau=plateau), indent=2))


if __name__ == '__main__':
    main()
