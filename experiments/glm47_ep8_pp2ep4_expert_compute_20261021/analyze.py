"""Summarize measured local expert invocations, not EP/PP system throughput."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def main(run):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'guard/skills/llm-experiment-guard/scripts'))
    from artifact_visibility import wait_for_stable_json
    result, visibility = wait_for_stable_json(run / 'profile/result.json', timeout_s=10)
    if result.get('passed') is not True: raise RuntimeError('Invalid GPU result')
    out = run / 'analysis'; out.mkdir(exist_ok=True)
    (out / 'result_visibility.json').write_text(json.dumps(visibility, indent=2) + '\n')
    rows = []
    for mode in ('eager', 'cuda_graph'):
        for b in sorted({r['global_logical_tokens'] for r in result['rows']}):
            points = [r for r in result['rows'] if r['mode'] == mode and r['global_logical_tokens'] == b]
            ep = next(r for r in points if r['layout'] == 'ep8')
            pp = sorted((r for r in points if r['layout'] == 'pp2_ep4'), key=lambda r: r['pp_slot'])
            if len(pp) != 2 or any(r['gemm_flops'] != ep['gemm_flops'] for r in pp):
                raise RuntimeError('Comparison is not equal-work across both PP slots')
            mean_pp = sum(r['timing']['median_ms'] for r in pp) / 2
            def block_medians(point):
                values = point['cuda_event_ms']
                return [statistics.median(values[i:i+10]) for i in range(0, len(values), 10)]
            ep_blocks = block_medians(ep)
            pp_blocks = list(zip(*(block_medians(p) for p in pp)))
            paired_ratios = [statistics.mean(p) / e for p, e in zip(pp_blocks, ep_blocks)]
            rows.append({'mode': mode, 'global_logical_tokens': b,
                'local_assignments_each': ep['local_assignments'],
                'gemm_flops_each': ep['gemm_flops'],
                'ep8_active_experts': ep['active_experts'],
                'pp2ep4_active_experts': pp[0]['active_experts'],
                'ep8_median_ms': ep['timing']['median_ms'],
                'pp2ep4_slot0_median_ms': pp[0]['timing']['median_ms'],
                'pp2ep4_slot1_median_ms': pp[1]['timing']['median_ms'],
                'pp2ep4_mean_slot_median_ms': mean_pp,
                'pp_over_ep_latency_ratio': mean_pp / ep['timing']['median_ms'],
                'paired_block_ratio_min': min(paired_ratios),
                'paired_block_ratio_max': max(paired_ratios),
                'max_sample_cv': max(r['timing']['cv'] for r in points)})
    with (out / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    full = []
    for r in result['rows']:
        full.append({k: v for k, v in r.items() if k not in ('timing', 'counts', 'cuda_event_ms')} |
                    r['timing'] | {'counts': json.dumps(r['counts'])})
    with (out / 'full_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(full[0])); writer.writeheader(); writer.writerows(full)
    geometry = []
    for point in result['rows']:
        # Source: HummingRunnerCore.estimate_local_valid_shape_m returns
        # expected_m * num_local_experts, not expected_m alone. These are
        # static block-coverage estimates, not measured CTA or memory counters.
        experts = point['local_experts']
        valid_m = point['expected_m'] * experts
        for projection in ('w13', 'w2'):
            configurations = result['tuning'][str(experts)][projection + '_tuning_config']
            selected = [c for lo, hi, c in configurations if lo <= valid_m < hi]
            if len(selected) != 1: raise RuntimeError('Ambiguous static tuning range')
            config = selected[0]
            tile_m = config['block_shape'][0]
            tiles = sum((n + tile_m - 1) // tile_m for n in point['counts'])
            geometry.append({
                'mode': point['mode'], 'B': point['global_logical_tokens'],
                'layout': point['layout'], 'pp_slot': point['pp_slot'],
                'projection': projection, 'valid_shape_m': valid_m,
                'local_experts': experts, 'active_experts': point['active_experts'],
                'local_assignments': point['local_assignments'],
                'block_shape': json.dumps(config['block_shape']),
                'static_expert_m_tiles': tiles, 'static_padded_rows': tiles * tile_m,
            })
    with (out / 'kernel_geometry.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(geometry[0])); writer.writeheader(); writer.writerows(geometry)
    summary = {'scope': result['scope'], 'backend': result['backend'], 'gpu': result['gpu'],
               'rows': rows, 'numerical': result['numerical'],
               'raw_result_sha256': hashlib.sha256((run / 'profile/result.json').read_bytes()).hexdigest()}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    text = ['# GLM4.7 expert-only: EP8 versus PP2 x EP4', '',
            'One real layer, one A100 GPU, equal local routed assignments and ideal GEMM FLOPs.',
            'No Attention, router, shared expert, dispatch/combine, or pipeline scheduling.', '',
            '| Mode | Global logical tokens | EP8 ms | PP2xEP4 mean slot ms | PP/EP |',
            '|---|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['mode']} | {r['global_logical_tokens']} | {r['ep8_median_ms']:.4f} | {r['pp2ep4_mean_slot_median_ms']:.4f} | {r['pp_over_ep_latency_ratio']:.3f} |")
    text += ['', 'B denotes newly processed logical token rows, not request count.',
        'D=4 physical TP source-copy convention of the saved probe: both configurations receive 4*B local assignments. This convention is not a claim of four times as many unique user tokens in a production server.',
        'PP means two B/2 microbatches; both slots are measured, and the table averages their medians.',
        'The equal-FLOPs normalization is L*tEP vs (L/2)*(tPPslot0+tPPslot1).',
        'A pure compute emulation does not measure eight-GPU end-to-end throughput or pipeline bubbles.',
        'Weight sizes in provenance describe stored checkpoint tensors, not measured HBM traffic.',
        '', '## Interpretation', '',
        'At B=32, Graph PP/EP is 1.426. Both layouts perform 128 local assignments, but EP8 touches 20 experts (4-8 rows each) and PP touches 32 (4 rows each). The recorded tuning map selects M=16 blocks: static M-tile coverage is 320 versus 512 rows. This is a concrete fragmentation/padding mechanism despite equal useful GEMM FLOPs; it does not separately quantify HBM traffic or SM utilization.',
        'At B=128, all 20/40 experts are hit; Graph PP/EP is 1.097. Counts, selected kernel shapes and expert granularity differ, so equal FLOPs do not require equal execution time.',
        'At B=2048 and 4096, the Graph difference is below 1%. This run supports approximate compute equivalence at these larger shapes, not a stable sub-percent winner. B=512 has block-to-block drift, so its small gap should not be overstated.',
        'Eager and Graph use the same compute backend but different launch protocols; their absolute values are not interchangeable. These measurements cannot attribute historical full-model differences entirely to expert compute.',
        'Correctness: four sampled common/boundary experts pass the independent AWQ dequantization reference, and every measured configuration has finite active outputs. This is not a quality evaluation or an exhaustive numerical proof for every expert.',
        'Timing: three alternating blocks of ten CUDA-event samples per configuration after warmup. This is one GPU session, not three independent cluster runs. Raw samples and both PP slots are retained.']
    (out / 'summary.md').write_text('\n'.join(text) + '\n')
    print('\n'.join(text))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    for ax, mode in zip(axes, ('eager', 'cuda_graph')):
        points = [r for r in rows if r['mode'] == mode]
        x = range(len(points))
        ax.plot(x, [r['ep8_median_ms'] for r in points], 'o-', label='EP8:20 experts/GPU')
        ax.plot(x, [r['pp2ep4_mean_slot_median_ms'] for r in points], 's-', label='PP2 x EP4:40 experts/GPU')
        ax.set_xticks(list(x), [str(r['global_logical_tokens']) for r in points])
        ax.set_xlabel('Global logical new-token rows B')
        ax.set_ylabel('Local expert compute latency (ms)')
        ax.set_yscale('log')
        ax.set_title(mode.replace('_', ' ').title()); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('Real GLM4.7-AWQ layer3 on one A100; equal local FLOPs')
    fig.savefig(out / 'expert_compute.png', dpi=180)
    fig.savefig(out / 'expert_compute.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
