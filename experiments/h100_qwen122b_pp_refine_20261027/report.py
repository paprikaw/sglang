"""Publish audited PP refinement results and traceable, compact archives."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics as st

ROOT = Path(__file__).resolve().parent
ID = 'qwen122b_h100_pp_refinement_20261027'
DEST = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments') / ID


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def index(path, body):
    start, end = '<!-- begin ' + ID + ' -->', '<!-- end ' + ID + ' -->'
    old = path.read_text()
    block = start + '\n' + body + '\n' + end
    if start in old:
        text = old[:old.index(start)] + block + old[old.index(end) + len(end):]
    else:
        text = old.rstrip() + '\n\n' + block + '\n'
    if text != old:
        path.write_text(text)


def main():
    out = ROOT / 'report'
    out.mkdir(exist_ok=True)
    feedback = read(ROOT / 'analysis/pp_feedback_r3/results.json')
    ep = read(ROOT / 'analysis/ep_control_r1/results.json')
    balance_path = ROOT / 'analysis/pp_balance_r2/results.json'
    balance = read(balance_path) if balance_path.exists() else None
    old_windows = read(ROOT / 'analysis/old_window_audit.json')
    accepted = [feedback, ep] + ([balance] if balance else [])
    rows = []
    for run in [feedback] + ([balance] if balance else []):
        name = Path(run['run']).name
        for row in run['metrics']:
            rows.append(dict(run=name, configuration=('PP2 25+23' if 'balance' in name else 'PP2 24+24') +
                            (' + tail-local feedback' if row['policy'] == 10 else ' + policy9'), **row))
    for row in ep['metrics']:
        rows.append(dict(run=Path(ep['run']).name, configuration='EP2 native optimized',
                         workload=row['workload'], policy=0, throughput_tok_s=row['output_tok_s'],
                         throughput_stdev=row['stdev_tok_s'], tpot_ms=row['tpot_ms'],
                         ttft_ms=row['ttft_ms'], request_e2e_ms=row['e2e_ms'], repeats=row['repeats']))
    comparisons = []
    for case in {row['workload'] for row in rows}:
        base = next(row for row in rows if row['run'] == 'pp_feedback_r3' and row['policy'] == 9 and row['workload'] == case)
        ep_row = next(row for row in rows if row['run'] == 'ep_control_r1' and row['workload'] == case)
        for row in [r for r in rows if r['workload'] == case]:
            row['throughput_vs_pp9_pct'] = (row['throughput_tok_s'] / base['throughput_tok_s'] - 1) * 100
            row['throughput_vs_ep_pct'] = (row['throughput_tok_s'] / ep_row['throughput_tok_s'] - 1) * 100
        if balance:
            previous = ROOT / 'runs/pp_feedback_r3/pp2/workloads' / case / 'native_output_reference.json'
            current = ROOT / 'runs/pp_balance_r2/pp2/workloads' / case / 'native_output_reference.json'
            assert read(previous)['output_ids'] == read(current)['output_ids'], 'Repartition changed full output IDs'
            comparisons.append(dict(workload=case, exact_full_output_ids_across_partitions=True))
    for row in feedback['profiles']:
        assert row['selected_sequence_sum_keys'] == ep['components'][0]['selected_sequence_sum_keys']
    latest_pp = [r for r in feedback['profiles'] if r['policy'] == 10]
    current_budget = dict(
        pp_expert_gemm_us_per_layer=st.mean(r['components_us_per_card_layer']['expert.routed_gemm'] for r in latest_pp),
        ep_expert_gemm_us_per_layer=st.mean(r['kernel_us_per_layer']['expert.routed_gemm'] for r in ep['components']),
        ep_dispatch_combine_us_per_layer=st.mean(sum(r['kernel_us_per_layer'].get(k, 0) for k in ('communication.dispatch', 'communication.combine')) for r in ep['components']),
        ep_comm_intersection_all_compute_us_per_layer=st.mean(r['comm_intersection_all_compute_us_per_layer'] for r in ep['components']),
        pp_model_graphs_ms=st.mean(r['cycle_mean']['two_local_graphs_ms'] for r in latest_pp),
        ep_model_graph_ms=st.mean(r['cycle_mean']['model_graph_ms'] for r in ep['components']))
    result = dict(id=ID, metrics=rows, paired=feedback['paired'] + (balance['paired'] if balance else []),
                  repartition_checks=comparisons, accepted_runs=[r['run'] for r in accepted],
                  profiles=[p for r in [feedback] + ([balance] if balance else []) for p in r['profiles']],
                  ep_components=ep['components'], historical_results_used_as_current=False,
                  old_window_audit=old_windows, matching_position_budget=current_budget)
    save(out / 'results.json', result)
    fields = ['configuration', 'workload', 'throughput_tok_s', 'throughput_stdev', 'tpot_ms',
              'request_e2e_ms', 'throughput_vs_pp9_pct', 'throughput_vs_ep_pct', 'repeats', 'run']
    with (out / 'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 122B H100 PP2 refinement', '',
        '2×H100 80GB, node178 GPU0/1 NVLink; full Qwen3.5-122B-A10B FP8. B32 repeated Decode: GPUcached4096 + new1 → output128. EP AttentionDP2/TP1; PP two fixed B16 slots. Both retain native FP8/DeepGEMM and the common Decode prefix-state lock optimization.', '',
        '| Configuration | Decode tok/s ↑ | Stdev | TPOT ms ↓ | vs original PP9 | vs fresh EP2 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['configuration']} | {row['throughput_tok_s']:.2f} | {row['throughput_stdev']:.2f} | {row['tpot_ms']:.3f} | {row['throughput_vs_pp9_pct']:+.2f}% | {row['throughput_vs_ep_pct']:+.2f}% |")
    lines += ['', 'Tail-local retirement gives a reproducible small improvement (0.44% on 24+24, 0.41% on 25+23). Repartitioning gives no material additional gain. Keep the 24+24 policy10 candidate; this workload still favors the native optimized EP2.', '',
        f"At matching generated positions, PP Expert GEMM is {current_budget['pp_expert_gemm_us_per_layer']:.2f} µs/layer. EP Expert GEMM is {current_budget['ep_expert_gemm_us_per_layer']:.2f} µs plus {current_budget['ep_dispatch_combine_us_per_layer']:.2f} µs Dispatch/Combine; {current_budget['ep_comm_intersection_all_compute_us_per_layer']:.2f} µs of communication overlaps other computation. PP's apparent component advantage is therefore very small before considering overlap. Profiled model graph time is PP {current_budget['pp_model_graphs_ms']:.3f} ms versus EP {current_budget['ep_model_graph_ms']:.3f} ms per matched local work amount."]
    lines += ['', '## Correction to the previous component comparison', '',
        'The previous EP slice used sequence-length sums 65888–66128, while PP used 65728–65840. The selected generated positions did not overlap. Thus the old 147 µs PP versus roughly 170 µs EP GEMM+communication comparison mixes generation windows.', '',
        '| Identical position window (original traces) | EP GEMM µs/layer | EP Dispatch+Combine µs/layer | PP GEMM µs/layer | EP graph ms/generation | PP two local graphs ms/generation |',
        '|---|---:|---:|---:|---:|---:|']
    for window in ('earlier_pp_window', 'later_common_window'):
        values = {}
        for name in ('ep2_native_r5', 'pp2_lookahead_r6'):
            selected = [r for r in old_windows['aligned'] if r['window'] == window and r['run'] == name]
            values[name] = dict(gemm=st.mean(r['kernel_us_per_layer']['expert.routed_gemm'] for r in selected),
                comm=st.mean(sum(r['kernel_us_per_layer'].get(k, 0) for k in ('communication.dispatch', 'communication.combine')) for r in selected),
                graph=st.mean(r['model_graphs_ms_per_generation'] for r in selected))
        ev, pv = values['ep2_native_r5'], values['pp2_lookahead_r6']
        lines.append(f"| {window} | {ev['gemm']:.2f} | {ev['comm']:.2f} | {pv['gemm']:.2f} | {ev['graph']:.3f} | {pv['graph']:.3f} |")
    lines += ['', 'The aligned original profiles show PP slightly slower inside the model graphs. Communication kernel duration also overlaps other EP compute, so GEMM+communication sums do not equal an exposed critical-path cost.', '',
        '## PP-specific changes', '',
        '1. **Tail-local retirement (policy10):** keep policy9 early activation send/lookahead, but let the tail consume its own sampled-token tensor. The head still receives the real tail output; it stops echoing the same tensor back. Producer events, two-slot ownership and send-buffer lifetimes remain enforced. Expert/Attention kernels and captured model graphs are unchanged.',
        '2. **Measured-time stage partition:** test 25+23 instead of 24+24 layers, moving GDN layer24 to the head while retaining six full-Attention layers per stage. The tail also owns the output head. P2P channels are initialized before Prefill temporary allocations; KV capacity is unchanged.', '',
        '## Paired scheduling evidence', '']
    for run in [feedback] + ([balance] if balance else []):
        for pair in run['paired']:
            lines.append(f"- {Path(run['run']).name}: policy10 vs policy9 {pair['p10_vs_p9_mean_throughput_pct']:+.3f}% throughput, {pair['positive_pairs']}/{pair['pairs']} positive pairs; pair changes: " + ', '.join(f'{v:+.3f}%' for v in pair['per_pair_throughput_change_pct']) + '.')
    lines += ['', 'Each scheduling comparison uses one model process and unchanged CUDA Graphs, three full warmups per policy, five counterbalanced profiler-off pairs, exact 128-token output agreement against native policy0, exact GPU-prefix cache counts and drained ownership receipts. Stage-partition and EP comparisons are fresh sequential deployments; they are not within-process pairs.', '',
        '## Profiles at matching generated positions', '',
        '| Run / policy | Rank | Layers | Two local graphs ms | Between graphs ms | Cycle ms |',
        '|---|---:|---:|---:|---:|---:|']
    for run in [feedback] + ([balance] if balance else []):
        for row in run['profiles']:
            value = row['cycle_mean']
            lines.append(f"| {Path(run['run']).name} / {row['policy']} | {row['rank']} | {row['layers_per_graph']} | {value['two_local_graphs_ms']:.3f} | {value['between_graphs_ms']:.3f} | {value['cycle_ms']:.3f} |")
    lines += ['', 'EP profiles cover one 48-layer graph per generation; PP profiles cover two local stage graphs (both B16 slots). Selected sequence-length sums are matched. Profile-on cycles explain mechanisms and are kept separate from full profiler-off TPOT.', '',
              '| Fresh EP2 rank | Graph ms | Gap ms | Cycle ms | Dispatch + Combine µs/layer | Comm ∩ all compute µs/layer |',
              '|---|---:|---:|---:|---:|---:|']
    for row in ep['components']:
        value, costs = row['cycle_mean'], row['kernel_us_per_layer']
        lines.append(f"| {row['rank']} | {value['model_graph_ms']:.3f} | {value['between_graphs_ms']:.3f} | {value['cycle_ms']:.3f} | {costs.get('communication.dispatch', 0) + costs.get('communication.combine', 0):.2f} | {row['comm_intersection_all_compute_us_per_layer']:.2f} |")
    lines += ['', '## Reproduction and attempts', '',
              'All accepted runs passed frozen-input hashes, native full-model FP8/DeepGEMM qualification, canonical CPU weight-cache loader receipts, exact workload checks, owned GPU cleanup and Guard release. Source and result hashes are preserved with each run. No CPU KV restore was used.', '']
    attempts = []
    for run in sorted((ROOT / 'runs').iterdir()):
        progress = read(run / 'task/progress.json')
        entry = dict(run=str(run), stage=progress['stage'], accepted=str(run) in result['accepted_runs'])
        if (run / 'diagnosis.json').exists():
            entry['diagnosis'] = read(run / 'diagnosis.json')
            if run.name == 'pp_feedback_r1':
                entry['stage'] = 'admission_failed_before_GPU_claim'
        attempts.append(entry)
        lines.append(f"- `{run.name}`: {entry['stage']}; " + ('included' if entry['accepted'] else 'excluded') + '.')
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    labels = [r['configuration'].replace(' + ', '\n') for r in rows]
    colors = ['#3979a8' if r['run'].startswith('pp') else '#139b8e' for r in rows]
    for axis, key, ylabel in [(axes[0], 'throughput_tok_s', 'Output tokens / s ↑'), (axes[1], 'tpot_ms', 'TPOT (ms) ↓')]:
        bars = axis.bar(range(len(rows)), [r[key] for r in rows], color=colors)
        axis.set_xticks(range(len(rows)), labels, fontsize=8)
        axis.set_ylabel(ylabel)
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
        for bar, row in zip(bars, rows):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{row[key]:.2f}", ha='center', va='bottom', fontsize=9)
        axis.set_ylim(0, max(r[key] for r in rows) * 1.14)
    fig.suptitle('Qwen3.5-122B FP8 · 2×H100 NVLink · B32 repeated Decode\nGPUcached4096 + new1 → output128 · 5 profiler-off cohorts / point')
    fig.savefig(out / 'decode_refinement.png', dpi=180)
    plt.close(fig)
    DEST.mkdir(parents=True, exist_ok=True)
    for path in out.iterdir():
        if path.is_file():
            copy(path, DEST / 'report' / path.name)
    for path in ROOT.glob('*.py'):
        copy(path, DEST / 'scripts' / path.name)
    for name in ('source_cpu_qualification.json', 'balance_warm_cpu_qualification.json', 'kv_checkpoint_discovery.json'):
        copy(ROOT / name, DEST / 'qualification' / name)
    for folder in ('source', 'balance_source', 'balance_warm_source'):
        for path in (ROOT / folder).glob('*.py'):
            copy(path, DEST / 'source' / folder / path.name)
    copy(ROOT / 'analysis/old_window_audit.json', DEST / 'analysis/old_window_audit.json')
    raw = []
    patterns = ['config.json', 'dashboard_spec.json', 'launch_manifest.json', 'diagnosis.json',
                '*qualification.json', 'task/*.json', 'dashboard_guard/checks/*.json', 'code/*.py',
                '*/qualification/*.json', '*/cleanup.json', '*/completed.json', '*/workloads/*/*.json',
                '*/workloads/*/overlap_qualification/*.json', '*/workloads/*/p*/metrics.*.json',
                '*/workloads/*/p*/traces/decode/manifest.json', '*/workloads/*/traces/decode/manifest.json']
    for run in sorted((ROOT / 'runs').iterdir()):
        paths = {path for pattern in patterns for path in run.glob(pattern)}
        for path in sorted(paths):
            assert path.name != 'admin.key'
            copy(path, DEST / 'runs' / run.name / path.relative_to(run))
            raw.append(dict(path=str(path), sha256=sha(path), kind='run_evidence'))
    for value in accepted:
        name = Path(value['run']).name
        for path in (ROOT / 'analysis' / name).glob('*.json'):
            copy(path, DEST / 'analysis' / name / path.name)
        raw += value['sources']
    raw += [row['trace'] for row in old_windows['aligned']]
    for folder in ('h100_qwen122b_latest_decode_20261027', 'h100_pp_lookahead_workloads_20261027',
                   'h100_pp_expert_fragmentation_20261026', 'pp_ep_steady_decode_20261026'):
        path = ROOT.parent / folder / 'analyze.py'
        copy(path, DEST / 'analysis_dependencies' / folder / 'analyze.py')
        raw.append(dict(path=str(path), sha256=sha(path), kind='analysis_dependency'))
    save(DEST / 'raw_sources.json', raw)
    cfg = read(ROOT / 'runs/pp_feedback_r3/config.json')
    save(DEST / 'manifest.json', dict(id=ID, status='complete', created_at=datetime.now(timezone.utc).isoformat(),
        model=dict(path=cfg['model_path'], revision=cfg['model_revision'], precision=cfg['precision']),
        engine=dict(name='sglang', worktree=cfg['engine_root'], commit=cfg['engine_commit'], tag=cfg['engine_tag'], unchanged=True,
                    branch='codex/qwen35b-decode-ep-overlap-20261027', old_implementation=False),
        private_adapter_sources=[dict(run=value['run'], sources=read(Path(value['run']) / 'dashboard_spec.json')['sources']) for value in accepted],
        deployment=dict(node=cfg['node'], job_id=cfg['job_id'], gpus=[0, 1], link='NVLink', retired=True,
                        launch_manifests=[value['run'] + '/launch_manifest.json' for value in accepted], gateway_route=None),
        workload=dict(requests=32, family='repeated', prefix=4096, new=1, output=128, cpu_kv_restore=False),
        accepted_runs=result['accepted_runs'], attempts=attempts, metrics=rows,
        report='report/REPORT.md', raw_sources='raw_sources.json'))
    (DEST / 'summary.md').write_text('# Qwen122B H100 PP refinement\n\n'
        'Tail-local PP retirement improves B32 repeated Decode by 0.44%; 25+23 stage partition gives no material further gain. '
        'Fresh native optimized EP2 remains faster. Earlier component samples used different generated positions; the report corrects that comparison.\n\n'
        '[Full report](report/REPORT.md)\n')
    checksums = [f'{sha(path)}  {path.relative_to(DEST)}' for path in sorted(DEST.rglob('*')) if path.is_file() and path.name != 'SHA256SUMS']
    (DEST / 'SHA256SUMS').write_text('\n'.join(checksums) + '\n')
    for line in checksums:
        digest, name = line.split('  ', 1)
        assert sha(DEST / name) == digest
    description = ('Qwen3.5-122B-A10B native FP8, node178 two H100s over NVLink, B32 repeated GPUcached4096+new1→output128. '
                   'PP9 versus PP10 tail-local retirement: five counterbalanced profiler-off pairs, unchanged Graph compute, complete native output agreement. '
                   'Measured-time 25+23 PP partition trial and fresh native optimized EP2 control; matching-position independent profiles. ')
    description += '; '.join(f"{r['configuration']}: {r['throughput_tok_s']:.2f} tok/s, TPOT {r['tpot_ms']:.3f} ms" for r in rows) + '.'
    body = f'### {ID}\n\n{description}\n\n[Report]({DEST}/report/REPORT.md) · [Figure]({DEST}/report/decode_refinement.png)\n\nSnapshot: {DEST}'
    index(Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md'), body)
    index(ROOT.parent / 'experiment_summary/README.md', body)
    print(json.dumps(dict(archive=str(DEST), files=len(checksums), metrics=rows), indent=2))


if __name__ == '__main__':
    main()
