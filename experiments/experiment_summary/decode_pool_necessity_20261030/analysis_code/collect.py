"""Rebuild the archived mean tables and decision evidence from explicit runs."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ACCEPTED = ['011_screen_r3', '011_ep2_r4', '171_screen_r1',
            '171_dense_r2', '171_reverse_r3', '171_long32_r4']


def read(path):
    return json.loads(path.read_text())


def csv_rows(path):
    return list(csv.DictReader(path.open())) if path.exists() else []


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    runs = [ROOT / 'runs' / name for name in ACCEPTED]
    assert all(read(r / 'task/progress.json')['stage'] in ('failed', 'complete') for r in runs)
    commands = []
    for run in runs:
        commands.append([sys.executable, str(ROOT / 'analyze.py'), '--run', str(run),
                         '--output', str(out / 'per_run' / run.name)])
    for name, selected in [('primary', runs[:2]), ('confirmation', runs[4:]), ('controls', runs[2:4])]:
        cmd = [sys.executable, str(ROOT / 'analyze.py')]
        for run in selected:
            cmd += ['--run', str(run)]
        commands.append(cmd + ['--output', str(out / name)])
    def invoke(command):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-4000:])
        return json.loads(result.stdout)
    with ThreadPoolExecutor(max_workers=3) as pool:
        summaries = list(pool.map(invoke, commands))
    audit_command = [sys.executable, str(ROOT / 'audit_arms.py')]
    for run in runs:
        audit_command += ['--run', str(run)]
    invoke(audit_command + ['--output', str(out / 'accepted_arm_audit.json')])
    all_cells = [row for name in ACCEPTED for row in csv_rows(out / 'per_run' / name / 'mean_metrics.csv')]
    rounds = [row for name in ACCEPTED for row in csv_rows(out / 'per_run' / name / 'formal_rounds.csv')]
    forwards = [row for name in ACCEPTED for row in csv_rows(out / 'per_run' / name / 'forward_histograms.csv')]
    write_csv(out / 'full_metrics.csv', all_cells)
    write_csv(out / 'formal_rounds.csv', rounds)
    write_csv(out / 'forward_histograms.csv', forwards)
    primary = csv_rows(out / 'primary/mean_metrics.csv')
    by_case = {}
    for cell in primary:
        if cell['status'] == 'measured':
            by_case.setdefault(cell['case'], {})[cell['layout']] = cell
    decision = []
    for case, kinds in sorted(by_case.items()):
        pp = kinds['two_pp2']
        non_pp = {k: v for k, v in kinds.items() if k not in ('two_pp2', 'pp4')}
        for metric in ('mean_request_tpot_ms', 'mean_request_e2e_ms', 'cohort_makespan_ms'):
            best = min(non_pp, key=lambda k: float(non_pp[k][metric]))
            baseline = float(non_pp[best][metric]); value = float(pp[metric])
            decision.append(dict(case=case, profile=pp['profile'], P=pp['P'], N=pp['N'], metric=metric,
                                 pp2_ms=value, best_non_pp=best, best_non_pp_ms=baseline,
                                 pp2_latency_reduction_percent=100 * (1 - value / baseline),
                                 measured_non_pp=','.join(sorted(non_pp))))
    write_csv(out / 'pp_marginal_by_case.csv', decision)
    oracles = csv_rows(out / 'primary/oracle_pairs.csv')
    proxy = {}
    for metric in ('mean_request_tpot_ms', 'mean_request_e2e_ms', 'cohort_makespan_ms'):
        candidates = [r for r in oracles if r['metric'] == metric]
        proxy[metric] = max(candidates, key=lambda r: float(r['oracle_time_reduction_percent'])) if candidates else None
    config_path = Path('/data/gpfs/projects/punim2715/huggingface/Qwen3.6-35B-A3B/config.json')
    config = read(config_path)['text_config']
    expert_bytes = (config['num_hidden_layers'] * config['num_experts'] * config['hidden_size']
                    * config['moe_intermediate_size'] * 3 * 2)
    kv_bytes = (config['layer_types'].count('full_attention') * config['num_key_value_heads']
                * config['head_dim'] * 2 * 2)
    capacity = dict(model_config=dict(path=str(config_path), sha256=sha(config_path)),
                    expert_weight_dtype='bfloat16', kv_dtype='bfloat16',
                    expert_weight_bytes=expert_bytes, expert_weight_gib=expert_bytes / 2**30,
                    full_model_kv_bytes_per_cached_token=kv_bytes,
                    eight_requests_131072_prefix_kv_gib=8 * 131072 * kv_bytes / 2**30,
                    expert_plus_kv_gib=(expert_bytes + 8 * 131072 * kv_bytes) / 2**30,
                    scope='Unsharded full BF16 model per GPU; four replicas, global N32/P131072. '
                          'Experts plus KV alone consume80GiB per GPU before other weights, GDN or workspace. '
                          'Not a claim about N128/P256, alternative precision, offload, or state sharding.')
    (out / 'capacity_derivation.json').write_text(json.dumps(capacity, indent=2) + '\n')
    summary = dict(measured_points=sum(r['status'] == 'measured' for r in all_cells),
                   native_capacity_rows=sum(r['status'] != 'measured' for r in all_cells),
                   formal_cohorts=len(rounds), formal_requests=sum(int(r['N']) for r in rounds),
                   accepted_runs=ACCEPTED, primary_pp2_cases=len(by_case),
                   pp2_wins={m: [r for r in decision if r['metric'] == m and r['pp2_latency_reduction_percent'] > 0]
                             for m in ('mean_request_tpot_ms', 'mean_request_e2e_ms', 'cohort_makespan_ms')},
                   largest_common_feasible_proxy_gains=proxy,
                   caveats=['011_screen_r3 is a partial run: last single-GPU dense startup failed; completed arms have independent original-Guard loader/model/cleanup checks.',
                            '171_screen_r1 native-capacity rows reflect an undersized Mamba profile and are not intrinsic EP capacity limits.',
                            '011_single_dense_r5 retuning failed during untimed native GDN CoW warmup; no performance point accepted.',
                            'Whole-node homogeneous cohorts and a metric oracle do not prove an upper bound for heterogeneous online pools.'])
    (out / 'decision.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('measured_points', 'formal_cohorts', 'formal_requests', 'primary_pp2_cases')}))


if __name__ == '__main__':
    main()
