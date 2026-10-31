"""Read-only historical artifact aggregation; never launches a GPU experiment."""
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
ARCHIVE = WORK.parents[2] / 'experiment_manager' / 'experiments'
HUB = WORK / 'experiment-report-hub/dist/reports/decode-ep8-pp2ep4-cross-model-20261022-v1/metrics.csv'
sys.path.insert(0, '/home/bxb1/.codex/skills/llm-experiment-guard/scripts')
from artifact_visibility import wait_for_stable_json

sources = {}
metrics = []
profiles = []
hash_checks = []


def source(p):
    p = Path(p).resolve()
    if str(p) not in sources:
        data = p.read_bytes()
        sources[str(p)] = {'path': str(p), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    return sources[str(p)]


def rows(p):
    source(p)
    with Path(p).open(newline='') as f:
        return list(csv.DictReader(f))


def js(p):
    value, _ = wait_for_stable_json(Path(p), timeout_s=3)
    source(p)
    return value


def add(p, dataset, arm, workload='decode', **values):
    item = dict(dataset_id=dataset, arm=arm, workload=workload,
                comparison_scope='EP8_vs_PP2xEP4', model='GLM-4.7-AWQ', gpu='A100',
                gpu_count=8, node_count=2, result_status='measured',
                source_path=source(p)['path'], source_sha256=source(p)['sha256'])
    item.update(values)
    metrics.append(item)
    return item


def stats(out, v, name, factor=1):
    for stat in ('mean', 'median', 'p99'):
        if isinstance(v, dict) and v.get(stat) is not None:
            out[f'{name}_{stat}'] = float(v[stat]) * factor


aliases = {
    'glm-h100-cold-b4': 'D01', 'glm-h100-cold-b16': 'D02',
    'glm-a100-graph-b0c15ccb': 'D03', 'glm-a100-graph-9dc34887': 'D04',
    'glm-a100-eager-ce': 'D05', 'glm-a100-eager-rdma': 'D06',
    'hy3-a100-p2304': 'D07', 'hy3-a100-p3712': 'D08',
    'history-a100-arrival': 'H01', 'history-a100-cap8': 'H02',
}
for r in rows(HUB):
    ds = aliases.get(r['cohort_id'], r['cohort_id'])
    snap, filename = r['source_id'].split(':', 1)
    p = ARCHIVE / snap / filename
    if not p.exists():
        p = WORK / 'experiments' / snap / filename
    raw = source(p)
    expected = r.get('source_metric_sha256')
    hash_checks.append({'path': str(p), 'matches_report_snapshot': raw['sha256'] == expected,
                        'recorded_sha256': expected, 'current_sha256': raw['sha256']})
    side = r['arm'] in ('two_ep4', 'pp2ep4_single') or ds not in set(aliases.values())
    item = add(p, ds, r['arm'], comparison_scope='related_2xEP4' if side else 'EP8_vs_PP2xEP4',
               historical_cohort_id=r['cohort_id'], label=r['cohort_label'], model=r['model'], gpu=r['gpu'],
               B=r['request_count'], P=r['prefix_tokens'], Q=r['query_tokens'], O=r['output_tokens'],
               M=r['total_context_tokens'], cuda_graph=r['cuda_graph'], backend=r['backend'],
               result_status=r['result_status'], cache_protocol=r['protocol_class'],
               throughput_tok_s=r['throughput_tokens_per_s'], throughput_unit=r['throughput_unit'],
               notes=r['notes'], cohort='archived_report_row', recorded_source_sha256=expected)
    for stat in ('mean', 'median', 'p99'):
        item[f'ttft_s_{stat}'] = r[f'ttft_s_{stat}']
        item[f'tpot_ms_{stat}'] = r[f'tpot_full_request_ms_{stat}']
        item[f'e2e_s_{stat}'] = r[f'e2e_s_{stat}']
    if ds in ('H01', 'H02'):
        item['B'] = ''  # Arrival counts are not simultaneous batch sizes.
        item['arrival_requests'] = r['request_count']

p = ARCHIVE / 'glm47_2xep4_heavy_20261004_r2/comparison.csv'
for r in rows(p):
    if r['workload'] != 'prefill':
        continue
    arm = {'2xEP4': 'two_ep4', 'EP8': 'ep8', 'PP2xEP4': 'pp2ep4'}[r['configuration']]
    item = add(p, 'H01', arm, 'prefill', comparison_scope='related_2xEP4' if arm == 'two_ep4' else 'EP8_vs_PP2xEP4',
               arrival_requests=65, P=8192, Q=4096,
               result_status='censored' if int(r['cohort_censored']) else 'completed',
               completed_requests=r['cohort_completed'], censored_requests=r['cohort_censored'],
               throughput_tok_s=r['computed_prefill_tokens_per_s'], throughput_unit='completed_new_input_tok/s',
               cache_protocol='historical_arrival_stream',
               notes='65 arrivals at .722 req/s; fixed90s completion window; not a simultaneous B65 batch; O not supplied in comparison table')
    for stat in ('mean', 'median', 'p99'):
        item[f'e2e_s_{stat}'] = r[f'e2e_{stat}_s']


def flat_stats(item, r):
    for stat in ('mean', 'median', 'p99'):
        for field, candidates in {
            'ttft_s': [f'ttft_s_{stat}', f'{stat}_ttft_s'],
            'tpot_ms': [f'tpot_ms_{stat}', f'tpot_full_request_ms_{stat}'],
            'e2e_s': [f'e2e_s_{stat}', f'end_to_end_s_{stat}', f'{stat}_e2e_s'],
        }.items():
            for name in candidates:
                if r.get(name) not in ('', None):
                    item[f'{field}_{stat}'] = r[name]
                    break
        if not item.get(f'tpot_ms_{stat}') and r.get(f'{stat}_tpot_s'):
            item[f'tpot_ms_{stat}'] = float(r[f'{stat}_tpot_s']) * 1000


# Cold Prefill: use native source tables; exclude historical reference rows.
p = ARCHIVE / 'glm47_h100_expert_tp8_ep8_pp2ep4_20261007/metrics.csv'
for r in rows(p):
    if r['arm'] not in ('ep8', 'pp2ep4') or not r['workload'].startswith('prefill'):
        continue
    b = int(float(r['batch']))
    item = add(p, 'P01' if b == 4 else 'P02', r['arm'], 'prefill', gpu='H100', B=b, P=0,
               Q=8192, O=1, M=0, cuda_graph='off', cache_protocol='no_prefix_cache',
               throughput_tok_s=r['input_tok_s'], throughput_unit='input_tok/s',
               notes='EP/PP native libraries differ; EP local chunk512 vs PP1024; three formal trials')
    flat_stats(item, r)

for ds, name in [('P03', 'glm47_a100_ep8_pp2ep4_optimized_20261008_r4'),
                 ('P04', 'glm47_a100_ep8_pp2ep4_pair_rdma_20261010_r1'),
                 ('P10', 'glm47_a100_ep8_pp_attribution_20261010_r1')]:
    p = ARCHIVE / name / 'full_metrics.csv'
    for r in rows(p):
        if 'prefill' not in r['workload'].lower():
            continue
        item = add(p, ds, r['arm'], 'prefill', B=4, P=0, Q=8192, O=1, M=0,
                   cuda_graph='off', cache_protocol='no_prefix_cache',
                   throughput_tok_s=r['input_tok_s'], throughput_unit='input_tok/s',
                   notes='unprofiled end-to-end cohort; profile is a separate diagnostic' if ds == 'P10' else '')
        flat_stats(item, r)

p = ARCHIVE / 'glm47_a100_pcie_prefill_20261010_r2/five_conditions.csv'
for r in rows(p):
    if r['condition'] in ('old_pcie_threshold1_prefill', 'old_pcie_threshold1_decode'):
        continue  # Already listed as P03 / D03, not new measurements.
    phase = r['phase']
    ds = 'P05' if 'new_pcie' in r['condition'] else ('P06' if phase == 'prefill' else 'D04b')
    for arm in ('ep8', 'pp2ep4'):
        sub = {k[len(arm) + 1:]: v for k, v in r.items() if k.startswith(arm + '_')}
        item = add(p, ds, arm, phase, B=4, P=0, Q=8192 if phase == 'prefill' else 128,
                   O=1 if phase == 'prefill' else 512, M=0,
                   cuda_graph='off' if phase == 'prefill' else 'on',
                   cache_protocol='no_prefix_cache', cohort=r['condition'],
                   throughput_tok_s=sub.get('input_tok_s' if phase == 'prefill' else 'output_tok_s'),
                   throughput_unit='input_tok/s' if phase == 'prefill' else 'output_tok/s',
                   notes='Matched local chunk1024/capacity256; inspect condition for SM-PCIe vs RDMA')
        flat_stats(item, sub)

p = ARCHIVE / 'glm47_a100_ep8_pp2ep4_ce_prefill_20261010_r4/full_metrics.csv'
for r in rows(p):
    if r['cohort'] != 'current_ce':
        continue
    item = add(p, 'P07', r['arm'], 'prefill', B=4, P=0, Q=8192, O=1, M=0,
               cuda_graph='off', cache_protocol='no_prefix_cache',
               throughput_tok_s=r['input_tok_s'], throughput_unit='input_tok/s',
               throughput_http_completion_tok_s=r['input_tok_s_http_completion'],
               notes='CE c3405cad; primary throughput retains original archival-inclusive wall boundary')
    for field in ('ttft_s', 'e2e_s'):
        stats(item, json.loads(r[field]), field)

p = WORK / 'experiments/hy3_stream_cached_measure_v4_20261020/comparison.csv'
for r in rows(p):
    if r['workload'] != 'prefill':
        continue
    item = add(p, 'P08' if r['cached_context_per_request'] == '2304' else 'P09', r['layout'], 'prefill',
               model='Hy3 BF16', B=8, P=r['cached_context_per_request'], M=r['total_cached_context'],
               Q=256, O=1, cuda_graph='off', cache_protocol='synthetic_CPU_HiCache_full_GPU_empty',
               throughput_tok_s=r['throughput_tok_s'], throughput_unit='new_input_tok/s', cohort=r['run'],
               notes='H2D/admission timed; streaming weight load excluded; local chunk512')
    for stat in ('mean', 'median', 'p99'):
        item[f'ttft_s_{stat}'] = float(r[f'ttft_ms_{stat}']) / 1000
        item[f'e2e_s_{stat}'] = r[f'end_to_end_s_{stat}']

p = ARCHIVE / 'glm47_a100_ep8_pp2ep4_m655360_b8_decode_20261022_r1/full_metrics.csv'
for r in rows(p):
    item = add(p, 'D09', r['layout'], B=8, P=81920, Q=1, O=128, M=655360,
               cuda_graph='off', cache_protocol='synthetic_CPU_HiCache_full_GPU_empty',
               throughput_tok_s=r['output_tok_s'], throughput_unit='output_tok/s',
               notes='065/167; H2D+admission timed; one cohort; separate GPU-hot O32 profile F03')
    flat_stats(item, r)

# Fill Prefill counterparts for the nine-point 2xEP4 study and its A100 boundary row.
for name in ['glm47_h100_hicache_grid_strict9_20261016_v2', 'glm47_a100_m655360_b8_hicache_cold_20261015_r2']:
    p = ARCHIVE / name / ('strict9_metrics.csv' if 'strict9' in name else 'full_metrics.csv')
    for r in rows(p):
        if r['workload'] != 'prefill':
            continue
        item = add(p, 'S01' if 'h100' in name else 'S02', r['arm'], 'prefill',
                   comparison_scope='related_2xEP4', gpu='H100' if 'h100' in name else 'A100',
                   B=r.get('request_count', r.get('requests')), P=r.get('prefix_tokens', r.get('context_per_request')),
                   M=r['total_context_tokens'], Q=256, O=1, cuda_graph='off',
                   cache_protocol='synthetic_CPU_HiCache_full_GPU_empty',
                   throughput_tok_s=r['throughput_tokens_per_s'], throughput_unit=r['throughput_unit'])
        flat_stats(item, r)

# Semantic bank: every production cohort retains its own request statistics.
run = WORK / 'experiments/glm47_semantic_bank_fourway_20261023_v5/runs/fourway_reuse_r3'
for arm in ('ep8_mtp_off', 'pp2ep4_mtp_off', 'ep8_mtp_on', 'pp2ep4_mtp_on'):
    for suffix in ('', '_repeat'):
        p = run / arm / f'p4096_decode{suffix}_results.json'
        v = js(p)
        item = add(p, 'D10' if arm.endswith('off') else 'D11', v['layout'], B=8, P=4096,
                   actual_CPU_P=4032, first_forward_Q=65, Q=1, O=64, M=32768,
                   cuda_graph='on', cache_protocol='semantic_CPU_HiCache_full_GPU_empty',
                   throughput_tok_s=v['throughput_tok_s'], throughput_unit='output_tok/s',
                   cohort=v['cohort'], mtp=v['mtp'], profile_id='F04',
                   notes='Separate v8 event profile; old Verify backend' if v['mtp'] else 'Separate v8 event profile')
        stats(item, v['ttft_s'], 'ttft_s')
        stats(item, v['tpot_s'], 'tpot_ms', 1000)
        stats(item, v['end_to_end_s'], 'e2e_s')

p = WORK / 'experiments/glm47_mtp_verify_fixed_ep8_pp2ep4_20261023/summary.json'
v = js(p)
for arm, a in v['arms'].items():
    for cohort, c in a['cohorts'].items():
        item = add(p, 'D12', c['layout'], B=8, P=4096, actual_CPU_P=4032, first_forward_Q=65,
                   Q=1, O=64, M=32768, cuda_graph='on', mtp=True,
                   cache_protocol='semantic_CPU_HiCache_full_GPU_empty',
                   throughput_tok_s=c['output_throughput_tok_s'], throughput_unit='output_tok/s',
                   cohort=cohort, profile_id='F05', notes='Corrected Verify; lossless speculation not qualified')
        stats(item, c['ttft_ms'], 'ttft_s', .001)
        stats(item, c['tpot_ms'], 'tpot_ms')
        stats(item, c['end_to_end_ms'], 'e2e_s', .001)

p = WORK / 'experiments/glm47_pp_stage_balance_20261023/summary.json'
v = js(p)
variants = [(k, x['performance']) for k, x in v['previous_optimization']['arms'].items()]
variants.append(('pp2ep4_balanced_50_42', v['balanced']['performance']))
for variant, perf in variants:
    item = add(p, 'O01', 'pp2ep4', comparison_scope='PP_only_optimization_historical_EP_reference',
               variant=variant, B=8, P=4096, Q=1, O=64, M=32768, cuda_graph='on', mtp=True,
               cache_protocol='semantic_CPU_HiCache_full_GPU_empty',
               throughput_tok_s=perf['throughput_tok_s']['mean'], throughput_unit='output_tok/s',
               statistic_scope='mean_of_two_cohort_means; median/P99 not aggregated', profile_id='F06')
    item['ttft_s_mean'] = perf['ttft_s']['mean']
    item['tpot_ms_mean'] = perf['tpot_s']['mean'] * 1000
    item['e2e_s_mean'] = perf['end_to_end_s']['mean']

for ds, folder, runname in [('O02', 'glm47_pp_feedback_payload_ab_20261024', 'feedback_ab_r1'),
                          ('O03', 'glm47_pp_seq_cpu_reuse_20261024', 'seq_cpu_ab_r1')]:
    p = WORK / 'experiments' / folder / 'runs' / runname / 'comparison.json'
    v = js(p)
    for c in v['cohorts']:
        add(p, ds, 'pp2ep4', comparison_scope='PP_only_optimization_no_fresh_EP8',
            variant=c['variant'], cohort=c['cohort'], B=8, P=4096, Q=1, O=64, M=32768,
            cuda_graph='on', mtp=True, cache_protocol='semantic_CPU_HiCache_full_GPU_empty',
            throughput_tok_s=c['throughput_tok_s'], throughput_unit='output_tok/s',
            ttft_s_mean=c['ttft_ms'] / 1000, tpot_ms_mean=c['tpot_ms'], e2e_s_mean=c['e2e_ms'] / 1000,
            statistic_scope='per_cohort_request_mean; median/P99 not provided by comparison table',
            notes='Same-process ABBA, PP50/42; diagnostic scheduling timeline is separate')

# Profiles: units/scope are explicit, never pretend these are production TPOT partitions.
def profile(p, pid, **kw):
    profiles.append(dict(profile_id=pid, source_path=source(p)['path'],
                         source_sha256=source(p)['sha256'], **kw))

p = ARCHIVE / 'glm47_a100_ep8_pp_attribution_20261010_r1/analysis/root_layer_comparison.csv'
for r in rows(p):
    profile(p, 'F01', arm=r['arm'], role='prefill', scope='mean_per_rank_MoE_layer_call',
            P=r['cached_prefix_tokens'], Q=r['local_new_tokens'],
            attention_ms=r['attention'], routed_expert_ms=r['expert.routed'], shared_expert_ms=r['expert.shared'],
            dispatch_ms=r['deep_ep.dispatch'], combine_ms=r['deep_ep.combine'], tp_ms=r['tp_collective'],
            total_layer_ms=r['layer_ms'], samples=r['sampled_rank_layer_calls'])

p = ARCHIVE / 'glm47_h100_ep8_pp_attribution_20261008/layer_metrics.csv'
for r in rows(p):
    profile(p, 'F02', arm=r['arm'], role='prefill', rank=r['rank'], layer=r['layer'],
            scope='one_sampled_real_layer_exclusive_partition', B=r['B'], P=r['P'], Q=r['T'],
            attention_ms=r['attention'], routed_expert_ms=r['routed_expert'],
            dispatch_combine_ms=r['deep_ep'], tp_ms=r['tp_collective'], total_layer_ms=r['layer_ms'],
            notes='Published layer table combines Dispatch+Combine; no invented split')

p = WORK / 'experiments/glm47_a100_cached_decode_gpu_activity_20261022/analysis_output/gpu_r1/summary.json'
v = js(p)
for arm, groups in v['groups'].items():
    for group in (['all_rank_mean'] if arm == 'ep8' else ['stage0', 'stage1']):
        g = groups[group]
        phases = g['mean_phase_union_ms']
        profile(p, 'F03', arm=arm, stage=group, role='decode_GPU_hot_O32', B=8, P=81920, Q=1,
                scope='per_rank_cycle_phase_union; overlapping phases are not additive',
                attention_ms=phases['attention'], all_expert_ms=phases['expert_ffn'],
                dispatch_ms=phases['deep_ep.dispatch'], combine_ms=phases['deep_ep.combine'],
                tp_ms=phases.get('tp_communication'), pp_ms=phases.get('pp_communication'),
                microbatch_cycle_ms=g['mean_microbatch_cycle_ms'],
                notes='Separate from D09 O128 production; profiler perturbs timing; no routed/shared expert split')

event_sets = [
    ('F04', WORK / 'experiments/glm47_semantic_bank_fourway_20261023_v8/runs/event_profile_followup_r1',
     ['ep8_mtp_off', 'pp2ep4_mtp_off', 'ep8_mtp_on', 'pp2ep4_mtp_on']),
    ('F05', WORK / 'experiments/glm47_mtp_verify_fixed_ep8_pp2ep4_20261023/runs/paired_r1',
     ['ep8_mtp_fixed', 'pp2ep4_mtp_fixed']),
    ('F06', WORK / 'experiments/glm47_pp_stage_balance_20261023/runs/balance_r1', ['pp2ep4_balanced']),
]
for pid, root, arms in event_sets:
    for arm in arms:
        p = root / arm / 'stage_activity_summary.json'
        v = js(p)
        assert v['passed'], (pid, arm)
        groups = defaultdict(list)
        for r in v['ranks']:
            for role in ('target', 'draft'):
                if r.get(role):
                    groups[('all8' if arm.startswith('ep8') else f'stage{int(r["rank"]) // 4}', role)].append(r[role])
        for (stage, role), records in groups.items():
            avg = lambda key: statistics.mean(x[key] for x in records)
            exclusive = lambda key: statistics.mean(x['exclusive_ms'].get(key, 0) for x in records)
            profile(p, pid, arm=arm, stage=stage, role=role, B=8, P=4096,
                    scope='per_actual_Graph_role_per_rank_mean; not ms_per_output_token',
                    rank_count=len(records), graph_ms=avg('graph_span_ms'), attention_ms=avg('attention_ms'),
                    all_expert_ms=avg('expert_ms'), routed_expert_ms=exclusive('expert_routed'),
                    shared_expert_ms=exclusive('expert_shared'), dense_ffn_ms=exclusive('dense_ffn'),
                    dispatch_ms=exclusive('deepep_dispatch'), combine_ms=exclusive('deepep_combine'),
                    tp_ms=exclusive('nccl_tp'), other_ms=avg('other_graph_ms'),
                    notes='Outer completion event changes cadence; omit HiCache/PP boundary from Graph-only timing')


def write_csv(path, records):
    fields = list(dict.fromkeys(k for r in records for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


write_csv(HERE / 'metrics.csv', metrics)
write_csv(HERE / 'profiles.csv', profiles)
unique_checks = {x['path']: x for x in hash_checks}
validation = {
    'no_gpu_launch': True,
    'metric_rows': len(metrics), 'profile_rows': len(profiles), 'source_files': len(sources),
    'metrics_by_scope': dict(Counter(x['comparison_scope'] for x in metrics)),
    'metrics_by_workload': dict(Counter(x['workload'] for x in metrics)),
    'report_snapshot_hash_checks': list(unique_checks.values()),
    'all_original_report_sources_unchanged': all(x['matches_report_snapshot'] for x in unique_checks.values()),
    'missing_metrics_are_blank_not_zero': True,
    'profile_is_not_production_tpot_partition': True,
    'historical_mtp_correctness_not_qualified': True,
}
readme = HERE / 'README.md'
if readme.exists():
    links = re.findall(r'\]\((/[^)]+)\)', readme.read_text())
    missing = [p for p in links if not Path(p).exists()]
    validation['markdown_absolute_links_checked'] = len(links)
    validation['missing_markdown_links'] = missing
    assert not missing, missing
assert validation['all_original_report_sources_unchanged']
(HERE / 'provenance.json').write_text(json.dumps({'generated_date': '2026-10-24', 'sources': list(sources.values())}, indent=2) + '\n')
(HERE / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
print(json.dumps({k: v for k, v in validation.items() if k != 'report_snapshot_hash_checks'}, indent=2))
