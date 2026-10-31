"""Build the PP/EP regression specification and index saved results. No GPU work."""
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from statistics import fmean
import csv
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parent
ARCHIVES = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')
BENCHMARK_ID = 'pp_ep_regression_benchmark_v1_20261027'
SOURCE_IDS = {
    'decode': 'qwen35b_h100_pp_lookahead_workloads_20261027',
    'prefill': 'qwen35b_h100_prefill_optimized_20261027',
    'a100_35': 'qwen35b_a100_ep2_pp2_20261025_r2',
    'a100_122': 'qwen122_a100_nvlink_ep2_pp2_20261025',
    'h100_122': 'qwen_h100_ep2_pp2_20261025_r7',
    'a100_domains': 'a100_ep_pp_comm_domain_comparison_20261026',
    'h100_domains': 'h100_ep_pp_comm_domain_comparison_20261026',
    'geometry': 'ep_pp_bandwidth_model_20261027',
}
SOURCES = {}


def read(label, relative):
    path = ARCHIVES / SOURCE_IDS[label] / relative
    raw = path.read_bytes()
    SOURCES[str(path)] = {
        'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
        'bytes': len(raw), 'snapshot': f'sources/{label}/{relative}',
    }
    return raw.decode()


def rows(label, relative):
    return list(csv.DictReader(read(label, relative).splitlines()))


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_csv(name, data):
    with (HERE / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


manifests = {k: json.loads(read(k, 'manifest.json')) for k in SOURCE_IDS}
geometry = {r['model']: r for r in rows('geometry', 'model_geometry.csv')}
for label in ['a100_35', 'a100_122', 'h100_122', 'a100_domains', 'h100_domains']:
    rows(label, 'full_metrics.csv')  # Source inventory only, never latest coverage.

models = [
    {'id': 'qwen35b_bf16', 'name': 'Qwen3.6-35B-A3B', 'precision': 'BF16',
     'gpus': 2, 'nodes': 1, 'communication_domain': 'one_NVLink_pair',
     'layouts': [
         {'id': 'ep2', 'pp': 1, 'expert_ep': 2, 'attention_tp': 1, 'attention_dp': 2},
         {'id': 'pp2', 'pp': 2, 'expert_ep': 1, 'attention_tp': 1, 'attention_dp': 1}],
     'historical': {'A100': 'a100_35', 'H100': 'decode'}},
    {'id': 'qwen122b_fp8', 'name': 'Qwen3.5-122B-A10B-FP8', 'precision': 'FP8 checkpoint',
     'gpus': 2, 'nodes': 1, 'communication_domain': 'one_NVLink_pair',
     'layouts': [
         {'id': 'ep2', 'pp': 1, 'expert_ep': 2, 'attention_tp': 1, 'attention_dp': 2},
         {'id': 'pp2', 'pp': 2, 'expert_ep': 1, 'attention_tp': 1, 'attention_dp': 1}],
     'historical': {'A100': 'a100_122', 'H100': 'h100_122'}},
    {'id': 'glm47_awq', 'name': 'GLM-4.7-AWQ', 'precision': 'INT4 AWQ checkpoint',
     'gpus': 8, 'nodes': 2, 'communication_domain': 'two_nodes_four_GPUs_per_node',
     'layouts': [
         {'id': 'ep8', 'pp': 1, 'expert_ep': 8, 'attention_tp': 4, 'attention_dp': 2},
         {'id': 'pp2_ep4', 'pp': 2, 'expert_ep': 4, 'attention_tp': 4, 'attention_dp': 1}],
     'historical': {'A100': 'a100_domains', 'H100': 'h100_domains'}},
]
for model in models:
    geometry_key = 'Qwen3.5-122B-A10B-FP8' if model['id'] == 'qwen122b_fp8' else model['name']
    model['routed_layer_geometry'] = geometry[geometry_key]
    model['historical_manifests'] = {
        gpu: str(ARCHIVES / SOURCE_IDS[label] / 'manifest.json')
        for gpu, label in model.pop('historical').items()
    }

workloads = []
for b, family in [(8, 'repeated'), (16, 'repeated'), (32, 'repeated'), (32, 'diverse')]:
    workloads.append({'id': f'decode_b{b}_{family}', 'source_workload': f'b{b}_{family}',
                      'phase': 'decode', 'B': b, 'P': 4096, 'Q': 1, 'O': 128,
                      'family': family, 'cache_location': 'GPU_hot_model_produced',
                      'profile': 'drop20_select16_actual_Decode_forwards_each_rank'})
for b, q, family in [*( (b, q, 'repeated') for b, q in product([8, 32], [128, 512, 4096]) ),
                     (32, 4096, 'diverse')]:
    workloads.append({'id': f'prefill_b{b}_q{q}_{family}', 'source_workload': f'b{b}_q{q}_{family}',
                      'phase': 'prefill', 'B': b, 'P': 0, 'Q': q, 'O': 1,
                      'family': family, 'cache_location': 'none',
                      'profile': 'all_actual_Prefill_layer_calls_by_host_launch_correlation'})

suite = {
    'id': BENCHMARK_ID, 'version': 1, 'created_at': '2026-10-27',
    'kind': 'benchmark_specification_and_saved_result_index', 'launches_gpu': False,
    'hardware': ['A100', 'H100'], 'models': models, 'workloads': workloads,
    'measurement': {
        'warmup_cohorts': 3, 'formal_cohorts_per_cell': 5,
        'formal_profiler': False, 'independent_profile_required': True,
        'order': 'counterbalance_before_after_and_layout_within_each_model_GPU_pair',
        'same_GPU_set_per_comparison': True, 'fixed_output_count_ignore_eos': True,
        'prefill_budget_tokens_per_attention_instance': 4096,
        'pp_slots': 2, 'pp_requests_per_slot': 'B/2',
        'pp_layer_split': 'equal target layers initially; record exact split',
        'speculative_decoding': False,
        'before_after': 'same benchmark cells and workload hashes; record both revisions',
        'support': 'A100/122B/GLM latest optimization needs compatibility/admission qualification',
        'prompts': 'freeze text and per-model token IDs; same groups/route inputs within comparison',
        'large_model_workload_note': 'B8/B16/B32 GPU-hot protocol is a new target; historical cold128->512 is not coverage',
    },
    'metrics': {
        'prefill': ['input_tok_s_cohort', 'ttft_ms', 'e2e_ms', 'makespan_ms', 'sample_stdev'],
        'decode': ['output_tok_s_cohort', 'tpot_ms', 'ttft_ms', 'e2e_ms', 'sample_stdev'],
        'components': ['routed_expert_gemm_us_per_actual_layer_call',
                       'dispatch_us_per_actual_layer_call', 'combine_us_per_actual_layer_call',
                       'attention_core_us_per_actual_layer_call', 'attention_module_us_if_available',
                       'pp_forward_handoff_ms', 'pp_feedback_ms', 'pp_stage_model_graph_ms',
                       'actual_local_source_tokens', 'actual_layer_calls',
                       'active_experts_and_assignments', 'overlap_scope_and_profile_window'],
        'missing_components': 'explicit not_measured, never zero-fill',
        'percentiles': 'mean/median plus request-level tails when raw samples exist; retain sample count',
    },
    'latest_reference': {
        'generic': 'same T<=16 Expert selector in EP/PP; large Prefill uses native GEMM',
        'decode': 'H10035 PP policy9 vs PP policy0; EP shared-expert overlap ON; TBO OFF',
        'prefill': 'PP policy0 reference; policy1 early-send retained as ablation, no stable gain',
        'cross_layout_quality': 'separate from measured timing and within-PP schedule equivalence',
    },
}

latest = []
comparisons_source = {phase: rows(phase, 'analysis/comparisons.csv') for phase in ['decode', 'prefill']}
prefill_agreement = {r['workload']: r['cross_layout_first_token_equal'] for r in comparisons_source['prefill']}
for phase in ['decode', 'prefill']:
    for row in rows(phase, 'analysis/metrics.csv'):
        if phase == 'decode' and row['phase'] != 'decode':
            continue
        workload = next(w for w in workloads if w['phase'] == phase and w['source_workload'] == row['workload'])
        arm, policy = row['arm'], int(row['policy'])
        variant = ('ep_shared_expert_kernel' if arm == 'ep2' else
                   'pp_lookahead' if phase == 'decode' and policy == 9 else
                   'pp_early_send_ablation' if phase == 'prefill' and policy == 1 else 'pp_native')
        is_target = arm == 'ep2' or policy == (9 if phase == 'decode' else 0)
        quality = ('within_PP_full_generation_equal;cross_layout_equivalence_not_asserted' if phase == 'decode' else
                   'first_token_match_only' if prefill_agreement[row['workload']] == 'True' else
                   'cross_layout_numerical_difference_unresolved')
        latest.append({
            'model_id': 'qwen35b_bf16', 'gpu': 'H100', 'layout': arm,
            'workload_id': workload['id'], 'source_workload': row['workload'], 'phase': phase,
            'B': workload['B'], 'P': workload['P'], 'Q': workload['Q'], 'O': workload['O'],
            'family': workload['family'], 'policy': policy, 'variant': variant,
            'latest_reference_cell': is_target, 'repeats': int(row['repeats']),
            'throughput_tok_s': float(row['throughput_tok_s']),
            'throughput_stdev': float(row['throughput_stdev']),
            'ttft_ms': float(row['ttft_ms']), 'tpot_ms': row.get('tpot_ms', ''),
            'e2e_ms': float(row['e2e_ms']), 'quality_scope': quality,
            'source_csv': str(ARCHIVES / SOURCE_IDS[phase] / 'analysis/metrics.csv'),
            'raw_metrics': row['source'], 'raw_metrics_sha256': row['sha256'],
        })

component_rows = []
decode_groups = defaultdict(list)
for r in rows('decode', 'analysis/components.csv'):
    decode_groups[(r['workload'], r['arm'], int(r['policy']))].append(r)
for (workload, arm, policy), group in decode_groups.items():
    assert len(group) == 2 and {r['rank'] for r in group} == {'0', '1'}
    def mean(key):
        if arm == 'pp2' and key in ['communication.dispatch', 'communication.combine']:
            assert all(r[key] == '' or float(r[key]) == 0 for r in group)
            return 0.0  # This layout is EP1; the operation does not exist.
        return fmean(float(r[key]) for r in group) * 1000
    component_rows.append({
        'workload_id': f'decode_{workload}', 'layout': arm, 'policy': policy,
        'expert_gemm_us': mean('expert.routed_gemm'),
        'dispatch_us': mean('communication.dispatch'), 'combine_us': mean('communication.combine'),
        'moe_communication_scope': 'no_exchange_EP1' if arm == 'pp2' else 'measured_DeepEP',
        'attention_us': mean('attention.core'), 'attention_scope': 'core_only',
        'local_source_tokens_mean': group[0]['local_tokens'],
        'actual_layer_calls_per_card': fmean(int(r['actual_layer_invocations']) for r in group),
        'unit': 'us/card/actual_layer_call; average of two rank means',
        'profile_window': 'drop20/select16 Graph forwards; independent from formal timing',
        'source_csv': str(ARCHIVES / SOURCE_IDS['decode'] / 'analysis/components.csv'),
    })
for r in rows('prefill', 'analysis/components.csv'):
    component_rows.append({
        'workload_id': f"prefill_{r['workload']}", 'layout': r['arm'], 'policy': int(r['policy']),
        'expert_gemm_us': float(r['expert_gemm_us_per_layer']),
        'dispatch_us': float(r['dispatch_us_per_layer']), 'combine_us': float(r['combine_us_per_layer']),
        'moe_communication_scope': 'no_exchange_EP1' if r['arm'] == 'pp2' else 'measured_DeepEP',
        'attention_us': float(r['attention_module_us_per_layer']),
        'attention_scope': 'module_including_projections;not_same_as_Decode_core',
        'local_source_tokens_mean': r['local_source_tokens_mean'],
        'actual_layer_calls_per_card': int(r['actual_layer_calls']),
        'unit': 'us/card/actual_layer_call; average of two rank means',
        'profile_window': 'all cold Prefill calls; separate profiler run',
        'source_csv': str(ARCHIVES / SOURCE_IDS['prefill'] / 'analysis/components.csv'),
    })
read('decode', 'analysis/profile_cycles.csv')
read('decode', 'analysis/profile_summary.csv')
read('prefill', 'report/followups.json')
read('prefill', 'report/numerical_summary.csv')

by_cell = {(r['model_id'], r['gpu'], r['layout'], r['workload_id']): r
           for r in latest if r['latest_reference_cell']}
# Optional dated A100 extension. Read only the completed immutable archive;
# an in-progress workspace run can never become benchmark coverage here.
a100_latest = []
a100_archive = ARCHIVES / 'qwen35b_a100_nvlink_regression_20261027'
if (a100_archive / 'manifest.json').exists():
    SOURCE_IDS['a100_latest'] = a100_archive.name
    manifests['a100_latest'] = json.loads(read('a100_latest', 'manifest.json'))
    assert manifests['a100_latest']['status'] == 'complete'
    a100_result = json.loads(read('a100_latest', 'analysis/result.json'))
    assert a100_result['audit']['guard_postcheck'] and len(a100_result['metrics']) == 15
    matches = {r['workload']: r for r in a100_result['cross_layout_output_agreement']}
    for r in a100_result['metrics']:
        w = next(w for w in workloads if w['id'] == r['workload'])
        match = matches[r['workload']]
        item = {
            'model_id': 'qwen35b_bf16', 'gpu': 'A100', 'layout': r['arm'],
            'workload_id': w['id'], 'source_workload': r['workload'], 'phase': r['phase'],
            'B': w['B'], 'P': w['P'], 'Q': w['Q'], 'O': w['O'], 'family': w['family'],
            'policy': r['policy'],
            'variant': 'ep_native_triton' if r['arm'] == 'ep2' else 'pp_lookahead' if r['policy'] == 9 else 'pp_native',
            'latest_reference_cell': r['arm'] == 'ep2' or r['policy'] == (9 if r['phase'] == 'decode' else 0),
            'repeats': r['formal_cohorts'], 'throughput_tok_s': r['throughput_tok_s'],
            'throughput_stdev': r['throughput_stdev'], 'ttft_ms': r['ttft_ms'],
            'tpot_ms': r['tpot_ms'], 'e2e_ms': r['e2e_ms'],
            'quality_scope': f"within_PP_full_outputs_equal;cross_layout_full_output_match={match['equal_requests']}/{match['total_requests']};no_task_quality_evaluation",
            'execution': 'A100 eager; native Triton Expert/Attention; DeepEP normal BF16; overlap schedule OFF',
            'source_csv': str(a100_archive / 'analysis/metrics.csv'),
            'raw_metrics': r['source'], 'raw_metrics_sha256': r['sha256'],
        }
        a100_latest.append(item)
        if item['latest_reference_cell']:
            by_cell[(item['model_id'], item['gpu'], item['layout'], item['workload_id'])] = item
    assert sum(r['latest_reference_cell'] for r in a100_latest) == 12
    assert {r['repeats'] for r in a100_latest} == {5}
    write_csv('latest_a10035_results.csv', a100_latest)
    write_csv('latest_a10035_components.csv', a100_result['components'])
    write_csv('latest_a10035_comparisons.csv', a100_result['comparisons'])
cases = []
for model, gpu, workload in product(models, suite['hardware'], workloads):
    for layout in model['layouts']:
        existing = by_cell.get((model['id'], gpu, layout['id'], workload['id']))
        cases.append({
            'case_id': '__'.join([model['id'], gpu.lower(), layout['id'], workload['id']]),
            'model_id': model['id'], 'model': model['name'], 'gpu': gpu,
            'gpus': model['gpus'], 'nodes': model['nodes'], 'layout': layout['id'],
            'attention_tp': layout['attention_tp'], 'attention_dp': layout['attention_dp'],
            'expert_ep': layout['expert_ep'], 'pp': layout['pp'], 'pp_slots': 2 if layout['pp'] > 1 else '',
            'workload_id': workload['id'], 'phase': workload['phase'],
            'B': workload['B'], 'P': workload['P'], 'Q': workload['Q'], 'O': workload['O'],
            'family': workload['family'], 'formal_repeats_target': 5,
            'latest_status': 'measured' if existing else 'pending_compatibility_and_rerun',
            'quality_scope': existing['quality_scope'] if existing else 'not_measured',
            'latest_metrics_source': existing['source_csv'] if existing else '',
            'historical_manifest': model['historical_manifests'][gpu],
        })

comparisons = []
for w in workloads:
    pair = [r for r in latest if r['workload_id'] == w['id'] and r['latest_reference_cell']]
    ep = next(r for r in pair if r['layout'] == 'ep2')
    pp = next(r for r in pair if r['layout'] == 'pp2')
    comparisons.append({
        'workload_id': w['id'], 'phase': w['phase'], 'ep_tok_s': ep['throughput_tok_s'],
        'pp_tok_s': pp['throughput_tok_s'],
        'pp_vs_ep_throughput_pct': (pp['throughput_tok_s']/ep['throughput_tok_s']-1)*100,
        'ep_tpot_ms': ep['tpot_ms'], 'pp_tpot_ms': pp['tpot_ms'],
        'ep_ttft_ms': ep['ttft_ms'], 'pp_ttft_ms': pp['ttft_ms'],
        'pp_variant': pp['variant'], 'quality_scope': pp['quality_scope'],
    })

assert len(cases) == 132 and len({r['case_id'] for r in cases}) == 132
assert len(latest) == len(component_rows) == 33 and len(by_cell) == 22 + (12 if a100_latest else 0)
assert len(comparisons) == 11
assert sum(c['latest_status'] != 'measured' for c in cases) == len(cases) - len(by_cell)
assert sum('difference_unresolved' in c['quality_scope'] for c in cases) == 8
assert {r['repeats'] for r in latest} == {5}
assert {(r['workload_id'],r['layout'],r['policy']) for r in latest} == {
    (r['workload_id'],r['layout'],r['policy']) for r in component_rows}
suite['case_count_per_revision'] = len(cases)
suite['coverage_as_of_20261027'] = {'measured': len(by_cell), 'pending': len(cases) - len(by_cell),
    'a10035_measured_cells': 12 if a100_latest else 0,
    'measured_short_prefill_cells_with_unresolved_cross_layout_numerics': 8,
    'note': 'coverage is timing availability, not full correctness or hardware qualification'}
write_json('suite_v1.json', suite)
write_csv('cases.csv', cases)
write_csv('latest_h10035_results.csv', latest)
write_csv('latest_h10035_components.csv', component_rows)
write_csv('latest_h10035_comparisons.csv', comparisons)
coverage = []
for model, gpu in product(models, suite['hardware']):
    subset = [c for c in cases if c['model_id'] == model['id'] and c['gpu'] == gpu]
    coverage.append({'model_id': model['id'], 'gpu': gpu, 'target_cells': len(subset),
                     'latest_measured': sum(c['latest_status']=='measured' for c in subset),
                     'latest_pending': sum(c['latest_status']!='measured' for c in subset),
                     'historical_manifest': model['historical_manifests'][gpu]})
write_csv('coverage.csv', coverage)

def comparison_table(phase):
    if phase == 'decode':
        lines = ['| Workload | EP tok/s ↑ | PP native tok/s ↑ | PP lookahead tok/s ↑ | PP vs EP | EP / PP lookahead TPOT ms ↓ |',
                 '|---|---:|---:|---:|---:|---:|']
        for r in comparisons_source['decode']:
            lines.append(f"| {r['workload']} | {float(r['ep_tok_s']):,.1f} | {float(r['pp_baseline_tok_s']):,.1f} | {float(r['pp_lookahead_tok_s']):,.1f} | {float(r['pp_vs_ep_pct']):+.2f}% | {float(r['ep_tpot_ms']):.3f} / {float(r['pp_lookahead_tpot_ms']):.3f} |")
    else:
        lines = ['| Workload | EP input tok/s ↑ | PP native input tok/s ↑ | PP vs EP | Cross-layout first token |',
                 '|---|---:|---:|---:|---|']
        for r in comparisons_source['prefill']:
            gain = (float(r['pp_native_tok_s'])/float(r['ep_tok_s'])-1)*100
            quality = '一致，仅验证首 token' if r['cross_layout_first_token_equal'] == 'True' else '不一致，数值问题待查'
            lines.append(f"| {r['workload']} | {float(r['ep_tok_s']):,.1f} | {float(r['pp_native_tok_s']):,.1f} | {gain:+.2f}% | {quality} |")
    return '\n'.join(lines)

template = (HERE/'REPORT_TEMPLATE.md').read_text()
report = template.replace('@@DECODE_TABLE@@', comparison_table('decode')).replace('@@PREFILL_TABLE@@', comparison_table('prefill'))
if a100_latest:
    a100_section = f'''### 新增 A100 35B NVLink 补测

已补齐6组workload的EP2／PP2，共12个主矩阵组合，另保存3个原生PP Decode对照；每配置5轮。物理GPU0/1在同一NV12域。A100采用原生Triton Expert／Attention、DeepEP normal BF16、eager执行；PP Decode policy9调度优化另列，通用Expert后端双方相同。

[完整报告]({a100_archive}/report/REPORT.md) · [A100指标](benchmark/latest_a10035_results.csv) · [A100分项](benchmark/latest_a10035_components.csv) · [EP／PP对照](benchmark/latest_a10035_comparisons.csv)

A100与H100保留各自的原生执行路径，跨硬件差值不只包含GPU硬件。所有EP／PP生成输出一致率、缓存与通信校验见报告。其他5组A10035 workload及其他模型仍待补测。
'''
    report = report.replace('@@A100_SECTION@@', a100_section)
    report = report.replace('有历史 EP2／PP2，最新优化待适配／重测', '**6 workload、12／22组合已补测**')
    report = report.replace('22／132 已有最新参考结果，110／132 待验证', '34／132 已有最新参考结果，98／132 待验证')
else:
    report = report.replace('@@A100_SECTION@@', '')
(SUMMARY/'BENCHMARK.md').write_text(report)
write_json('sources.json', {'files': list(SOURCES.values())})
for source in SOURCES.values():
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
for result in latest:
    assert hashlib.sha256(Path(result['raw_metrics']).read_bytes()).hexdigest() == result['raw_metrics_sha256']
for result in a100_latest:
    assert hashlib.sha256(Path(result['raw_metrics']).read_bytes()).hexdigest() == result['raw_metrics_sha256']
write_json('validation.json', {
    'status': 'passed', 'no_gpu_launch': True, 'cases': len(cases),
    'coverage': Counter(c['latest_status'] for c in cases), 'latest_results': len(latest),
    'latest_component_rows': len(component_rows), 'latest_comparison_rows': len(comparisons),
    'latest_a100_timing_rows': len(a100_latest),
    'all_latest_rows_have_five_repeats': True, 'source_hashes_verified': len(SOURCES),
    'raw_metric_hashes_verified': len(latest) + len(a100_latest),
    'historical_results_counted_as_latest': False,
    'component_and_metric_keys_match': True, 'scope': 'offline inventory and arithmetic, not new measurements',
})
write_json('manifest.json', {
    'id': BENCHMARK_ID, 'title': 'PP/EP regression benchmark v1 and latest H10035 coverage',
    'created_at': '2026-10-27', 'status': 'specification_and_inventory_complete',
    'no_gpu_launch': True, 'implementation': 'offline extraction; source versions in sources.json and source manifests',
    'deployment': 'no new server or booking; future runs choose live resources through Experiment Guard',
    'suite': 'suite_v1.json', 'cases': 'cases.csv', 'summary': '../BENCHMARK.md',
    'model_and_testbed': models, 'workload': workloads,
    'key_results': suite['coverage_as_of_20261027'], 'sources': 'sources.json',
    'note': 'A specification is not a multi-model runnable launcher; compatibility, topology and admission must be qualified per future run',
})
links = re.findall(r'\]\(([^)]+)\)', report)
for target in links:
    assert (Path(target) if Path(target).is_absolute() else SUMMARY / target).exists(), target
validation = json.loads((HERE/'validation.json').read_text())
validation['documentation_local_links_verified'] = len(links)
write_json('validation.json', validation)
print(json.dumps({'cases': len(cases), 'latest_measured': len(by_cell),
                  'pending': len(cases)-len(by_cell), 'source_files': len(SOURCES)}, ensure_ascii=False))
