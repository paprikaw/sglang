"""Read saved four-way results; never launch a model or rewrite source artifacts."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import itertools
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ANALYSIS = EXPERIMENTS / 'glm47_semantic_bank_fourway_20261023_v7' / 'analysis'
DEFAULT_PERFORMANCE = EXPERIMENTS / 'glm47_semantic_bank_fourway_20261023_v5' / 'runs' / 'fourway_reuse_r3'
COHORTS = {'timed': 'p4096_decode_results.json', 'repeat': 'p4096_decode_repeat_results.json'}
METRICS = ('ttft_ms', 'tpot_ms', 'end_to_end_ms')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def two_cohort(values):
    """Arithmetic mean of two cohort estimates, not an independent-run CI."""
    if len(values) != 2:
        return None
    values = [float(value) for value in values]
    mean = statistics.fmean(values)
    low, high = min(values), max(values)
    return dict(cohort_values=values, mean=mean, minimum=low, maximum=high,
                half_range=(high - low) / 2,
                full_range_percent_of_mean=100 * (high - low) / mean if mean else None,
                estimator='Arithmetic mean of two unprofiled cohort estimates',
                uncertainty='Two cohorts in one serving process; half-range is not a confidence interval')


def acceptance(raw, mtp):
    if not mtp:
        return None
    rows = raw['requests']
    verifications = sum(row['meta']['spec_verify_ct'] for row in rows)
    correct = sum(row['meta']['spec_num_correct_drafts'] for row in rows)
    proposed = sum(row['meta']['spec_num_proposed_drafts'] for row in rows)
    completion = sum(row['meta']['completion_tokens'] for row in rows)
    if verifications <= 0 or proposed != 2 * verifications:
        raise ValueError('Expected native three-token MTP verify with two proposed drafts')
    for row in rows:
        meta = row['meta']
        if meta['spec_num_proposed_drafts'] != 2 * meta['spec_verify_ct']:
            raise ValueError('Per-request proposed-draft denominator mismatch')
        if meta['spec_accept_rate'] != meta['spec_num_correct_drafts'] / meta['spec_num_proposed_drafts']:
            raise ValueError('Native strict acceptance rate formula mismatch')
        if meta['spec_accept_length'] != meta['completion_tokens'] / meta['spec_verify_ct']:
            raise ValueError('Native acceptance-length formula mismatch')
    return dict(verify_count=verifications, correct_draft_count=correct,
                proposed_draft_count=proposed, completion_tokens=completion,
                proposed_drafts_per_verify=2,
                pooled_draft_accept_rate=correct / proposed,
                pooled_completion_tokens_per_verify=completion / verifications,
                request_mean_accept_rate=statistics.fmean(row['meta']['spec_accept_rate'] for row in rows),
                request_mean_accept_length=statistics.fmean(row['meta']['spec_accept_length'] for row in rows),
                request_mean_verify_count=verifications / len(rows),
                formulas=dict(rate='sum(correct drafts) / sum(verify_ct * (draft_tokens - 1)); draft_tokens=3',
                              length='sum(completion_tokens) / sum(verify_ct)',
                              request_mean='Arithmetic mean of each request native rate/length'),
                caveat='Strict draft rate excludes bonus tokens. Completion length includes emitted tokens and the 64-token cap; final verified runs can extend beyond the emitted cap. Do not infer strict rate from acceptance length.')


def topology(arm):
    pp = arm.startswith('pp2ep4')
    mtp = arm.endswith('_on')
    return dict(attention_tp=4, attention_dp=1 if pp else 2,
                expert_ep=4 if pp else 8, pipeline_stages=2 if pp else 1,
                target_layers_per_rank_group={'ranks0_3': 46 if pp else 92,
                                             'ranks4_7': 46 if pp else 92},
                draft_layers_per_rank_group={'ranks0_3': 0 if pp or not mtp else 1,
                                            'ranks4_7': 1 if mtp else 0},
                local_graph_requests=4, target_tokens_per_request_per_graph=3 if mtp else 1,
                pp_request_slots=2 if pp else None,
                cycle_definition='Per-device local B4 cycle: EP full 92-layer target; PP one 46-layer stage, with MTP draft only on the tail stage. A PP slot alternates request groups; this is not request TPOT.')


def profile_overhead(profile, source_headline):
    """Descriptive profiler slowdown; never correct production by this factor."""
    rows = profile['requests']
    wall = float(profile['wall_s'])
    values = dict(wall_s=wall,
                  ttft_ms=1000 * statistics.fmean(row['ttft_s'] for row in rows),
                  tpot_ms=1000 * statistics.fmean(row['tpot_s'] for row in rows),
                  end_to_end_ms=1000 * statistics.fmean(row['end_to_end_s'] for row in rows))
    if len(rows) != 8 or wall <= 0:
        raise ValueError('Incomplete strict profile client cohort')
    return dict(profile_cohort_values=values,
                source_unprofiled_two_cohort_means={key: source_headline[key]['mean'] for key in values},
                profile_over_unprofiled_factor={key: value / source_headline[key]['mean'] for key, value in values.items()},
                diagnostic_only=True,
                caveat='Separate profiled client cohort may have inflated CPU/GPU idle and cadence. Do not use these factors to adjust production or partition its request TPOT.')


def profile_data(profile_run, performance_run, source_headlines=None):
    pending = dict(requested=profile_run is not None, run=str(profile_run.resolve()) if profile_run else None,
                   valid_profile_points=0, all_profiles_valid=False, arms={}, data=None)
    if profile_run is None:
        pending['status'] = 'not_requested'
        return pending
    try:
        helper = load_module('fourway_existing_profile_join', ANALYSIS / 'summarize_profile_followup.py')
        joined = helper.summarize(profile_run)
        if Path(joined['original_production_run']).resolve() != performance_run.resolve():
            raise ValueError('Profile follow-up is bound to a different production run')
        pending.update(status='complete' if joined['passed'] else 'pending_or_failed',
                       valid_profile_points=joined['valid_profile_points'],
                       all_profiles_valid=joined['passed'], source_binding_validated=True,
                       evidence=joined['evidence'], validator_source=str((ANALYSIS / 'summarize_profile_followup.py').resolve()))
        for arm, value in joined['profiles'].items():
            # Invalid/partial traces never become numerical report data.
            pending['arms'][arm] = dict(valid=value['valid'], status=value['status'],
                                       missing=value.get('missing', []), failure=value.get('failure'),
                                       evidence=value.get('evidence', []),
                                       source_output_parity=value.get('source_output_parity'),
                                       activity=value.get('activity') if value['valid'] else None,
                                       profiler_slowdown_diagnostic=None,
                                       topology=topology(arm))
            if value['valid'] and source_headlines:
                profile = json.loads((Path(profile_run) / arm / 'p4096_decode_profile_results.json').read_text())
                pending['arms'][arm]['profiler_slowdown_diagnostic'] = profile_overhead(profile, source_headlines[arm])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        pending.update(status='pending_or_invalid', source_binding_validated=False,
                       valid_profile_points=0, all_profiles_valid=False, arms={},
                       error=f'{type(exc).__name__}: {exc}')
    return pending


def build(performance_run, profile_run=None):
    performance_run = Path(performance_run).resolve()
    helper = load_module('fourway_existing_performance_summary', ANALYSIS / 'summarize.py')
    original = helper.summarize(performance_run)
    raw, headlines, native_acceptance, rows, evidence = {}, {}, {}, [], []
    source_validity = {}
    for arm in helper.ARMS:
        entry = original['arms'][arm]
        if entry['status'] != 'measured_repeatable':
            raise ValueError(f'{arm}: expected two completed repeatable performance cohorts')
        raw[arm], native_acceptance[arm] = {}, {}
        owner = helper.read(performance_run / arm / 'arm_result.json')
        source_validity[arm] = dict(performance_valid=owner.get('performance_valid'),
                                   old_profile_valid=owner.get('profile_valid'),
                                   old_combined_arm_passed=owner.get('passed'),
                                   cleanup_verified=owner.get('cleanup_verified'), failure=owner.get('failure'))
        # A failed old diagnostic does not invalidate explicitly qualified production.
        if owner.get('performance_valid') is not True or owner.get('cleanup_verified') is not True or owner.get('failure') is not None:
            raise ValueError(f'{arm}: production or owned cleanup is not qualified')
        for cohort, filename in COHORTS.items():
            path = performance_run / arm / filename
            value = helper.read(path)
            if (value.get('performance_valid') is not True
                    or value['actual_graph_proof'].get('passed') is not True
                    or value['actual_scheduler_proof'].get('passed') is not True):
                raise ValueError(f'{arm}/{cohort}: actual performance/Graph/schedule proof missing')
            raw[arm][cohort] = value
            native_acceptance[arm][cohort] = acceptance(value, value['mtp'])
            data = entry['cohorts'][cohort]
            csv_row = dict(arm=arm, cohort=cohort, output_tok_s=data['output_throughput_tok_s'],
                           wall_s=data['wall_s'], source_path=str(path), source_sha256=data['source']['sha256'])
            for metric in METRICS:
                for statistic in ('mean', 'median', 'p99'):
                    csv_row[metric + '_' + statistic] = data[metric][statistic]
            native = native_acceptance[arm][cohort]
            for key in ('verify_count', 'correct_draft_count', 'proposed_draft_count',
                        'pooled_draft_accept_rate', 'pooled_completion_tokens_per_verify',
                        'request_mean_accept_rate', 'request_mean_accept_length'):
                csv_row[key] = native[key] if native else None
            rows.append(csv_row)
            evidence.append(helper.provenance(path))
        selected = [entry['cohorts'][key] for key in COHORTS]
        headlines[arm] = dict(output_tok_s=two_cohort([x['output_throughput_tok_s'] for x in selected]),
                              wall_s=two_cohort([x['wall_s'] for x in selected]),
                              topology=topology(arm))
        for metric in METRICS:
            headlines[arm][metric] = two_cohort([x[metric]['mean'] for x in selected])
    comparisons = []
    for left, right in itertools.combinations(helper.ARMS, 2):
        item = dict(left=left, right=right, cohorts={})
        for cohort in COHORTS:
            a, b = raw[left][cohort], raw[right][cohort]
            parity = helper.output_parity(a['requests'], b['requests'])
            aa, bb = helper.index_requests(a['requests']), helper.index_requests(b['requests'])
            parity.update(full_outputs_equal_count=sum(x['token_ids_equal'] and x['text_equal'] for x in parity['requests']),
                          first_output_token_equal_count=sum(aa[i]['output_ids'][0] == bb[i]['output_ids'][0] for i in range(8)))
            item['cohorts'][cohort] = parity
        item['same_semantic_bank'] = raw[left]['timed']['bank_identity_sha256'] == raw[right]['timed']['bank_identity_sha256']
        comparisons.append(item)
    deltas = []
    for new, old in (('pp2ep4_mtp_on', 'ep8_mtp_on'), ('pp2ep4_mtp_off', 'ep8_mtp_off'),
                     ('ep8_mtp_on', 'ep8_mtp_off'), ('pp2ep4_mtp_on', 'pp2ep4_mtp_off')):
        deltas.append(dict(new=new, reference=old, percent_change={key:100*(headlines[new][key]['mean']/headlines[old][key]['mean']-1)
                                                                for key in ('output_tok_s', *METRICS)}))
    result = dict(schema_version=1, generated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  source_performance_run=str(performance_run), completed_valid_performance_points=4,
                  headline_estimator='Arithmetic mean of timed and repeat cohort means; throughput is the mean of the two measured cohort rates.',
                  source_validity=source_validity, performance=original, headlines=headlines,
                  native_acceptance=native_acceptance, cross_arm_output_parity=comparisons,
                  headline_relative_changes=deltas, profiles=profile_data(Path(profile_run) if profile_run else None, performance_run, headlines),
                  evidence=evidence, analyzer_source=helper.provenance(Path(__file__)),
                  existing_validator_source=helper.provenance(ANALYSIS / 'summarize.py'),
                  caveats=['Same semantic inputs, but full output histories differ between arms. These are configuration measurements, not an equal-token-history PP/communication overhead experiment.',
                           'Within-arm exact repeat parity is checked. Different target logits or actual GPU KV at first divergence were not captured here; numerical or state root cause remains unproven.',
                           'All production metrics exclude profiler, weight loading and source-bank import; include native CPU HiCache H2D and initial 65-token tail forward.',
                           'P99 is the maximum of eight request samples, not a robust tail estimate. Two same-process cohorts are not independent run confidence intervals.',
                           'Profile activity is diagnostic only. EP target covers 92 layers; each PP stage covers46 target layers and only tail runs the MTP draft. Never sum stage/rank maxima into TPOT.',
                           'Per-local-B4-cycle CUDA work is not per-request TPOT. PP alternates two request slots; output normalization needs actual committed-token receipts.',
                           'Profiler can inflate CPU/GPU idle and scheduling cadence. Traced GPU idle is not proof of pipeline bubble; component fractions are not a direct partition of unprofiled request TPOT.',
                           'Missing, partial or failed profiles remain null/pending; unknown activity is never redistributed into Attention, Expert or communication.'])
    return result, rows


def markdown(result):
    lines = ['# GLM-4.7-AWQ：EP8 与 PP2×EP4，MTP 开／关', '',
             '065／167，8×A100。B8，P4096，原生 CPU 缓存4032／GPU0，Q1→64；首次 forward 为65 token。计时包含原生 HiCache→GPU，排除权重加载、源 Bank 导入和 profiler。', '',
             '## 两轮平均', '', '下表为两轮均值 ± 半极差；不是置信区间。', '',
             '| 配置 | 输出 tok/s | TTFT ms | TPOT ms | E2E ms |', '|---|---:|---:|---:|---:|']
    def spread(value):
        return f"{value['mean']:.3f} ± {value['half_range']:.3f}"
    def triple(value):
        return ' / '.join(f'{value[key]:.3f}' for key in ('mean', 'median', 'p99'))
    for arm, value in result['headlines'].items():
        lines.append('| ' + arm + ' | ' + ' | '.join(spread(value[key]) for key in ('output_tok_s', *METRICS)) + ' |')
    lines += ['', '## 每轮完整延迟', '', '| 配置 | Cohort | tok/s | TTFT mean / median / P99 ms | TPOT mean / median / P99 ms | E2E mean / median / P99 ms |', '|---|---|---:|---:|---:|---:|']
    for arm, entry in result['performance']['arms'].items():
        for cohort, value in entry['cohorts'].items():
            lines.append(f"| {arm} | {cohort} | {value['output_throughput_tok_s']:.3f} | " + ' | '.join(triple(value[key]) for key in METRICS) + ' |')
    lines += ['', '## MTP 原生接受统计', '', '| 配置／轮 | Verify 总数 | 正确 draft／提议 draft | 汇总接受率 | 汇总输出／Verify | 请求平均接受率 | 请求平均 length |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm, cohorts in result['native_acceptance'].items():
        for cohort, value in cohorts.items():
            if value:
                lines.append(f"| {arm} / {cohort} | {value['verify_count']} | {value['correct_draft_count']} / {value['proposed_draft_count']} | {100*value['pooled_draft_accept_rate']:.3f}% | {value['pooled_completion_tokens_per_verify']:.4f} | {100*value['request_mean_accept_rate']:.3f}% | {value['request_mean_accept_length']:.4f} |")
    lines += ['', '严格 draft 接受率分母为 Verify×2，不含 bonus；length 使用 completion_tokens／Verify。汇总比值与请求比值平均不同，末轮64-token截断也不能被忽略。', '',
              '## 完整跨臂输出校验', '', '每臂自身两轮8/8完全一致。下表跨臂按 request_index 匹配，未假设模式／拓扑等价。', '',
              '| 两臂 | Cohort | 首 token 相同 | 完整64-token ID与文本相同 | 首分歧位置（从0开始；—为相同） |', '|---|---|---:|---:|---|']
    for pair in result['cross_arm_output_parity']:
        for cohort, value in pair['cohorts'].items():
            positions = ', '.join('—' if x['first_different_output_index'] is None else str(x['first_different_output_index']) for x in value['requests'])
            lines.append(f"| {pair['left']} / {pair['right']} | {cohort} | {value['first_output_token_equal_count']}/8 | {value['full_outputs_equal_count']}/8 | {positions} |")
    profiles = result['profiles']
    lines += ['', '## 独立 CUDA profile', '', f"严格有效 profile：{profiles['valid_profile_points']}/4；状态：`{profiles['status']}`。未通过完整 validator 的数值不展示。", '',
              'EP8 每 rank 覆盖92层 target；PP 每 stage 为46层，MTP draft只在 tail。下列单位是本地 B4 微批次周期，不能解释为请求 TPOT，也不对不同 stage／rank 的最大值求和。Profiler可能增加CPU／GPU空闲和调度间隔；GPU idle不能直接称为pipeline bubble。', '']
    for arm, value in profiles['arms'].items():
        lines.append(f"- `{arm}`：{value['status']}" + ('' if value['valid'] else '；数值=null。'))
        if value['valid']:
            slowdown = value.get('profiler_slowdown_diagnostic')
            if slowdown:
                lines += ['', f"### {arm} · Profiler放大诊断", '', '| 指标 | Profile cohort | 原未profile两轮均值 | 倍数 |', '|---|---:|---:|---:|']
                for key in ('wall_s', 'ttft_ms', 'tpot_ms', 'end_to_end_ms'):
                    lines.append(f"| {key} | {slowdown['profile_cohort_values'][key]:.4f} | {slowdown['source_unprofiled_two_cohort_means'][key]:.4f} | {slowdown['profile_over_unprofiled_factor'][key]:.3f}× |")
                lines += ['', '此表只诊断 profiler 扰动；不用于修正生产延迟或把profile占比直接分摊到未profile TPOT。']
            for group in value['activity'].get('rank_groups', []):
                lines += ['', f"### {arm} · {group['group']}", '', '| 类别 | 每本地B4周期 mean / min / max ms |', '|---|---:|']
                for phase, data in group.get('per_rank_cycle_ms', {}).items():
                    lines.append(f"| {phase} | {data['mean']:.4f} / {data['min']:.4f} / {data['max']:.4f} |")
    lines += ['', '## 解释边界', ''] + ['- ' + text for text in result['caveats']]
    lines += ['', '## 来源', '', f"- 性能源：`{result['source_performance_run']}`", f"- Profile源：`{profiles['run']}`", '- `summary.json` 保留来源路径、SHA256、每轮统计、完整校验与已有 validator 的结果；`performance.csv` 为八个未 profile cohort。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--performance-run', type=Path, default=DEFAULT_PERFORMANCE)
    parser.add_argument('--profile-run', type=Path)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'current')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    for source in (args.performance_run, args.profile_run):
        if source is not None and (output == source.resolve() or source.resolve() in output.parents):
            raise ValueError('Derived reports must remain outside source runs')
    result, rows = build(args.performance_run, args.profile_run)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    with (output / 'performance.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / 'report.md').write_text(markdown(result))
    print(json.dumps(dict(output_dir=str(output), performance_points=4,
                          strict_valid_profile_points=result['profiles']['valid_profile_points'])))


if __name__ == '__main__':
    main()
