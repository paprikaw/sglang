"""Independent CPU-only audit of saved four-way artifacts; never runs a model.

The source run and its frozen runtime are read-only. Derived reports are written
only to an explicitly separate output directory. Missing profiles stay missing.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

ARMS = ('ep8_mtp_on', 'ep8_mtp_off', 'pp2ep4_mtp_on', 'pp2ep4_mtp_off')


def read(path):
    return json.loads(Path(path).read_text())


def source(path):
    path = Path(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stat(values):
    values = sorted(float(x) for x in values)
    require(bool(values) and all(math.isfinite(x) for x in values), 'Missing/nonfinite sample')
    return dict(n=len(values), mean=statistics.fmean(values), median=statistics.median(values),
                p99=values[math.ceil(.99 * len(values)) - 1], p99_estimator='nearest_rank')


def optional_stat(values):
    values = [value for value in values if value is not None]
    return stat(values) if values else None


def request_rows(value):
    rows = value['requests']
    require(len(rows) == 8 and sorted(r['request_index'] for r in rows) == list(range(8)),
            'Not exactly eight unique requests')
    return sorted(rows, key=lambda row: row['request_index'])


def latency_check(value, mtp):
    rows = request_rows(value)
    wall = float(value['wall_s'])
    require(math.isfinite(wall) and wall > 0, 'Invalid cohort wall time')
    for row in rows:
        require(row['completion_tokens'] == 64 and len(row['output_ids']) == 64,
                'Output length differs from64')
        require(row['first_stream_chunk_tokens'] == 1, 'Buffered first-token boundary')
        ttft, tpot, end = (float(row[k]) for k in ('ttft_s', 'tpot_s', 'end_to_end_s'))
        require(all(math.isfinite(x) and x >= 0 for x in (ttft, tpot, end)) and end > ttft,
                'Invalid request timestamps')
        require(math.isclose(tpot, (end - ttft) / 63, rel_tol=1e-10, abs_tol=1e-12),
                'TPOT is not actual received output duration /63')
        events = row['token_events']
        require(events[0][0] == 1 and events[-1][0] == 64, 'Incomplete streaming witnesses')
        require(events[0][1] == ttft and events[-1][1] == end, 'Stream/metric boundary differs')
        require(all(a[0] < b[0] and a[1] <= b[1] for a, b in zip(events, events[1:])),
                'Nonmonotonic stream events')
        meta = row['meta']
        require(meta['prompt_tokens'] == 4097 and meta['cached_tokens'] == 4032,
                'Wrong prompt or cached length')
        require(meta['cached_tokens_details']['host'] == 4032
                and meta['cached_tokens_details']['device'] == 0, 'Wrong native cache source')
        require((meta.get('spec_verify_ct', 0) > 0) == mtp, 'Speculative mode metadata differs')
    require(wall + 1e-9 >= max(r['end_to_end_s'] for r in rows), 'Wall excludes a last token')
    return dict(output_throughput_tok_s=512 / wall, wall_s=wall,
                ttft_ms=stat(1000 * r['ttft_s'] for r in rows),
                tpot_ms=stat(1000 * r['tpot_s'] for r in rows),
                end_to_end_ms=stat(1000 * r['end_to_end_s'] for r in rows),
                spec_accept_length=optional_stat(r['meta'].get('spec_accept_length') for r in rows) if mtp else None,
                spec_verify_count=optional_stat(r['meta'].get('spec_verify_ct') for r in rows) if mtp else None)


def rank_set(rows, label):
    require(len(rows) == 8 and sorted(row['rank'] for row in rows) == list(range(8)),
            label + ' lacks exact eight ranks')


def validate_saved_cohort(path, cfg, arm, cohort):
    value = read(path)
    layout, mtp = ('ep8' if arm.startswith('ep8') else 'pp2ep4'), arm.endswith('_on')
    require((value['arm'], value['layout'], value['mtp'], value['cohort']) == (arm, layout, mtp, cohort),
            'Cohort identity differs')
    require(value['engine_commit'] == cfg['engine_commit'], 'Engine revision differs')
    expected = cfg['source_semantic_bank_ready']
    require(value['bank_identity_sha256'] == expected['bank_identity_sha256']
            and value['bank_manifest_sha256'] == expected['manifest_sha256'], 'Bank identity differs')
    require(value['prefix_tokens_per_request'] == 4096 and value['actual_cached_tokens_per_request'] == 4032
            and value['first_forward_new_tokens_per_request'] == 65, 'Workload differs')
    require(value['cuda_profiler_off_during_production'] is True
            and value['source_bank_import_timed'] is False and value['weights_loading_timed'] is False
            and value['hicache_h2d_timed'] is True and value['synthetic_kv'] is False,
            'Timing or semantic cache boundary differs')
    metrics = latency_check(value, mtp)
    cold = value['native_cpu_ready_gpu_empty']
    rank_set(cold, 'Cold proof')
    for receipt in cold:
        require(receipt['passed'] is True and receipt['native_hicache_quiescent'] is True,
                'Cold residency proof failed')
        require(receipt['profiler']['armed'] is False and receipt['profiler']['active'] is False,
                'Profiler was active during production')
        indices = list(range(8)) if layout == 'pp2ep4' else list(range((receipt['rank']//4)*4, (receipt['rank']//4+1)*4))
        require(sorted(r['request_index'] for r in receipt['matches']) == indices,
                'Cold request assignment differs')
        require(all(r['cpu_tokens'] == 4032 and r['gpu_tokens'] == 0 for r in receipt['matches']),
                'Cold cache was not CPU4032/GPU0')
    imports = value['bank_import_receipts']
    rank_set(imports, 'Import proof')
    for receipt in imports:
        bank = receipt['bank_import']
        require(receipt['passed'] is True and bank['passed'] is True
                and bank['bank_identity_sha256'] == expected['bank_identity_sha256']
                and bank['page_readback_verified'] is True and bank['gpu_tensor_reads'] == 0,
                'Semantic native-host import proof failed')
        require(all(row['page_readback_verified'] is True
                    and row['source_view_sha256'] == row['cpu_host_sha256']
                    for row in bank['imported_requests']), 'Import/readback bytes differ')
    from spec_proof import validate_receipts
    graph = validate_receipts(value['all_rank_after'], layout, cohort + '-', mtp=mtp)
    if layout == 'ep8':
        from dp_proof import validate
        scheduling = validate(value['all_rank_after'], cohort + '-')
    else:
        from pp_proof import validate
        scheduling = validate(value['all_rank_after'], mtp, cohort + '-')
    require(graph['passed'] is True and scheduling['passed'] is True, 'Replayed native proofs failed')
    require(value['actual_graph_proof']['passed'] is True and value['actual_scheduler_proof']['passed'] is True,
            'Stored proof gate failed')
    return dict(passed=True, metrics=metrics, source=source(path),
                bank_identity_sha256=value['bank_identity_sha256'], rows=request_rows(value),
                declared_performance_valid=value.get('performance_valid'),
                declared_profile_valid=value.get('profile_valid'))


def parity(left, right):
    rows = []
    for a, b in zip(left, right):
        first = next((i for i, (x, y) in enumerate(zip(a['output_ids'], b['output_ids'])) if x != y), None)
        rows.append(dict(request_index=a['request_index'], token_ids_equal=a['output_ids'] == b['output_ids'],
            text_equal=isinstance(a['output_text'], str) and a['output_text'] == b['output_text'],
            first_token_equal=a['output_ids'][0] == b['output_ids'][0],
            first_different_output_index=first,
            left_token_window=a['output_ids'][max(0, first-2):first+3] if first is not None else None,
            right_token_window=b['output_ids'][max(0, first-2):first+3] if first is not None else None))
    return dict(passed=all(row['token_ids_equal'] and row['text_equal'] for row in rows), requests=rows)


def profile_rows(arm_dir, arm):
    path = arm_dir / 'p4096_decode_profile_results.json'
    if not path.exists():
        return dict(valid=False, status='missing_not_estimated', rows=[])
    value = read(path)
    require(value.get('profile_valid') is True and value.get('all_rank_gpu_activity_audited') is True,
            'Profile has no strict GPU audit')
    summary_path = arm_dir / 'stage_activity_summary.json'
    summary = read(summary_path)
    require(len(summary['ranks']) == 8 and sorted(row['rank'] for row in summary['ranks']) == list(range(8)),
            'Profile summary lacks exact eight ranks')
    rows = []
    for row in summary['ranks']:
        window = row['steady_window']
        ms = window['per_local_microbatch_cycle_ms']
        cycles = len([s for s in row['segments'] if s['phase'].startswith('target_')]) - 1
        require(cycles == 11 and math.isclose(sum(ms.values()), window['duration_ms']/cycles, rel_tol=1e-8),
                'Local cycle partition is not exhaustive')
        rows.append(dict(rank=row['rank'], target_layers=92 if arm.startswith('ep8') else 46,
            local_microbatch_cycles=cycles, elapsed_cycle_ms=window['duration_ms']/cycles,
            attention_ms=ms['attention'], expert_routed_ms=ms['expert_routed'],
            expert_shared_ms=ms['expert_shared'], dense_ffn_ms=ms['dense_ffn'],
            communication_ms=sum(ms[k] for k in ('deepep_dispatch', 'deepep_combine', 'nccl_tp', 'pp_boundary')),
            deepep_dispatch_ms=ms['deepep_dispatch'], deepep_combine_ms=ms['deepep_combine'],
            tp_collective_ms=ms['nccl_tp'], pp_boundary_ms=ms['pp_boundary'],
            other_gpu_ms=sum(ms[k] for k in ('other_gpu', 'copy_or_set', 'marker')),
            unclassified_gpu_ms=ms['unclassified_gpu'], overlap_ms=ms['overlap'], gpu_idle_ms=ms['gpu_idle'],
            actual_output_normalization=row['actual_output_normalization'],
            per_committed_local_output_token_ms=window['per_committed_local_output_token_ms'],
            same_request_slot_start_cadence_ms=row['observed_same_request_slot_start_cadence_ms'],
            attribution_coverage=row.get('attribution_coverage')))
    return dict(valid=True, status='strict_saved_gpu_profile', rows=rows,
                sources=[source(path), source(summary_path)])


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_descriptor(item):
    require(source(item['path'])['sha256'] == item['sha256'],
            'SHA-bound evidence changed: ' + item['path'])


def bind_profile_source(production_run, profile_run, arm):
    """Authenticate a distinct profile without importing its timing as production."""
    production_run, profile_run = Path(production_run).resolve(), Path(profile_run).resolve()
    cfg = read(profile_run/'config.json')
    bound = cfg.get('profile_source', {})
    require(cfg.get('profile_only_followup') is True and cfg.get('new_production_cohorts') == 0,
            'Profile is not an explicit zero-production follow-up')
    require(bound.get('schema') == 'glm47-profile-followup-source-v1'
            and Path(bound['source_run']).resolve() == production_run,
            'Profile refers to another production run')
    require(bound.get('source_config_sha256') == source(production_run/'config.json')['sha256'],
            'Profile source config SHA differs')
    row = bound['arms'][arm]
    require(row['arm'] == arm and row['performance_valid'] is True
            and row['both_cohorts_validated'] is True and row['calibration_times_imported'] is False
            and row['new_production_measurement'] is False,
            'Profile source arm/validity boundary differs')
    expected_cohorts = [production_run/arm/name for name in
                        ('p4096_decode_results.json', 'p4096_decode_repeat_results.json')]
    require([Path(item['path']).resolve() for item in row['production_cohorts']] == expected_cohorts,
            'Profile production cohort bindings differ')
    for item in [row['arm_result'], *row['production_cohorts'], row['repeatability'],
                 *row['cleanups'], *row['calibrations']]:
        check_descriptor(item)
    for item in row['calibrations']:
        copied = Path(item['copied_path']).resolve()
        require(copied.parent == profile_run/arm and copied.name == Path(item['path']).name,
                'Calibration was copied into another arm or renamed')
        require(source(copied)['sha256'] == item['sha256'], 'Copied calibration bytes differ')
    # Compare actual generated serving arguments and kernel flags, not just a
    # stored self-declaration. Imports here are pure CPU configuration modules.
    runtime = profile_run.parents[1]/'runtime'
    profile_source = load_module('independent_followup_identity', runtime/'profile_source.py')
    new_common = load_module('independent_followup_common', runtime/'common.py')
    old_common = load_module('independent_production_common', production_run.parents[1]/'runtime/common.py')
    expected_identity = profile_source.execution_identity(read(production_run/'config.json'), arm, old_common)
    require(expected_identity == row['execution_identity']
            and profile_source.execution_identity(cfg, arm, new_common) == expected_identity,
            'Actual profile engine/serving/kernel/Bank identity differs')
    return cfg, row


def followup_profile_rows(production_run, profile_run, arm):
    """Re-run saved GPU evidence analysis; never launch CUDA or infer missing data."""
    base = Path(profile_run)/arm
    formal = base/'p4096_decode_profile_results.json'
    control_path = base/'arm_result.json'
    if not formal.exists() or not control_path.exists():
        if control_path.exists():
            controlled = read(control_path)
            if controlled.get('failure') or controlled.get('profile_failure'):
                return dict(valid=False, status='followup_failed_before_complete_profile_not_estimated', rows=[],
                            failure=controlled.get('failure') or controlled.get('profile_failure'),
                            profile_cleanup_verified=controlled.get('cleanup_verified'),
                            sources=[source(control_path)])
        return dict(valid=False, status='followup_missing_or_cleanup_pending_not_estimated', rows=[])
    cfg, bound = bind_profile_source(production_run, profile_run, arm)
    controlled = read(control_path)
    require(controlled.get('profile_only_followup') is True
            and controlled.get('source_performance_valid') is True
            and controlled.get('performance_valid') is None
            and controlled.get('profile_valid') is True and controlled.get('cleanup_verified') is True
            and controlled.get('failure') is None,
            'Follow-up profile/control/cleanup gate failed')
    value = read(formal)
    require(value.get('profile_only') is True
            and value.get('performance_metrics_not_for_production_comparison') is True
            and value.get('unprofiled_cohorts_are_source_run') is True
            and value.get('source_performance') == bound,
            'Follow-up profile crossed production/source boundaries')
    original = read(bound['production_cohorts'][0]['path'])
    require(parity(request_rows(original), request_rows(value))['passed'],
            'Formal profile token IDs/text differ from same-arm production')
    for field in ('native_cpu_ready_gpu_empty', 'all_rank_after', 'import_receipts'):
        rank_set(value[field], 'Follow-up ' + field)
        require(all(row.get('passed') is True for row in value[field]), 'Failed follow-up rank receipt')
    for receipt in value['native_cpu_ready_gpu_empty']:
        require(receipt['profiler']['armed'] is True and receipt['profiler']['active'] is True,
                'Follow-up collector not armed before requests')
        indices = (list(range(8)) if value['layout'] == 'pp2ep4' else
                   list(range((receipt['rank']//4)*4, (receipt['rank']//4+1)*4)))
        require(sorted(row['request_index'] for row in receipt['matches']) == indices,
                'Follow-up cold request assignment differs')
        require(all(row['gpu_tokens'] == 0 and row['cpu_tokens'] == 4032 for row in receipt['matches']),
                'Follow-up did not start CPU4032/GPU0')
    for receipt in value['import_receipts']:
        bank = receipt['bank_import']
        require(bank['passed'] is True and bank['page_readback_verified'] is True
                and bank['bank_identity_sha256'] == original['bank_identity_sha256']
                and all(row['source_view_sha256'] == row['cpu_host_sha256']
                        for row in bank['imported_requests']), 'Follow-up source/native-host bytes differ')
    from spec_proof import validate_receipts
    require(validate_receipts(value['all_rank_after'], value['layout'], 'profile-', mtp=value['mtp'])['passed'],
            'Follow-up native Graph proof failed')
    if value['layout'] == 'ep8':
        from dp_proof import validate
        scheduled = validate(value['all_rank_after'], 'profile-')
    else:
        from pp_proof import validate
        scheduled = validate(value['all_rank_after'], value['mtp'], 'profile-')
    require(scheduled['passed'], 'Follow-up native scheduler proof failed')
    for rank in range(8):
        receipt = read(base/f'activity_receipt_rank{rank}.json')
        require(receipt['rank'] == rank and receipt['arm'] == arm and receipt['passed'] is True
                and receipt['target_forward_count'] == 12 and receipt['startup_calibration_capture_count'] == 0
                and receipt['inner_phase_synchronizations'] == 0 and receipt['inner_phase_events'] == 0,
                'Follow-up activity was not the single bounded unmodified-forward sample')
        require(receipt['source_performance'] == bound
                and receipt['bootstrap']['collector_lifecycles'] == 1
                and receipt['bootstrap']['actual_model_forward_count'] == 0
                and receipt['bootstrap']['same_session_marker_proofs'] is True,
                'Follow-up marker/provenance session differs')
    analysis = load_module('independent_followup_gpu_analysis', Path(profile_run).parents[1]/'analysis/stage_activity.py')
    reanalyzed = analysis.analyze_arm(base, ranks=range(8))
    require(reanalyzed == read(base/'stage_activity_summary.json'), 'Saved GPU summary differs from independently replayed analysis')
    result = profile_rows(base, arm)
    result.update(status='strict_independently_replayed_followup_gpu_profile',
                  production_source_run=str(Path(production_run).resolve()), profile_run=str(Path(profile_run).resolve()),
                  source_binding=bound, profile_cleanup_verified=True,
                  sources=[*result['sources'], source(control_path), source(Path(profile_run)/'config.json')])
    return result


def audit(run, profile_run=None):
    run = Path(run).resolve()
    cfg = read(run/'config.json')
    sys.path.insert(0, str(run.parents[1]/'runtime'))
    expected = cfg['source_semantic_bank_ready']
    require(source(expected['path'])['sha256'] == expected['sha256'], 'Source READY metadata changed')
    arms, comparisons = {}, []
    for arm in ARMS:
        entry = dict(cohorts={}, errors=[], performance_valid=False, profile_valid=False,
                     cleanup_verified=None, status='not_measured')
        base = run/arm
        for cohort, name in (('timed', 'p4096_decode_results.json'), ('repeat', 'p4096_decode_repeat_results.json')):
            path = base/name
            if path.exists():
                try:
                    entry['cohorts'][cohort] = validate_saved_cohort(path, cfg, arm, cohort)
                except Exception as error:
                    entry['errors'].append(dict(stage=cohort, type=type(error).__name__, message=str(error)))
        if len(entry['cohorts']) == 2:
            entry['self_repeat_parity'] = parity(entry['cohorts']['timed']['rows'], entry['cohorts']['repeat']['rows'])
            entry['performance_cohorts_valid'] = entry['self_repeat_parity']['passed']
            entry['status'] = 'validated_cohorts_cleanup_pending' if entry['performance_cohorts_valid'] else 'repeatability_failed'
        elif entry['cohorts']:
            entry['status'] = 'single_validated_cohort_only'
        warm = base/'p4096_decode_warm_results.json'
        if warm.exists() and 'timed' in entry['cohorts']:
            try:
                value = read(warm)
                warm_rows = request_rows(value)
                latency_check(value, arm.endswith('_on'))
                rank_set(value['import_receipts'], 'Warm import proof')
                require(all(row['bank_import']['bank_identity_sha256'] == expected['bank_identity_sha256']
                            and row['bank_import']['page_readback_verified'] is True
                            for row in value['import_receipts']), 'Warm source Bank differs')
                entry['warm_vs_timed_parity'] = parity(warm_rows, entry['cohorts']['timed']['rows'])
                entry['warm_source'] = source(warm)
            except Exception as error:
                entry['errors'].append(dict(stage='warm_vs_timed', type=type(error).__name__, message=str(error)))
        control = base/'arm_result.json'
        if control.exists():
            owned = read(control)
            entry.update(controller_passed=owned['passed'], controller_failure=owned.get('failure'),
                         cleanup_verified=owned['cleanup_verified'], profile_failure=owned.get('profile_failure'),
                         control_source=source(control))
            entry['performance_valid'] = (entry.get('performance_cohorts_valid') is True
                and owned.get('performance_valid') is True and owned['cleanup_verified'] is True)
            if entry['performance_valid']:
                entry['status'] = 'performance_valid_profile_pending_or_failed'
        try:
            entry['profile'] = profile_rows(base, arm)
            entry['profile_valid'] = entry['profile']['valid']
            if entry['performance_valid'] and entry['profile_valid']:
                entry['status'] = 'performance_and_profile_valid'
        except Exception as error:
            entry['profile'] = dict(valid=False, status='audit_failed_not_estimated', rows=[])
            entry['errors'].append(dict(stage='profile', type=type(error).__name__, message=str(error)))
        arms[arm] = entry
    if profile_run is not None:
        for arm, entry in arms.items():
            try:
                entry['profile'] = followup_profile_rows(run, profile_run, arm)
                entry['profile_valid'] = entry['profile']['valid']
                if entry['performance_valid'] and entry['profile_valid']:
                    entry['status'] = 'performance_and_independent_followup_profile_valid'
            except Exception as error:
                entry['profile'] = dict(valid=False, status='followup_audit_failed_not_estimated', rows=[])
                entry['profile_valid'] = False
                entry['errors'].append(dict(stage='followup_profile', type=type(error).__name__, message=str(error)))
    available = [arm for arm in ARMS if 'timed' in arms[arm]['cohorts']]
    for left, right in itertools.combinations(available, 2):
        a, b = arms[left]['cohorts']['timed'], arms[right]['cohorts']['timed']
        comparisons.append(dict(left=left, right=right,
            both_performance_valid=arms[left]['performance_valid'] and arms[right]['performance_valid'],
            same_semantic_bank_identity=a['bank_identity_sha256'] == b['bank_identity_sha256'],
            exact_output_parity=parity(a['rows'], b['rows']),
            right_over_left_throughput=b['metrics']['output_throughput_tok_s']/a['metrics']['output_throughput_tok_s']))
    for entry in arms.values():
        for cohort in entry['cohorts'].values():
            cohort.pop('rows')
    return dict(schema='independent-fourway-cpu-audit-v1', run=str(run),
                separate_profile_run=str(Path(profile_run).resolve()) if profile_run is not None else None,
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                config_source=source(run/'config.json'), arms=arms, comparisons=comparisons,
                caveats=[
                    'Production: B8 P4096 raw/4032 native CPU cached, Q1->64. Source import excluded, native HiCache H2D included.',
                    'TPOT=(last received-first received)/63, not verify time or profile cycle.',
                    'Profile ms are exclusive per-device wall partition per local B4 target cycle, including intervening draft/transport/idle.',
                    'EP8 rank executes92 target layers; PP rank executes46. PP cycles alternate two B4 request slots and are not request TPOT.',
                    'MTP Verify advances a measured variable output count; per-committed-local-output ms divides all B4 outputs, not one request.',
                    'Unknown GPU activity and overlap remain explicit; idle is not proof of a pipeline bubble.',
                    'P99 over only eight requests equals the sample maximum.'])


def markdown(result):
    lines = ['# Independent four-way Decode audit', '',
             '| Configuration | Cohort | Validity | Output tok/s | TTFT mean/median/P99 ms | TPOT mean/median/P99 ms | E2E mean/median/P99 ms |',
             '|---|---|---|---:|---:|---:|---:|']
    def triplet(value):
        return '/'.join(f'{value[key]:.3f}' for key in ('mean', 'median', 'p99'))
    for arm, entry in result['arms'].items():
        for name, cohort in entry['cohorts'].items():
            m = cohort['metrics']
            lines.append(f"| {arm} | {name} | {entry['status']} | {m['output_throughput_tok_s']:.3f} | {triplet(m['ttft_ms'])} | {triplet(m['tpot_ms'])} | {triplet(m['end_to_end_ms'])} |")
        if not entry['cohorts']:
            lines.append(f"| {arm} | — | {entry['status']} | — | — | — | — |")
    lines += ['', '## Separate profile: absolute ms per GPU / local B4 target cycle', '',
              '| Configuration | Rank | Target layers | A | Routed E | Shared E | Dense | Comm | Other GPU | Unknown | Overlap | Idle | Total cycle |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, entry in result['arms'].items():
        if not entry['profile']['valid']:
            lines.append(f"| {arm} | Missing/invalid; not estimated | — | — | — | — | — | — | — | — | — | — | — |")
        for row in entry['profile']['rows']:
            fields = ['attention_ms', 'expert_routed_ms', 'expert_shared_ms', 'dense_ffn_ms', 'communication_ms',
                      'other_gpu_ms', 'unclassified_gpu_ms', 'overlap_ms', 'gpu_idle_ms', 'elapsed_cycle_ms']
            values = ' | '.join(f'{row[key]:.3f}' for key in fields)
            lines.append(f"| {arm} | {row['rank']} | {row['target_layers']} | {values} |")
    lines += ['', '## Boundaries and caveats', ''] + ['- '+s for s in result['caveats']]
    lines += ['', '## Failures and cleanup', '']
    for arm, entry in result['arms'].items():
        lines.append(f"- {arm}: production cleanup={entry['cleanup_verified']}; profile={entry['profile']['status']}; errors={json.dumps(entry['errors'])}; original profile failure={json.dumps(entry.get('profile_failure'))}; follow-up failure={json.dumps(entry['profile'].get('failure'))}")
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--profile-run', type=Path, help='Optional separate SHA-bound profile-only follow-up; no new production metrics')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.resolve().is_relative_to(args.run.resolve().parents[1]),
            'Derived report must be outside frozen experiment root')
    if args.profile_run is not None:
        require(not args.output_dir.resolve().is_relative_to(args.profile_run.resolve().parents[1]),
                'Derived report must be outside frozen profile experiment root')
    result = audit(args.run, args.profile_run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'independent_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (args.output_dir/'independent_audit.md').write_text(markdown(result))
    print(json.dumps({arm: dict(status=row['status'], errors=row['errors']) for arm, row in result['arms'].items()}))


if __name__ == '__main__':
    main()
