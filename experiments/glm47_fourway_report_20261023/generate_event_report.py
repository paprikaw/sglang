"""Read-only production + external CUDA Event report; never launches a GPU.

The old generator/current report is intentionally untouched. Missing or failed
diagnostics stay null. Event stream spans are not CUPTI busy time or TPOT parts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
V8 = EXPERIMENTS / 'glm47_semantic_bank_fourway_20261023_v8'
DEFAULT_PERFORMANCE = EXPERIMENTS / 'glm47_semantic_bank_fourway_20261023_v5/runs/fourway_reuse_r3'
DEFAULT_QUALIFICATION = EXPERIMENTS / 'deepep_graph_external_events_20261023/runs/external_events_r1'
MEASUREMENT = 'external_cuda_event_stream_spans'
ARMS = ('ep8_mtp_on', 'ep8_mtp_off', 'pp2ep4_mtp_on', 'pp2ep4_mtp_off')
COMPONENTS = ('attention', 'expert_routed', 'expert_shared', 'dense_ffn',
              'deepep_dispatch', 'deepep_combine', 'nccl_tp', 'other_graph', 'overlap')
NOTES = [
    'Measured per-device, per-actual-Graph external CUDA Event stream spans in milliseconds; includes stream waits and event nodes, not CUPTI kernel busy time.',
    'The one completion/read after every actual Graph changes diagnostic cadence. Instrumented wall/TPOT are shown separately from unprofiled production, never used to correct it.',
    'Target, draft and actual phases stay separate. A draft target_ordinal is the latest observed target, NOT proof of a complete same-request MTP cycle.',
    'Do not add target and draft means, ranks or PP stage maxima into request TPOT. PP boundary API stream spans are separate and do not measure asynchronous send/NIC completion.',
    'Attention covers the wrapped attention forward, including QKV/O projections, QK normalization, RoPE and KV store, but excludes the outer input RMSNorm. Expert totals include routed, shared and dense FFN scopes.',
    'The historical schema key nccl_tp is the wrapped GroupCoordinator collective API stream span. Raw data omit group ranks, so it cannot be asserted to be only Attention TP4 or split further into TP/DP groups.',
    'Other is uncovered Graph stream time, not GPU idle. Cross-category overlap is retained; nested NCCL collective API spans are removed from Attention by the audited interval partition.',
    'Twelve targets are local scheduler forwards, not twelve per request: each EP Attention DP group contributes twelve B4 targets; PP has two B4 request slots, six sampled targets per slot.',
    'Dispatch+Combine per MoE layer uses 89 EP target MoE layers or PP stage counts 43/46 (92 target layers, first three dense). This compares actual deployment spans; peer count, routing domain and request-group composition also change, so it is not a pure link-only causal comparison.',
    'Diagnostic committed tokens per actual target use bounded local event receipts; they are not the full 64-output production native accept_rate/accept_length denominators. Keep these populations separate.',
    'Missing, partial, mismatched or failed evidence produces null diagnostic data, never estimated component times.',
]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def evidence(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def bound(item):
    actual = evidence(item['path'])
    require(actual['sha256'] == item['sha256'], 'SHA-bound source changed: ' + item['path'])
    return actual


def all_ranks(rows, description):
    require(len(rows) == 8 and sorted(row.get('rank', -1) for row in rows) == list(range(8)),
            'All eight unique ranks required: ' + description)
    require(all(row.get('passed') is True for row in rows), 'Failed rank receipt: ' + description)


def owned_cleanup(rows, expected_nodes):
    require(len(rows) == 2 and {row.get('node') for row in rows} == set(expected_nodes),
            'Missing both owned node cleanups')
    require(all((row.get('owned_children_exited') is True or row.get('owned_processes_exited') is True)
                and row.get('gpu_cleanup', {}).get('passed') is True
                and row.get('failure') is None for row in rows), 'Owned GPU cleanup failed')


@contextmanager
def runtime_imports(runtime):
    """Existing validators are CPU-only; isolate their sibling module imports."""
    names = ('deterministic_proof', 'humming_receipt')
    saved = {name: sys.modules.pop(name, None) for name in names}
    sys.path.insert(0, str(runtime))
    try:
        yield
    finally:
        sys.path.remove(str(runtime))
        for name, value in saved.items():
            sys.modules.pop(name, None)
            if value is not None:
                sys.modules[name] = value


def qualification(run=DEFAULT_QUALIFICATION):
    result = dict(valid=False, status='pending_or_invalid', data=None, evidence=[])
    try:
        run = Path(run)
        raw, cleanup = read(run / 'task/results.json'), read(run / 'task/cleanup.json')
        mode = raw['modes']['external_events_after_capture']
        rows = mode['ranks']
        all_ranks(rows, 'tiny external-event qualification')
        require(raw.get('passed') is True and mode.get('event_valid') is True
                and mode.get('payload_valid') is True, 'Tiny event/payload qualification failed')
        require(all(row.get('event_valid') is True and row.get('measured_graph_replays') == 3
                    and row.get('cupti_api_calls') == 0 and row.get('freshness', {}).get('passed') is True
                    and row.get('freshness', {}).get('not_capture_constant') is True
                    and row.get('event_flags') == dict(enable_timing=True, external=True)
                    and len(row.get('samples', [])) == 3
                    and all(sample.get('payload_valid') is True for sample in row['samples'])
                    for row in rows), 'Tiny three-replay freshness/payload proof missing')
        owned_cleanup(mode['node_exits'], [row['node'] for row in mode['node_exits']])
        require(all(cleanup.get(key) is True for key in
                    ('passed', 'gpu_idle', 'gpu_resources_released', 'model_processes_exited')),
                'Tiny task cleanup missing')
        result.update(valid=True, status='qualified',
            data=dict(ranks=list(range(8)), replays_per_rank=3, cupti_enabled=False,
                      payload_valid=True, cleanup_verified=True,
                      scope='Mechanism qualification only; not a full-model result'),
            evidence=[evidence(run / name) for name in ('task/results.json', 'task/cleanup.json', 'config.json', 'dashboard_spec.json')])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def validate_binding(run, performance_run):
    cfg, original = read(run / 'config.json'), read(performance_run / 'config.json')
    source = cfg['profile_source']
    require(cfg.get('profile_only_followup') is True and cfg.get('new_production_cohorts') == 0,
            'Expected profile-only follow-up with zero production cohorts')
    require(cfg.get('profile_measurement') == MEASUREMENT and cfg.get('external_cuda_events') is True
            and cfg.get('startup_cupti_enabled') is False and cfg.get('startup_calibration_enabled') is False,
            'Expected external events without CUPTI/startup calibration')
    require(Path(source['source_run']).resolve() == performance_run.resolve(), 'Different source production run')
    require(sha(performance_run / 'config.json') == source['source_config_sha256'], 'Source configuration SHA changed')
    bank = cfg['source_semantic_bank_ready']
    require(bank == original['source_semantic_bank_ready']
            and cfg['bank_model_identity'] == original['bank_model_identity'], 'Different semantic Bank identity')
    records = [bound(bank)]
    for root in (performance_run, run):
        spec = read(root / 'dashboard_spec.json')
        require(bool(spec.get('frozen_inputs')), 'Missing frozen source SHA manifest')
        records.extend(bound(item) for item in spec['frozen_inputs'])
    common = load('event_report_profile_common', V8 / 'runtime/common.py')
    src = load('event_report_profile_source', V8 / 'runtime/profile_source.py')
    old_common, _ = src.source_common(performance_run)
    for arm in ARMS:
        row = source['arms'][arm]
        require(row.get('performance_valid') is True and row.get('both_cohorts_validated') is True,
                'Source cohorts not qualified: ' + arm)
        require(src.execution_identity(cfg, arm, common) == row['execution_identity']
                == src.execution_identity(original, arm, old_common), 'Execution source/Graph/kernel identity changed: ' + arm)
        for item in (row['arm_result'], *row['production_cohorts'], row['repeatability'], *row['cleanups']):
            records.append(bound(item))
    q = cfg['external_event_qualification']
    records.extend(bound(item) for item in (q, q['config'], q['frozen_spec']))
    probe = qualification(Path(q['path']).parent.parent)
    require(probe['valid'], 'External event qualification invalid')
    return cfg, dict(passed=True, source_config_sha256=source['source_config_sha256'],
        bank_identity_sha256=bank['bank_identity_sha256'], frozen_source_hashes_verified=True,
        execution_identities_verified=True, external_event_qualification=probe,
        evidence=records)


def validate_cache(receipts, cfg, arm):
    all_ranks(receipts, 'CPU-cold semantic Bank')
    bank = cfg['source_semantic_bank_ready']
    for row in receipts:
        imp = row.get('bank_import', {})
        require(row.get('arm') == arm and imp.get('passed') is True
                and imp.get('bank_identity_sha256') == bank['bank_identity_sha256']
                and imp.get('manifest_sha256') == bank['manifest_sha256']
                and imp.get('semantic_kv') is True and imp.get('synthetic_kv') is False,
                'Rank Bank/import identity mismatch')
        matches = row.get('matches', [])
        expected = 8 if cfg['arms'][arm]['layout'] == 'pp2ep4' else 4
        require(len(matches) == expected and all(x.get('gpu_tokens') == 0 and x.get('cpu_tokens') == 4032
                for x in matches), 'Expected native CPU4032/GPU0 at diagnostic start')


def components(value):
    """Only validated role/phase means; no target+draft or TPOT normalization."""
    if value is None:
        return None
    numbers = value['exclusive_ms']
    require(set(numbers) == set(COMPONENTS), 'Incomplete component partition')
    require(value['count'] > 0 and all(isinstance(numbers[k], (int, float)) and not isinstance(numbers[k], bool)
            and math.isfinite(numbers[k]) and numbers[k] >= 0 for k in COMPONENTS), 'Invalid component means')
    require(abs(sum(numbers.values()) - value['graph_span_ms']) < .01, 'Graph mean partition does not close')
    return dict(count=value['count'], graph_span_ms=value['graph_span_ms'],
                exclusive_ms={key: numbers[key] for key in COMPONENTS})


def instrumented_overhead(profile, headline):
    """Reject incomplete/nonfinite client diagnostics before exposing any data."""
    values = [profile['wall_s'], *[row[key] for row in profile['requests']
        for key in ('ttft_s', 'tpot_s', 'end_to_end_s')]]
    require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0 for value in values),
            'Nonfinite or negative instrumented client latency')
    original = load('event_report_original_overhead', HERE / 'generate_report.py')
    return original.profile_overhead(profile, headline)


def validate_activity(activity, arm):
    require(activity.get('passed') is True and activity.get('measurement') == MEASUREMENT
            and activity.get('production_tpot_partition') is False, 'Wrong Event activity protocol')
    all_ranks(activity['ranks'], 'Event activity')
    pp, mtp = arm.startswith('pp2ep4'), arm.endswith('_on')
    output = []
    for row in sorted(activity['ranks'], key=lambda value: value['rank']):
        rank = row['rank']
        require(row.get('arm') == arm and row.get('layout') == ('pp2ep4' if pp else 'ep8')
                and row.get('mtp') is mtp, 'Event arm mismatch')
        target, draft = components(row['target']), components(row['draft'])
        require(target is not None and target['count'] == 12, 'Expected twelve actual target Graphs')
        require((draft is not None) == bool(mtp and (not pp or rank >= 4)), 'Incorrect actual draft placement')
        output.append(dict(rank=rank, pp_stage=rank // 4 if pp else None,
            attention_dp_rank=None if pp else rank // 4,
            target=target, draft=draft, phases={key: components(value) for key, value in row['phases'].items()},
            pp_boundary_calls=row['pp_boundary_calls'],
            pp_boundary_api_stream_span_mean_ms=row['pp_boundary_api_stream_span_mean_ms']))
    return output


def rank_group_summary(ranks, arm):
    """Arithmetic means across ranks, not sums or complete pipeline cycles."""
    if ranks is None:
        return None
    pp = arm.startswith('pp2ep4')
    groups = [('ranks0_3', list(range(4))), ('ranks4_7', list(range(4, 8)))] if pp else [('ranks0_7', list(range(8)))]
    result = []
    by_rank = {row['rank']: row for row in ranks}
    for name, ids in groups:
        for role in ('target', 'draft'):
            values = [by_rank[rank][role] for rank in ids]
            require(all(value is None for value in values) or all(value is not None for value in values),
                    'Mixed missing role within rank group')
            if all(value is None for value in values):
                result.append(dict(group=name, ranks=ids, role=role, mean=None))
                continue
            means = {key: statistics.fmean(value['exclusive_ms'][key] for value in values) for key in COMPONENTS}
            moe_layers = 1 if role == 'draft' else (43 if ids[0] == 0 else 46) if pp else 89
            deep_ep = means['deepep_dispatch'] + means['deepep_combine']
            result.append(dict(group=name, ranks=ids, role=role,
                aggregation='Arithmetic mean of per-rank actual Graph-role means; not summed rank/stage time',
                mean=dict(graph_span_ms=statistics.fmean(value['graph_span_ms'] for value in values),
                          counts_per_rank=[value['count'] for value in values], exclusive_ms=means,
                          deepep_dispatch_combine_ms=deep_ep, target_or_draft_moe_layers=moe_layers,
                          deepep_dispatch_combine_ms_per_moe_layer=deep_ep / moe_layers)))
    return result


def kernel_protocol(cfg):
    """Source-supported execution-path distinction, not a measured causal split."""
    result = dict(passed=False, evidence=None)
    try:
        backend = Path(cfg['engine_root']) / 'python/sglang/srt/layers/attention/triton_backend.py'
        common = V8 / 'runtime/common.py'
        source = backend.read_text()
        launch_source = common.read_text()
        forward = source[source.index('    def forward_extend('):source.index('    def _forward_extend_unified(')]
        deterministic_return = forward.index('return self._forward_extend_unified(')
        require("'--attention-backend','triton'" in launch_source
                and "'--enable-deterministic-inference'" in launch_source
                and cfg.get('deterministic_inference') is True
                and 'if self.enable_deterministic:' in forward[:deterministic_return]
                and deterministic_return < forward.index('verify_fwd = None')
                and 'def _should_use_verify_shared_kv' in source
                and 'if not is_gfx95_supported() or topk != 1:' in source
                and 'self.use_verify_splitkv = (' in source
                and 'verify_fwd = None' in source
                and 'self.extend_attention_fwd(' in source
                and 'self.decode_attention_fwd(' in source, 'Pinned Triton source path/gates changed')
        result.update(passed=True, evidence=[evidence(common), evidence(backend)],
            evidence_kind='Source code paths, not a kernel launch trace or isolated cause experiment',
            engine_commit=cfg['engine_commit'], notes=[
                'Correction: pinned arguments also enable deterministic inference. forward_extend returns through _forward_extend_unified before the ordinary Verify dispatcher; actual target Verify uses extend_attention_fwd_unified, not the ordinary extend_attention_fwd fallback.',
                'Ordinary MTP-off Q1 Decode uses forward_decode/decode_attention_fwd with fixed-size KV splits. MTP-on Q3 Verify uses a unified single-stage query-tiled KV loop. Both are Triton, but their KV parallelism differs.',
                'The different query width/kernel path supports a possible Attention-cost explanation, but does not by itself attribute the observed difference to the kernel: scopes include projections, normalization, cache operations and stream waits.',
            ], source_locations=dict(common_attention_backend=141, verify_shared_gate=89,
                verify_splitkv_gate=210, verify_dispatch=1439, verify_extend_fallback=1476,
                deterministic_early_return=next(i for i, line in enumerate(source.splitlines(), 1)
                                                if 'return self._forward_extend_unified(' in line),
                ordinary_decode=1799, decode_dispatch=1926))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def pending_arm(status='not_requested'):
    return dict(valid=False, status=status, gates=None, ranks=None, instrumented_overhead=None,
                source_output_parity=None, rank_group_means=None, evidence=[], error=None)


def profile_arm(run, cfg, arm, production, headline):
    result = pending_arm('pending_or_invalid')
    out = run / arm
    try:
        owned = read(out / 'arm_result.json')
        profile = read(out / 'p4096_decode_profile_results.json')
        saved = read(out / 'stage_activity_summary.json')
        require(owned.get('passed') is True and owned.get('cleanup_verified') is True
                and owned.get('source_performance_valid') is True and owned.get('profile_valid') is True
                and owned.get('performance_valid') is None and owned.get('failure') is None,
                'Profile owner/source/cleanup gate failed')
        owned_cleanup(owned['cleanups'], cfg['nodes'])
        require(profile.get('arm') == arm and profile.get('measurement') == MEASUREMENT
                and profile.get('profile_valid') is True and profile.get('all_rank_gpu_activity_audited') is True
                and profile.get('cupti_enabled') is False and profile.get('no_production_tpot_partition') is True
                and profile.get('source_bank_import_timed') is False and profile.get('hicache_h2d_timed') is True,
                'Profile Event/cache protocol mismatch')
        require(profile['source_performance'] == cfg['profile_source']['arms'][arm], 'Profile source binding changed')
        validate_cache(profile['native_cpu_ready_gpu_empty'], cfg, arm)
        all_ranks(profile['all_rank_after'], 'after diagnostic')
        validator = load('event_report_production_validator', V8 / 'analysis/summarize.py')
        parity = validator.output_parity(production['requests'], profile['requests'])
        require(parity.get('passed') is True and profile.get('source_production_output_parity', {}).get('passed') is True,
                'Diagnostic output differs from original production')
        require(len(profile['requests']) == 8 and all(len(row['output_ids']) == 64 for row in profile['requests']),
                'Incomplete diagnostic output tokens')
        with runtime_imports(V8 / 'runtime'):
            spec = load('event_report_spec_proof', V8 / 'runtime/spec_proof.py')
            graph = spec.validate_receipts(profile['all_rank_after'], cfg['arms'][arm]['layout'], 'profile-', mtp=cfg['arms'][arm]['mtp'])
            layout = cfg['arms'][arm]['layout']
            sched = load('event_report_schedule_proof', V8 / ('runtime/pp_proof.py' if layout == 'pp2ep4' else 'runtime/dp_proof.py'))
            scheduling = sched.validate(profile['all_rank_after'], cfg['arms'][arm]['mtp'], 'profile-') if layout == 'pp2ep4' else sched.validate(profile['all_rank_after'], 'profile-')
        require(graph == profile['actual_graph_proof'] and scheduling == profile['actual_scheduler_proof'],
                'Stored Graph/scheduler proof disagrees with receipts')
        events = load('event_report_event_audit', V8 / 'analysis/event_spans.py')
        audited = events.analyze_arm(out)
        require(audited == saved, 'Saved Event summary disagrees with raw replay/receipt audit')
        for rank in range(8):
            receipt = read(out / f'activity_receipt_rank{rank}.json')
            require(receipt.get('target_forward_count') == 12 and receipt.get('actual_target_graph_replays') == 12
                    and receipt.get('source_performance') == cfg['profile_source']['arms'][arm],
                    'Event receipt missing actual12/source binding')
        ranks = validate_activity(audited, arm)
        for row in ranks:
            raw = read(out / f"event_spans_rank{row['rank']}.json")
            groups = Counter(tuple(segment['request_ids']) for segment in raw['segments'] if segment['phase'].startswith('target_'))
            expected_counts = [6, 6] if arm.startswith('pp2ep4') else [12]
            require(sorted(groups.values()) == expected_counts, 'Unexpected local target request-group sampling')
            row['target_request_groups'] = [dict(request_ids=list(ids), actual_target_forwards=count) for ids, count in groups.items()]
            outputs = [replay['actual_committed_local_output_tokens'] for replay in audited['ranks'][row['rank']]['replays'] if replay['role'] == 'target']
            row['actual_committed_tokens_per_local_target_mean'] = statistics.fmean(outputs)
            row['actual_committed_tokens_per_request_per_local_target_mean'] = statistics.fmean(outputs) / 4
        result.update(valid=True, status='valid_external_event_profile',
            gates=dict(all_eight_ranks=True, actual_twelve_target_graphs=True, actual_graph_shape_B4=True,
                       scheduler_receipts=True, immutable_bank_identity=True, exact_production_output_parity=True,
                       raw_event_summary_recomputed=True, owned_cleanup=True),
            ranks=ranks, rank_group_means=rank_group_summary(ranks, arm),
            instrumented_overhead=instrumented_overhead(profile, headline), source_output_parity=parity,
            evidence=[evidence(out / name) for name in ('arm_result.json', 'p4096_decode_profile_results.json',
                      'stage_activity_summary.json', *[f'event_spans_rank{x}.json' for x in range(8)],
                      *[f'activity_receipt_rank{x}.json' for x in range(8)])])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def build(performance_run=DEFAULT_PERFORMANCE, profile_run=None):
    performance_run = Path(performance_run).resolve()
    original = load('event_report_original_performance', HERE / 'generate_report.py')
    production, _ = original.build(performance_run, None)
    result = dict(schema_version=1, generated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        measurement=MEASUREMENT, new_production_points=0, production=production,
        profiles=dict(run=str(Path(profile_run).resolve()) if profile_run else None,
                      status='not_requested', binding=None, valid_profile_points=0,
                      task_lifecycle=None,
                      arms={arm: pending_arm() for arm in ARMS}),
        external_event_qualification=qualification(), notes=NOTES,
        kernel_protocol=kernel_protocol(read(performance_run / 'config.json')),
        generator=evidence(__file__), event_analyzer=evidence(V8 / 'analysis/event_spans.py'))
    if profile_run is None:
        return result
    profiles = result['profiles']
    run = Path(profile_run).resolve()
    try:
        cfg, binding = validate_binding(run, performance_run)
        profiles['binding'] = binding
        for arm in ARMS:
            source = read(performance_run / arm / 'p4096_decode_results.json')
            profiles['arms'][arm] = profile_arm(run, cfg, arm, source, production['headlines'][arm])
        profiles['valid_profile_points'] = sum(value['valid'] for value in profiles['arms'].values())
        if profiles['valid_profile_points'] == 4:
            task = read(run / 'task/results.json')
            cleanup = read(run / 'task/cleanup.json')
            require(task.get('passed') is True and task.get('profile_points') == 4
                    and task.get('source_performance_points') == 4 and task.get('performance_points') == 0
                    and task.get('mode') == 'profile_only_followup'
                    and task.get('profile_source') == cfg['profile_source'], 'Task profile/source completion failed')
            require(all(cleanup.get(key) is True for key in
                        ('passed', 'gpu_idle', 'model_processes_exited', 'gpu_resources_released')),
                    'Task-level owned cleanup/release failed')
            profiles['task_lifecycle'] = dict(passed=True, profile_points=4, source_performance_points=4,
                new_performance_points=0, gpu_idle=True, model_processes_exited=True,
                gpu_resources_released=True, evidence=[evidence(run / 'task/results.json'), evidence(run / 'task/cleanup.json')])
        profiles['status'] = 'complete' if profiles['valid_profile_points'] == 4 else 'pending_or_invalid'
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        profiles.update(status='pending_or_invalid', error=f'{type(exc).__name__}: {exc}',
                        valid_profile_points=0, task_lifecycle=None,
                        arms={arm: pending_arm('unvalidated_source') for arm in ARMS})
    return result


def markdown(result):
    production, profiles = result['production'], result['profiles']
    lines = ['# GLM-4.7-AWQ four-way：生产性能与独立 CUDA Event 诊断', '',
        '065／167，8×A100。B8，P4096，原生 CPU4032／GPU0，Q1→64；首次 forward 为65 token。生产计时包含 HiCache→GPU，排除权重加载、源 Bank 导入与诊断。', '',
        '## 未加仪器的生产结果（原 v5 两轮，未重测）', '',
        '两轮均值 ± 半极差，不是置信区间。', '',
        '| 配置 | 输出 tok/s | TTFT ms | TPOT ms | E2E ms |', '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        values = production['headlines'][arm]
        cells = [f"{values[key]['mean']:.3f} ± {values[key]['half_range']:.3f}"
                 for key in ('output_tok_s', 'ttft_ms', 'tpot_ms', 'end_to_end_ms')]
        lines.append('| ' + arm + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '同配置两轮 exact token/text parity 通过；不同配置完整输出历史并不相同，不能解释为等 token 历史的纯通信 overhead 对照。', '',
              '| 跨配置 | Timed 全输出相同 / 8 | Repeat 全输出相同 / 8 | 同语义 Bank |', '|---|---:|---:|---|']
    for pair in production['cross_arm_output_parity']:
        lines.append(f"| {pair['left']} / {pair['right']} | {pair['cohorts']['timed']['full_outputs_equal_count']} | {pair['cohorts']['repeat']['full_outputs_equal_count']} | {pair['same_semantic_bank']} |")
    lines += ['', '## CUDA Event 诊断', '', f"状态：{profiles['status']}；有效 profile：{profiles['valid_profile_points']}/4。缺失数据均为 null（表中 —）。", '',
              '以下是每个 rank 的实际 Graph 平均绝对 stream span，单位 ms。不是 CUPTI kernel busy，也不是 TPOT 分解。Draft ordinal 仅标记最近 target，不能把 target+draft 均值相加当成完整 MTP cycle。', '']
    lines += ['### 紧凑分组汇总', '',
              'EP 对八 rank 求均值；PP 分别对每 stage 四 rank 求均值。均为单次实际 Graph-role 平均 span，不能跨 rank/stage 相加。target 的实际 MoE 层数为 EP89、PP stage0 43／stage1 46；draft 为1。', '',
              '| 配置 | Rank组 | Role | Graph ms | A | RoutedE | SharedE | Dense | Dispatch+Combine | D+C / MoE层 ms | NCCL集体通信 | Other | Overlap |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        entries = profiles['arms'][arm]['rank_group_means']
        if entries is None:
            lines.append(f'| {arm} | — | — | ' + ' | '.join(['—'] * 10) + ' |')
            continue
        for row in entries:
            value = row['mean']
            cells = ['—'] * 10 if value is None else [f"{value['graph_span_ms']:.4f}",
                *[f"{value['exclusive_ms'][key]:.4f}" for key in ('attention', 'expert_routed', 'expert_shared', 'dense_ffn')],
                f"{value['deepep_dispatch_combine_ms']:.4f}", f"{value['deepep_dispatch_combine_ms_per_moe_layer']:.4f}",
                *[f"{value['exclusive_ms'][key]:.4f}" for key in ('nccl_tp', 'other_graph', 'overlap')]]
            lines.append(f"| {arm} | {row['group']} | {row['role']} | " + ' | '.join(cells) + ' |')
    lines += ['', 'D+C 归一化是部署配置的 span 对照；两组 peer 数、路由域与 request-group 组合同时变化，不能据此只归因于节点内／节点间链路。12 个 target 是局部 scheduler forwards：EP 每个 DP 组12次，PP 两个 B4 slot 各6次，不是每请求12次。', '']
    labels = ['Attention', 'RoutedE', 'SharedE', 'Dense', 'Dispatch', 'Combine', 'NCCL集体通信', 'Other', 'Overlap']
    for arm in ARMS:
        value = profiles['arms'][arm]
        lines += [f'### {arm}', '', f"状态：{value['status']}。", '']
        if value.get('error'):
            lines += ['验证错误：' + value['error'].replace('\n', ' '), '']
        lines += ['| Rank | Role / phase | Graphs | Graph span | ' + ' | '.join(labels) + ' |',
                  '|---:|---|---:|---:|' + '---:|' * len(labels)]
        ranks = value['ranks'] or [dict(rank=rank, target=None, draft=None, phases={}) for rank in range(8)]
        for row in ranks:
            for role, data in [('target', row['target']), ('draft', row['draft']), *row['phases'].items()]:
                cells = ['—'] * 11 if data is None else [str(data['count']), f"{data['graph_span_ms']:.4f}",
                    *[f"{data['exclusive_ms'][key]:.4f}" for key in COMPONENTS]]
                lines.append(f"| {row['rank']} | {role} | " + ' | '.join(cells) + ' |')
        lines += ['', '| Rank | PP boundary API calls | API stream span mean ms |', '|---:|---:|---:|']
        for row in ranks:
            calls, span = row.get('pp_boundary_calls'), row.get('pp_boundary_api_stream_span_mean_ms')
            lines.append(f"| {row['rank']} | {calls if calls is not None else '—'} | {f'{span:.4f}' if span is not None else '—'} |")
        lines += ['', 'PP boundary 是 current-stream API dependency/enqueue span；不表示异步发送完成或纯 NIC 传输延迟。', '']
        overhead = value['instrumented_overhead']
        lines += ['| 仪器影响（不是性能新结果） | Profile | 未加仪器两轮均值 | 倍数 |', '|---|---:|---:|---:|']
        for key in ('wall_s', 'ttft_ms', 'tpot_ms', 'end_to_end_ms'):
            cells = ['—'] * 3 if overhead is None else [f"{overhead[name][key]:.4f}" for name in
                ('profile_cohort_values', 'source_unprofiled_two_cohort_means', 'profile_over_unprofiled_factor')]
            lines.append('| ' + key + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
    probe = result['external_event_qualification']
    lines += ['## Verify 与普通 Decode 的源码路径', '']
    kernel = result['kernel_protocol']
    if kernel['passed']:
        lines.extend('- ' + note for note in kernel['notes'])
        lines += ['', '这是已固定源码支持的路径区别，不是对 Attention 差距的单因证明；源码 SHA 与位置保存在 summary.json。', '']
    else:
        lines += ['源码路径证据缺失／验证未通过，保持 unknown。', '']
    lines += ['## 机制资格与证据', '', f"Tiny external-event qualification：{probe['status']}（八 rank × 三次 Graph replay；只证明测量机制，不替代完整模型 profile）。", '',
              f"生产源：`{production['source_performance_run']}`", '', f"Profile 源：`{profiles['run']}`", '',
              'summary.json 保存逐文件 SHA、源绑定、Graph/cache/cleanup/输出 parity gates 以及完整生产统计。', '', '## 边界', '']
    lines.extend('- ' + note for note in result['notes'])
    if profiles.get('error'):
        lines += ['', '源绑定错误：' + profiles['error'].replace('\n', ' ')]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--performance-run', type=Path, default=DEFAULT_PERFORMANCE)
    parser.add_argument('--profile-run', type=Path)
    parser.add_argument('--output', type=Path, default=HERE / 'event_current')
    args = parser.parse_args()
    output = args.output.resolve()
    # This helper owns only a separate output. Never overwrite source/current.
    forbidden = [HERE / 'current', args.performance_run, *([args.profile_run] if args.profile_run else [])]
    require(not any(output == path.resolve() or path.resolve() in output.parents for path in forbidden),
            'Output must be separate from old current and source artifacts')
    result = build(args.performance_run, args.profile_run)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (output / 'report.md').write_text(markdown(result))
    print(json.dumps(dict(output=str(output), production_points=4,
                         valid_event_profiles=result['profiles']['valid_profile_points'])))


if __name__ == '__main__':
    main()
