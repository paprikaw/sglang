"""CPU-only reporting/protocol tests; no engine, Torch or GPU imports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('fourway_event_report', Path(__file__).with_name('generate_event_report.py'))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
events = report.load('fourway_report_event_test_audit', report.V8 / 'analysis/event_spans.py')


def role_mean(count=12):
    values = {key: float(index + 1) for index, key in enumerate(report.COMPONENTS)}
    return dict(count=count, graph_span_ms=sum(values.values()), exclusive_ms=values)


def activity(arm='ep8_mtp_off'):
    pp, mtp = arm.startswith('pp2ep4'), arm.endswith('_on')
    rows = []
    for rank in range(8):
        target, draft = role_mean(), role_mean(24) if mtp and (not pp or rank >= 4) else None
        rows.append(dict(rank=rank, passed=True, arm=arm, layout='pp2ep4' if pp else 'ep8', mtp=mtp,
            target=target, draft=draft, phases={'target_verify' if mtp else 'target_decode': target},
            pp_boundary_calls=2 if pp else 0, pp_boundary_api_stream_span_mean_ms=.1 if pp else None))
    return dict(passed=True, measurement=report.MEASUREMENT, production_tpot_partition=False, ranks=rows)


def raw_rank(rank=0):
    """Complete real analyzer schema; fabricated timings are only test fixtures."""
    segments, replays, accepted = [], [], []
    ids = [f'profile-decode-4096-{i}' for i in range(4)]
    intervals = []
    for layer in range(92):
        labels = ['attention', 'dense_ffn'] if layer < 3 else ['attention', 'expert_routed', 'expert_shared', 'deepep_dispatch', 'deepep_combine']
        for index, label in enumerate(labels):
            start = layer + index * .1
            intervals.append(dict(label=label, layer_id=layer, stream=1,
                                  start_ms=start, end_ms=start + .05, elapsed_ms=.05))
    for index in range(12):
        segment = dict(index=index, target_ordinal=index + 1, phase='target_decode', batch_size=4, request_ids=ids)
        segments.append(segment)
        accepted.append(dict(target_segment_index=index, output_tokens_by_request={rid: 1 for rid in ids}, output_tokens=4))
        replays.append(dict(segment, segment_index=index, role='target', graph_span_ms=92.,
            completion_fences=1, component_fences=0, read_before_same_graph_reuse=True,
            anchor_stream=1, layer_ids=list(range(92)), intervals=intervals))
    raw = dict(rank=rank, arm='ep8_mtp_off', layout='ep8', mtp=False, measurement=report.MEASUREMENT,
        cupti_enabled=False, production_timing_affected=False, diagnostic_cadence_changed=True,
        segments=segments, replays=replays, accepted=accepted, pp_boundaries=[])
    receipt = dict(rank=rank, arm=raw['arm'], passed=True, measurement=report.MEASUREMENT)
    return raw, receipt


class EventReportTests(unittest.TestCase):
    def test_components_are_absolute_means_not_tpot(self):
        value = report.components(role_mean())
        self.assertEqual(value['exclusive_ms']['attention'], 1)
        self.assertEqual(value['exclusive_ms']['deepep_dispatch'], 5)
        self.assertEqual(value['graph_span_ms'], 45)
        self.assertEqual(value['count'], 12)
        self.assertNotIn('tpot_ms', value)

    def test_missing_role_remains_null(self):
        self.assertIsNone(report.components(None))
        self.assertIsNone(report.pending_arm()['ranks'])
        self.assertIsNone(report.pending_arm()['instrumented_overhead'])

    def test_bad_partition_or_nan_fails(self):
        for key, val in [('attention', float('nan')), ('attention', -1)]:
            value = role_mean(); value['exclusive_ms'][key] = val
            with self.assertRaises(ValueError): report.components(value)
        value = role_mean(); value['graph_span_ms'] += 1
        with self.assertRaises(ValueError): report.components(value)

    def test_all_eight_rank_target_and_draft_placement(self):
        rows = report.validate_activity(activity('pp2ep4_mtp_on'), 'pp2ep4_mtp_on')
        self.assertEqual([x['rank'] for x in rows], list(range(8)))
        self.assertTrue(all(x['draft'] is None for x in rows[:4]))
        self.assertTrue(all(x['draft']['count'] == 24 for x in rows[4:]))
        self.assertEqual(rows[4]['pp_stage'], 1)
        self.assertNotIn('target_plus_draft_ms', rows[4])
        self.assertEqual(rows[4]['pp_boundary_api_stream_span_mean_ms'], .1)

    def test_missing_or_duplicate_rank_and_wrong_target_count_fail(self):
        for mutate in (lambda x: x['ranks'].pop(),
                       lambda x: x['ranks'][-1].update(rank=0),
                       lambda x: x['ranks'][0]['target'].update(count=11)):
            value = activity(); mutate(value)
            with self.assertRaises(ValueError): report.validate_activity(value, 'ep8_mtp_off')

    def test_wrong_measurement_and_phantom_draft_fail(self):
        value = activity(); value['measurement'] = 'cupti_kernel_busy'
        with self.assertRaises(ValueError): report.validate_activity(value, 'ep8_mtp_off')

    def test_rank_group_means_use_actual_moe_layer_denominators(self):
        ep = report.validate_activity(activity('ep8_mtp_off'), 'ep8_mtp_off')
        rows = report.rank_group_summary(ep, 'ep8_mtp_off')
        self.assertEqual(rows[0]['mean']['target_or_draft_moe_layers'], 89)
        self.assertAlmostEqual(rows[0]['mean']['deepep_dispatch_combine_ms_per_moe_layer'], 11 / 89)
        self.assertIsNone(rows[1]['mean'])
        pp = report.validate_activity(activity('pp2ep4_mtp_on'), 'pp2ep4_mtp_on')
        rows = report.rank_group_summary(pp, 'pp2ep4_mtp_on')
        self.assertEqual([row['mean']['target_or_draft_moe_layers'] if row['mean'] else None for row in rows], [43, None, 46, 1])
        self.assertAlmostEqual(rows[0]['mean']['deepep_dispatch_combine_ms_per_moe_layer'], 11 / 43)
        self.assertAlmostEqual(rows[2]['mean']['deepep_dispatch_combine_ms_per_moe_layer'], 11 / 46)
        self.assertEqual(rows[0]['mean']['graph_span_ms'], 45)  # Mean, not four-rank sum.
        self.assertIsNone(report.rank_group_summary(None, 'ep8_mtp_on'))

    def test_pinned_kernel_path_is_source_evidence_not_cause_proof(self):
        value = report.kernel_protocol(report.read(report.DEFAULT_PERFORMANCE / 'config.json'))
        self.assertTrue(value['passed'], value.get('error'))
        self.assertEqual(value['source_locations']['verify_extend_fallback'], 1476)
        self.assertIn('not a kernel launch trace', value['evidence_kind'])
        self.assertIn('does not by itself attribute', ' '.join(value['notes']))
        self.assertIn('extend_attention_fwd_unified', ' '.join(value['notes']))
        self.assertLess(value['source_locations']['deterministic_early_return'],
                        value['source_locations']['verify_dispatch'])
        nondeterministic = report.read(report.DEFAULT_PERFORMANCE / 'config.json')
        nondeterministic['deterministic_inference'] = False
        self.assertFalse(report.kernel_protocol(nondeterministic)['passed'])
        value = activity(); value['ranks'][0]['draft'] = role_mean()
        with self.assertRaises(ValueError): report.validate_activity(value, 'ep8_mtp_off')

    def test_real_event_analyzer_accepts_complete_actual12_fixture(self):
        raw, receipt = raw_rank()
        value = events.analyze_rank(raw, receipt)
        self.assertEqual(value['target']['count'], 12)
        self.assertIsNone(value['draft'])
        self.assertEqual(value['replays'][0]['actual_committed_local_output_tokens'], 4)
        self.assertAlmostEqual(sum(value['target']['exclusive_ms'].values()), 92.)

    def test_real_event_analyzer_rejects_incomplete_tokens_and_shape(self):
        raw, receipt = raw_rank()
        raw['accepted'][0]['output_tokens'] = 0
        with self.assertRaisesRegex(ValueError, 'committed output'): events.analyze_rank(raw, receipt)
        raw, receipt = raw_rank(); raw['replays'][0]['batch_size'] = 8
        with self.assertRaises(ValueError): events.analyze_rank(raw, receipt)
        raw, receipt = raw_rank(); raw['replays'][0]['layer_ids'].pop()
        with self.assertRaisesRegex(ValueError, 'layer coverage'): events.analyze_rank(raw, receipt)

    def test_tiny_qualification_saved_evidence(self):
        value = report.qualification()
        self.assertTrue(value['valid'], value.get('error'))
        self.assertEqual(value['data']['ranks'], list(range(8)))
        self.assertEqual(value['data']['replays_per_rank'], 3)

    def test_sha_binding_and_owned_cleanup_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.json'; path.write_text('{}')
            item = report.evidence(path)
            self.assertEqual(report.bound(item), item)
            path.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'SHA-bound'): report.bound(item)
        rows = [dict(node=x, owned_children_exited=True, failure=None, gpu_cleanup={'passed': True}) for x in ('a', 'b')]
        report.owned_cleanup(rows, ['a', 'b'])
        rows[0]['owned_children_exited'] = False
        with self.assertRaises(ValueError): report.owned_cleanup(rows, ['a', 'b'])

    def test_instrumented_overhead_is_separate_and_nonfinite_fails(self):
        profile = dict(wall_s=4, requests=[dict(ttft_s=.2, tpot_s=.04, end_to_end_s=3) for _ in range(8)])
        baseline = {key: dict(mean=value) for key, value in
                    dict(wall_s=2, ttft_ms=100, tpot_ms=20, end_to_end_ms=1500).items()}
        value = report.instrumented_overhead(profile, baseline)
        self.assertEqual(set(value['profile_over_unprofiled_factor'].values()), {2})
        self.assertTrue(value['diagnostic_only'])
        profile['requests'][0]['tpot_s'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'): report.instrumented_overhead(profile, baseline)

    def test_production_bank_receipts_match_real_cold_protocol(self):
        cfg = report.read(report.DEFAULT_PERFORMANCE / 'config.json')
        for arm in report.ARMS:
            raw = report.read(report.DEFAULT_PERFORMANCE / arm / 'p4096_decode_results.json')
            report.validate_cache(raw['native_cpu_ready_gpu_empty'], cfg, arm)
            raw['native_cpu_ready_gpu_empty'][0]['bank_import']['bank_identity_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'Bank/import identity'):
                report.validate_cache(raw['native_cpu_ready_gpu_empty'], cfg, arm)

    def test_missing_profile_produces_no_fake_numbers_and_source_unchanged(self):
        value = report.build()
        self.assertEqual(value['new_production_points'], 0)
        self.assertEqual(value['production']['completed_valid_performance_points'], 4)
        self.assertEqual(len(value['production']['cross_arm_output_parity']), 6)
        self.assertAlmostEqual(value['production']['headlines']['ep8_mtp_on']['tpot_ms']['mean'], 69.72082226818044)
        self.assertEqual(value['profiles']['valid_profile_points'], 0)
        self.assertTrue(all(row['ranks'] is None for row in value['profiles']['arms'].values()))
        text = report.markdown(value)
        self.assertIn('不是 CUPTI kernel busy', text)
        self.assertIn('target+draft', text)
        self.assertIn('异步发送完成', text)
        self.assertIn('NCCL集体通信', text)
        self.assertIn('Raw data omit group ranks', text)
        self.assertIn('outer input RMSNorm', text)
        self.assertIn('PP 两个 B4 slot 各6次', text)
        self.assertIn('D+C / MoE层 ms', text)
        self.assertIn('null', text)
        json.dumps(value, allow_nan=False)

    def test_nonexistent_profile_retains_production_and_null_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = report.build(profile_run=Path(tmp) / 'not-created')
        self.assertEqual(value['profiles']['status'], 'pending_or_invalid')
        self.assertEqual(value['profiles']['valid_profile_points'], 0)
        self.assertTrue(all(row['ranks'] is None for row in value['profiles']['arms'].values()))


if __name__ == '__main__':
    unittest.main()
