import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('fourway_report_generator', Path(__file__).with_name('generate_report.py'))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReportTests(unittest.TestCase):
    def test_two_cohort_estimator_and_variability(self):
        value = report.two_cohort([10, 14])
        self.assertEqual(value['mean'], 12)
        self.assertEqual(value['half_range'], 2)
        self.assertIsNone(report.two_cohort([10]))

    def test_native_acceptance_denominator_and_means(self):
        raw = dict(requests=[dict(meta=dict(spec_verify_ct=v, spec_num_correct_drafts=c,
            spec_num_proposed_drafts=2*v, completion_tokens=64,
            spec_accept_rate=c/(2*v), spec_accept_length=64/v)) for v, c in [(2, 1), (4, 4)]])
        result = report.acceptance(raw, True)
        self.assertEqual(result['proposed_draft_count'], 12)
        self.assertEqual(result['pooled_draft_accept_rate'], 5/12)
        self.assertNotEqual(result['pooled_draft_accept_rate'], result['request_mean_accept_rate'])
        self.assertIsNone(report.acceptance(raw, False))

    def test_denominator_is_fail_closed(self):
        raw = dict(requests=[dict(meta=dict(spec_verify_ct=2, spec_num_correct_drafts=1,
            spec_num_proposed_drafts=6, completion_tokens=64, spec_accept_rate=1/6, spec_accept_length=32))])
        with self.assertRaises(ValueError):
            report.acceptance(raw, True)

    def test_topology_cycle_not_tpot(self):
        ep, pp = report.topology('ep8_mtp_on'), report.topology('pp2ep4_mtp_on')
        self.assertEqual(ep['target_layers_per_rank_group']['ranks0_3'], 92)
        self.assertEqual(pp['target_layers_per_rank_group']['ranks0_3'], 46)
        self.assertEqual(pp['draft_layers_per_rank_group'], {'ranks0_3': 0, 'ranks4_7': 1})
        self.assertEqual(report.topology('pp2ep4_mtp_off')['draft_layers_per_rank_group']['ranks4_7'], 0)

    def test_no_profile_is_null(self):
        value = report.profile_data(None, report.DEFAULT_PERFORMANCE)
        self.assertEqual(value['valid_profile_points'], 0)
        self.assertIsNone(value['data'])
        self.assertEqual(value['status'], 'not_requested')

    def test_profiler_inflation_is_separate_diagnostic(self):
        profile = dict(wall_s=4, requests=[dict(ttft_s=.2, tpot_s=.04, end_to_end_s=3) for _ in range(8)])
        baseline = {key: dict(mean=value) for key, value in dict(wall_s=2, ttft_ms=100, tpot_ms=20, end_to_end_ms=1500).items()}
        value = report.profile_overhead(profile, baseline)
        self.assertEqual(set(value['profile_over_unprofiled_factor'].values()), {2})
        self.assertTrue(value['diagnostic_only'])

    def test_initial_saved_source_is_complete_and_discloses_parity(self):
        result, rows = report.build(report.DEFAULT_PERFORMANCE)
        self.assertEqual(len(rows), 8)
        self.assertEqual(result['completed_valid_performance_points'], 4)
        self.assertEqual(len(result['cross_arm_output_parity']), 6)
        pair = result['cross_arm_output_parity'][0]['cohorts']['timed']
        self.assertEqual(pair['first_output_token_equal_count'], 8)
        self.assertEqual(pair['full_outputs_equal_count'], 2)
        self.assertFalse(pair['passed'])
        self.assertAlmostEqual(result['headlines']['ep8_mtp_on']['tpot_ms']['mean'], 69.72082226818044)
        self.assertIn('不是置信区间', report.markdown(result))


if __name__ == '__main__':
    unittest.main()
