"""Run with the pinned AgentX runtime, without loading Torch or a model."""
from types import SimpleNamespace
import unittest

from aiperf.common.scenario.registry import SCENARIOS
from aiperf.common.scenario import validator
from aiperf.config.flags import CLIConfig, converter
from aiperf.timing.config import _build_profiling_config
from aiperf.timing.request_cancellation import RequestCancellationConfig
from finite_profile import install


class FiniteProfileTests(unittest.TestCase):
    def setUp(self):
        self.defaults = converter._apply_scenario_defaults
        self.spec = SCENARIOS['agentx']

    def tearDown(self):
        converter._apply_scenario_defaults = self.defaults
        SCENARIOS['agentx'] = self.spec

    def cli(self, **changes):
        return CLIConfig(**(dict(scenario='agentx',request_count=76,
            concurrency=10,model_names=['qwen122b']) | changes))

    def test_native_config_stops_by_count_and_keeps_agentic_timing(self):
        self.assertEqual(self.defaults(self.cli()).benchmark_duration,3600)
        install(76)
        cfg=converter.convert_cli_to_aiperf(self.cli())
        run=SimpleNamespace(cfg=cfg.benchmark)
        violations=[]; applied=[]
        spec=SCENARIOS['agentx']
        validator._apply_timing_mode(run,spec,violations,applied)
        validator._apply_duration(run,spec,violations,applied)
        self.assertEqual(violations,[])
        phase=cfg.benchmark.get_profiling_phases()[0]
        credit=_build_profiling_config(phase,
            default_cancellation=RequestCancellationConfig(),
            phase_index=1,profiling_index=0)
        self.assertEqual(credit.total_expected_requests,76)
        self.assertIsNone(credit.expected_duration_sec)
        self.assertEqual(str(credit.timing_mode),'agentic_replay')
        self.assertEqual(credit.concurrency,10)
        native=self.spec.model_dump();finite=spec.model_dump()
        self.assertEqual({k for k in native if native[k]!=finite[k]},
            {'default_benchmark_duration_seconds','min_benchmark_duration_seconds'})

    def test_mixed_stop_rules_and_changed_request_count_are_rejected(self):
        install(76)
        for changes in (dict(benchmark_duration=1200),dict(benchmark_grace_period=150),
                        dict(request_count=75),dict(scenario=None)):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                converter.convert_cli_to_aiperf(self.cli(**changes))

    def test_install_is_idempotent_but_cannot_change_count(self):
        install(76);installed=converter._apply_scenario_defaults
        install(76)
        self.assertIs(converter._apply_scenario_defaults,installed)
        with self.assertRaises(RuntimeError):install(75)

    def test_invalid_count_leaves_native_scenario_unchanged(self):
        with self.assertRaises(ValueError):install(0)
        self.assertIs(SCENARIOS['agentx'],self.spec)


if __name__=='__main__':
    unittest.main()
