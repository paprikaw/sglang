"""Append-only cohort: all batch evidence survives with bounded inode use."""
import asyncio
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import prefill_worker_append as worker
from test_prefill_worker import GuardFixture


def row(sample_id, status='ok'):
    return dict(node='a', sample_id=sample_id, status=status, planned_prompt_tokens=10,
        prompt_tokens=10, completion_tokens=1, ttft_s=.01, e2e_s=.02, tpot_s=None,
        server_queue_s=.03, dispatch_epoch=1., finished_epoch=2., source_index=1,
        source_cycle=0, source_model='GPT-4', original_output_tokens=50, error=None)


class AppendArtifacts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='append-artifacts-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.stage = Path(self.temp.name) / 'stages/run-a/a'

    def write_batch(self, phase, rows, offered=None):
        workload = dict(requests=[{'sample_id': r['sample_id']} for r in rows],
                        stream=dict(source_cycle=0), marker=phase)
        with worker.BatchArtifactWriter(self.stage, phase, workload) as artifacts:
            for result in rows:
                artifacts.append_request(result)
        summary = worker.batch_summary(rows, 1, 2, len(rows) if offered is None else offered)
        worker.append_batch_summary(self.stage, phase, summary)

    def test_two_batches_preserve_all_phase_linked_rows_and_single_csv_header(self):
        self.write_batch('phase-1', [row('a'), row('b')])
        initial_files = {p.name for p in self.stage.iterdir()}
        self.write_batch('phase-2', [row('c')])
        self.assertEqual(initial_files, {p.name for p in self.stage.iterdir()})
        workloads = [json.loads(line) for line in (self.stage/'workloads.jsonl').read_text().splitlines()]
        requests = [json.loads(line) for line in (self.stage/'requests.jsonl').read_text().splitlines()]
        summaries = [json.loads(line) for line in (self.stage/'summaries.jsonl').read_text().splitlines()]
        with (self.stage/'full_metrics.csv').open() as handle:
            metrics = list(csv.DictReader(handle))
        self.assertEqual([w['phase_id'] for w in workloads], ['phase-1','phase-2'])
        self.assertEqual([w['workload']['marker'] for w in workloads], ['phase-1','phase-2'])
        self.assertEqual([r['sample_id'] for r in requests], ['a','b','c'])
        self.assertEqual([r['phase_id'] for r in requests], ['phase-1','phase-1','phase-2'])
        self.assertEqual([r['phase_id'] for r in metrics], [r['phase_id'] for r in requests])
        self.assertEqual([s['phase_id'] for s in summaries], ['phase-1','phase-2'])
        self.assertEqual(json.loads((self.stage/'summary.json').read_text())['phase_id'], 'phase-2')
        self.assertEqual(metrics[0]['ttft_ms'], '10.0')

    def test_partial_and_failed_batches_preserve_prior_rows_without_committing_validity(self):
        self.write_batch('good', [row('a')])
        self.write_batch('partial', [row('b'), row('c', 'cancelled')], offered=4)
        self.write_batch('retry', [row('b'), row('c')], offered=2)
        summaries = [json.loads(line) for line in (self.stage/'summaries.jsonl').read_text().splitlines()]
        self.assertEqual([s['valid'] for s in summaries], [True,False,True])
        requests = [json.loads(line) for line in (self.stage/'requests.jsonl').read_text().splitlines()]
        self.assertEqual([r['phase_id'] for r in requests], ['good','partial','partial','retry','retry'])
        self.assertEqual(requests[2]['status'], 'cancelled')
        self.assertEqual(len(list(self.stage.iterdir())), 5)

    def test_file_count_remains_constant_after_many_batches(self):
        self.write_batch('phase-0', [row('first')])
        files = {p.relative_to(self.stage) for p in self.stage.rglob('*')}
        for index in range(1, 20):
            self.write_batch('phase-'+str(index), [row(str(index))])
        self.assertEqual({p.relative_to(self.stage) for p in self.stage.rglob('*')}, files)
        self.assertEqual(len((self.stage/'summaries.jsonl').read_text().splitlines()), 20)


class AppendWorkerIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_work_loop_two_batches_retains_real_guard_trace_commit_and_constant_stage(self):
        with tempfile.TemporaryDirectory(prefix='append-worker-', dir='/tmp') as name:
            fixture = GuardFixture(Path(name))
            fixture.config['concurrent_requests_per_instance'] = 8
            fixture.sync()
            sent = []
            async def send(session, endpoint, request, stop_check, **kwargs):
                sent.append(request)
                await asyncio.sleep(0)
                result = row(request['sample_id'])
                result.update(planned_prompt_tokens=len(request['input_ids']), prompt_tokens=len(request['input_ids']))
                if len(sent) >= 8:
                    fixture.lease_a['preemption_requested'] = dict(request_id='foreground')
                    fixture.sync()
                return result
            loop = asyncio.get_running_loop()
            with mock.patch.dict(os.environ, EXPERIMENT_GUARD_LEASE_RECEIPT=str(fixture.receipt)), \
                 mock.patch.object(loop, 'add_signal_handler'), \
                 mock.patch.object(worker, 'send_prefill', side_effect=send):
                await asyncio.wait_for(worker.work(fixture.args()), 3)
            stages = list(fixture.runtime.glob('stages/*/a'))
            self.assertEqual(len(stages), 1)
            stage = stages[0]
            self.assertEqual(stage.parent.name, fixture.run.name)
            workloads = [json.loads(s) for s in (stage/'workloads.jsonl').read_text().splitlines()]
            summaries = [json.loads(s) for s in (stage/'summaries.jsonl').read_text().splitlines()]
            requests = [json.loads(s) for s in (stage/'requests.jsonl').read_text().splitlines()]
            self.assertEqual(len(sent), 8)
            self.assertEqual(len(workloads), 2)
            self.assertEqual(len(summaries), 2)
            self.assertEqual(len({w['phase_id'] for w in workloads}), 2)
            self.assertEqual({r['phase_id'] for r in requests}, {w['phase_id'] for w in workloads})
            progress = json.loads((fixture.run/'benchmark_progress.json').read_text())
            raw_progress = json.loads((stage/'progress.json').read_text())
            self.assertEqual(progress['storage_layout'], 'append_per_run_v1')
            self.assertEqual(progress['phase'], raw_progress['phase'])
            self.assertEqual(progress['storage_phase'], fixture.run.name)
            cursor = json.loads((fixture.runtime/'cursors/a.json').read_text())
            self.assertEqual(cursor['committed_batches'], 2)
            self.assertEqual(cursor['committed_requests'], 8)
            for path in fixture.runtime.rglob('*'):
                if path.is_file():
                    self.assertNotIn('input_ids', path.read_text())


class AppendProgressConsumer(unittest.TestCase):
    def test_append_layout_is_exact_run_phase_bound_and_legacy_remains_supported(self):
        runner_path = Path(__file__).resolve().parents[2]/'experiments/cluster_backfill_20261020/node_runner_qwen_resilient.py'
        spec = importlib.util.spec_from_file_location('append_progress_runner', runner_path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory(prefix='append-consumer-', dir='/tmp') as name:
            root = Path(name)
            run = root/'run-a';run.mkdir()
            benchmark_config = root/'benchmark.json'
            runner.save(benchmark_config, dict(runtime_root=str(root/'prefill')))
            cfg = dict(benchmark=dict(config_path=str(benchmark_config)))
            stage = root/'prefill/stages/run-a/a'
            progress = dict(phase='batch-1', storage_phase='run-a',storage_layout='append_per_run_v1',
                offered_requests=4,dispatched_requests=4,completed_requests=4,successful_requests=4,inflight=0)
            runner.save(stage/'progress.json', progress)
            binding = dict(phase='batch-1',storage_phase='run-a',storage_layout='append_per_run_v1',stage_dir=str(stage))
            runner.save(run/'benchmark_progress.json', binding)
            self.assertEqual(runner.benchmark_progress_snapshot(cfg,run,'a')['benchmark_phase'],'batch-1')
            runner.save(stage/'progress.json', dict(progress,phase='previous-batch'))
            self.assertEqual(runner.benchmark_progress_snapshot(cfg,run,'a'), {})
            runner.save(stage/'progress.json', progress)
            runner.save(run/'benchmark_progress.json', dict(binding,storage_phase='wrong-run'))
            self.assertEqual(runner.benchmark_progress_snapshot(cfg,run,'a'), {})
            legacy = root/'prefill/stages/legacy-batch/a'
            runner.save(legacy/'progress.json',progress)
            runner.save(run/'benchmark_progress.json',dict(phase='legacy-batch',stage_dir=str(legacy)))
            self.assertEqual(runner.benchmark_progress_snapshot(cfg,run,'a')['benchmark_phase'],'legacy-batch')


if __name__ == '__main__':
    unittest.main()
