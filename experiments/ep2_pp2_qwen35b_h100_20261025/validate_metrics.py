"""Recompute request metrics before publishing a small reproducible report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys


def analyze(run):
    sys.path.insert(0, '/home/bxb1/.codex/skills/llm-experiment-guard/scripts')
    from artifact_visibility import wait_for_stable_json
    cfg = json.loads((run / 'config.json').read_text())
    complete, visibility = wait_for_stable_json(run / 'completed.json', timeout_s=120)
    if complete.get('passed') is not True:
        raise AssertionError('No complete paired comparison')
    cleanup, cleanup_visibility = wait_for_stable_json(run / 'worker_cleanup.json', timeout_s=120)
    if cleanup.get('cleanup_verified') is not True:
        raise AssertionError('Cleanup has not passed')
    postchecks = [(p, json.loads(p.read_text()))
                  for p in (run / 'dashboard_guard/checks').glob('*-post.json')]
    passed_postchecks = [(p, d) for p, d in postchecks
                         if d.get('passed') is True and d.get('scientific_success') is True
                         and d.get('cleanup_verified') is True]
    cache_review = None
    if not passed_postchecks:
        from cache_review import review
        assert len(postchecks) == 1, 'Ambiguous Guard postcheck history'
        cache_review = review(run, postchecks[0][0])
        assert cache_review['passed']
    selected_post = passed_postchecks[-1] if passed_postchecks else postchecks[0]
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    rows, provenance, checks = [], [], []
    for arm in ('ep2', 'pp2'):
        for path in sorted((run / arm).glob('metrics.*.json')):
            data, receipt = wait_for_stable_json(path, timeout_s=120)
            checks.append(receipt)
            assert data['valid'] and not data['profiler_enabled']
            assert len(data['rounds']) == cfg['repeats']
            phase = path.stem.split('.')[-1]
            case = data['case']
            flat = []
            rates = []
            for group in data['rounds']:
                reqs = group['requests']
                assert len(reqs) == case['requests']
                elapsed = max(r['finished'] for r in reqs) - min(r['started'] for r in reqs)
                assert abs(elapsed - group['makespan_s']) < 1e-7
                count = 1 if phase == 'prefill' else case['output_tokens']
                assert all(r['output_tokens'] == count for r in reqs)
                context = case['prefill_cached_context_tokens'] if phase == 'prefill' else case['cached_context_tokens']
                fresh = case['new_prefill_tokens'] if phase == 'prefill' else 1
                assert all(r['prompt_tokens'] == context + fresh for r in reqs)
                if context:
                    assert all(r['cached_tokens'] >= context - 64 for r in reqs)
                else:
                    assert all((r['cached_tokens'] or 0) == 0 for r in reqs)
                rates.append(case['requests'] * (fresh if phase == 'prefill' else count) / elapsed)
                flat.extend(reqs)
            row = dict(arm=arm, phase=phase, case=case['id'], requests=case['requests'],
                cached_context_per_request=case['prefill_cached_context_tokens'] if phase == 'prefill' else case['cached_context_tokens'],
                fresh_input_per_request=case['new_prefill_tokens'] if phase == 'prefill' else 1,
                output_per_request=1 if phase == 'prefill' else case['output_tokens'],
                tokens_per_s=statistics.mean(rates), tokens_per_s_min=min(rates),
                tokens_per_s_max=max(rates), tokens_per_s_std=statistics.stdev(rates),
                rounds=len(data['rounds']),
                ready_send=data['selected_ready_send'], gpu_pair='0,1' if arm == 'ep2' else '2,3')
            for metric in ('ttft_s', 'tpot_ms', 'e2e_s'):
                values = sorted(r[metric] for r in flat if r[metric] is not None)
                if values:
                    expected = dict(mean=statistics.mean(values), median=statistics.median(values),
                        p99=values[min(len(values)-1, int(.99 * len(values)))])
                    for stat, value in expected.items():
                        assert abs(data[metric][stat] - value) < 1e-7
                        row[metric + '_' + stat] = value
                else:
                    for stat in ('mean', 'median', 'p99'):
                        row[metric + '_' + stat] = None
            measured = data['fresh_input_tokens_per_s'] if phase == 'prefill' else data['output_tokens_per_s']
            assert abs(row['tokens_per_s'] - measured) < 1e-6
            rows.append(row)
            provenance.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(rows) == 12 + 2 * len(complete.get('extra_cases', [])), len(rows)
    with (out / 'full_metrics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    qualifications = {}
    tuning = {}
    for arm in ('ep2', 'pp2'):
        qualifications[arm] = [json.loads(p.read_text())
                               for p in (run / arm / 'qualification').glob('*.json')]
        assert len(qualifications[arm]) == 2 and all(q['passed'] for q in qualifications[arm])
        log = (run / 'cache' / (arm + '.log')).read_text(errors='replace')
        assert any('Decode batch' in line and 'cuda graph: True' in line
                   for line in log.splitlines()), 'No real Decode Graph replay in ' + arm
        tuning[arm] = json.loads((run / arm / 'tuning.json').read_text())
        assert tuning[arm]['output_agreement'] is True
    evidence = dict(passed=True, rows=rows, sources=provenance,
        guard_postcheck=str(selected_post[0]),
        guard_original_passed=selected_post[1]['passed'],
        supplemental_cache_review=cache_review,
        logical_capacities=complete['logical_capacities'], qualifications=qualifications,
        tuning=tuning, actual_decode_graph_replay_both_arms=True,
        visibility=[visibility, cleanup_visibility] + checks,
        percentiles='Descriptive request samples, not stable service-tail estimates')
    (out / 'validated_metrics.json').write_text(json.dumps(evidence, indent=2) + '\n')
    single, _ = wait_for_stable_json(run/'single_gpu_fit.json', timeout_s=60)
    assert single['passed'] and single['allocator_tokens'] > 4096
    assert len(single['proof']['requests']) == 1
    assert single['proof']['requests'][0]['output_tokens'] == 4
    assert single['proof']['requests'][0]['prompt_tokens'] == 4096
    single_qual = list((run/'single/qualification').glob('*.json'))
    assert len(single_qual) == 1
    single_qual = json.loads(single_qual[0].read_text())
    assert single_qual['passed'] and single_qual['layers'] == 40
    assert single_qual['weight_dtype'] == 'torch.bfloat16'
    single_cleanup, _ = wait_for_stable_json(run/'single_gpu_cleanup.json',timeout_s=60)
    assert single_cleanup['cleanup_verified']
    for arm in ('single','ep2','pp2'):
        frozen, _ = wait_for_stable_json(run/arm/'gc_freeze.json', timeout_s=60)
        assert frozen['passed'] and frozen['initialization_only']
    evidence['single_gpu_fit'] = single
    evidence['single_gpu_qualification'] = single_qual
    (out/'validated_metrics.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return evidence
