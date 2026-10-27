"""Counterbalanced PP scheduling comparisons; compute kernels stay unchanged."""
import hashlib
from pathlib import Path
import time

from lookahead_workloads import byte
from previous_workloads import prompts
from artifact_visibility import wait_for_stable_json


def install(base):
    def switch(cfg, policy):
        byte(cfg['gap_policy_path'], ord('a') if policy == 10 else ord(str(policy)))

    def states(cfg, key, policy):
        # The public PP response contains only the head stage. Each broadcast
        # control request also publishes a private receipt on both stages.
        for _ in range(3):
            base.h.http_json(cfg['port'], '/get_server_info', key, timeout=30)
        directory = Path(cfg['code_root']).parent / 'pp2/gap_receipts'
        values = [wait_for_stable_json(directory / f'policy{policy}.pp{rank}.json',
                                      timeout_s=20, poll_s=.01, min_stable_s=.02)[0]
                  for rank in (0, 1)]
        assert len(values) == 2
        for s in values:
            assert s['policy'] == policy
            assert not s['pending'] and not s['ahead_pending']
            assert s['produced'] == s['retired']
            assert s['ahead_produced'] == s['ahead_consumed']
            assert s['prepared_next'] == s['prepared_consumed']
        return values

    def workloads(run, cfg, arm, key, health):
        assert arm == 'pp2'
        from benchmark import policy as expert_policy
        from sampling import set_diagnostic
        expert_policy(cfg, b'0')
        set_diagnostic(cfg, False)
        for case in cfg['workloads']:
            current = dict(cfg, requests=case['requests'])
            local = case['requests'] // 2
            root = run / arm / 'workloads' / case['id']
            root.mkdir(parents=True, exist_ok=True)
            Path(cfg['expert_workload_path']).write_text(case['id'] + '\n')
            byte(cfg['fixed_cohort_size'], case['requests'])
            byte(cfg['transfer_audit_control'], ord('0'))
            switch(cfg, 0)
            health()
            response = base.h.http_json(cfg['port'], '/set_internal_state', key,
                                       dict(server_args=dict(pp_max_micro_batch_size=local)), timeout=30)
            assert not isinstance(response, dict) or response.get('success') is not False
            base.h.save(root / 'server_info.json', base.h.safe_server_info(
                base.h.http_json(cfg['port'], '/get_server_info', key, timeout=30)))
            tokens = prompts(current, case)
            pref = [base.packet(current, arm, t, 1, i) for i, t in enumerate(tokens)]
            dec = [base.packet(current, arm, t + [198], 128, i) for i, t in enumerate(tokens)]
            base.h.save(root / 'protocol.json', dict(**case, cached_context_per_request=4096,
                new_input_tokens_per_request=1, output_tokens_per_request=128, target_local_decode_tokens=local,
                pp_slots=2, no_cpu_kv_restore=True, profiler_off_formal=True,
                prompt_sha256=[hashlib.sha256(str(t).encode()).hexdigest() for t in tokens],
                compute_kernels_unchanged=True, counterbalanced_policies=cfg['gap_policies']))

            def check(group, count=None, cached=4096, output=128, reference=None):
                assert len(group['requests']) == (case['requests'] if count is None else count)
                for row in group['requests']:
                    assert row['cached_tokens'] == cached and row['output_tokens'] == output
                    assert len(row['output_ids']) == output
                    assert [e['completion_tokens'] for e in row['events']] == list(range(1, output + 1))
                if reference is not None:
                    assert [r['output_ids'] for r in group['requests']] == reference, 'PP optimization changed full output sequences'

            health()
            base.flush_cache(cfg, key)
            check(base.cohort(current, key, pref), cached=0, output=1)
            counts = sorted(set([2, 4, 6, 8, local, case['requests']])) + [case['requests']] * 2
            for count in counts:
                health()
                group = base.cohort(current, key, dec[:min(count, case['requests'])])
                check(group, count=min(count, case['requests']))
            reference = [r['output_ids'] for r in group['requests']]
            base.h.save(root / 'native_output_reference.json', dict(performance_sample=False, output_ids=reference))
            for policy in cfg['gap_policies']:
                switch(cfg, policy)
                for _ in range(3):
                    health()
                    check(base.cohort(current, key, dec), reference=reference)
                    states(cfg, key, policy)
            base.h.save(root / 'numerical_equivalence.json', dict(passed=True, exact_full_output_ids=True,
                output_tokens=128, native_reference_policy=0, policies=cfg['gap_policies']))
            values = {p: [] for p in cfg['gap_policies']}
            for repeat in range(cfg['repeats']):
                order = cfg['gap_policies'][::1 if repeat % 2 == 0 else -1]
                for policy in order:
                    health()
                    switch(cfg, policy)
                    before = states(cfg, key, policy)
                    base.progress(run, cfg, f'timing_{case["id"]}_policy{policy}', repeat=repeat + 1)
                    group = base.cohort(current, key, dec)
                    check(group, reference=reference)
                    after = states(cfg, key, policy)
                    counters = ('echoes_skipped', 'produced', 'retired', 'fast_sent', 'fast_received', 'ahead_produced', 'ahead_consumed')
                    deltas = [{k: a[k] - b[k] for k in counters} for b, a in zip(before, after)]
                    if policy == 10:
                        assert sum(d['echoes_skipped'] for d in deltas) > 0
                        assert sum(d['produced'] for d in deltas) > 0
                        assert all(d['produced'] == d['retired'] for d in deltas)
                    group['scheduling_receipt'] = dict(policy=policy, states=after, counter_deltas=deltas)
                    values[policy].append(group)
                    folder = root / f'p{policy}'
                    folder.mkdir(exist_ok=True)
                    base.h.save(folder / 'metrics.decode.json', base.summarize(values[policy], cfg['protocol']))
            for policy in cfg.get('profile_policies', cfg['gap_policies']):
                health()
                switch(cfg, policy)
                check(base.cohort(current, key, dec), reference=reference)
                directory = root / f'p{policy}' / 'traces' / 'decode'
                directory.mkdir(parents=True, exist_ok=True)
                base.profile_action(cfg, key, '/start_profile', dict(output_dir=str(directory),
                    num_steps=cfg['profile_steps'], activities=['CPU', 'GPU'], record_shapes=True,
                    with_stack=False, profile_id=f'pp2-{case["id"]}-p{policy}',
                    merge_profiles=False, detailed_annotations=True))
                group = base.cohort(current, key, dec)
                check(group, reference=reference)
                base.h.save(directory / 'cohort.json', dict(performance_sample=False, result=group))
                files = base.await_exports(directory, health, timeout=240)
                assert len(files) == 2
                base.h.save(directory / 'manifest.json', dict(files=files, performance_sample=False,
                    target_local_tokens=local, selection='Align sequence-length sums across EP/PP, not Graph indices'))
                states(cfg, key, policy)
            switch(cfg, 0)
            byte(cfg['fixed_cohort_control'], 0)
            base.h.save(root / 'completed.json', dict(passed=True, formal_rounds_per_policy=cfg['repeats'],
                exact_full_output_ids=True, policies=cfg['gap_policies']))
        base.h.save(run / arm / 'completed.json', dict(passed=True, workloads=len(cfg['workloads'])))

    def progress(run, cfg, stage, **extra):
        base.h.save(run / 'task/progress.json', dict(experiment_id=cfg['id'], task_id='comparison',
            timestamp=time.time(), stage=stage,
            valid_points=len(list(run.glob('pp2/workloads/*/p*/metrics.decode.json'))),
            expected=cfg['expected_points'], **extra))

    base.run_workloads = workloads
    base.progress = progress
