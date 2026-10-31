"""Same-process A/B/B/A Prefill and bounded Decode regression check."""
import hashlib
import json
from pathlib import Path
import time

from pp_ready_send import set_policy


def run(api, run, cfg, arm, key, health):
    if arm != 'pp2':
        raise RuntimeError('This optimization benchmark only runs PP2')
    root = run / arm
    policy = cfg['launch_env']['PP_READY_POLICY_PATH']
    tokens = api.make_tokens(cfg)
    pref = [api.packet(cfg, arm, t, 1, i) for i, t in enumerate(tokens)]
    dec = [api.packet(cfg, arm, t + [198], cfg['output_tokens'], i) for i, t in enumerate(tokens)]
    reference = None
    rounds = {'baseline': [], 'ready': []}

    def select(name):
        health()
        api.flush_cache(cfg, key)  # acknowledged idle boundary, outside timing
        set_policy(policy, name == 'ready')

    def outputs(group):
        return [r['output_ids'] for r in group['requests']]

    def verdict(groups):
        aa = all(outputs(g) == outputs(groups['baseline'][0]) for g in groups['baseline'])
        bb = all(outputs(g) == outputs(groups['ready'][0]) for g in groups['ready'])
        cross = all(outputs(g) == outputs(groups['baseline'][0]) for gs in groups.values() for g in gs)
        return dict(passed=cross, original_AA_stable=aa, candidate_BB_stable=bb,
            verdict='passed' if cross else 'rejected' if aa else 'inconclusive_original_unstable',
            output_ids={k:[outputs(g) for g in gs] for k,gs in groups.items()})

    # Save cold/warm observations before adjudication. Both policies receive two
    # warm cohorts, and no first-run comparison is silently treated as formal.
    warm_groups = {'baseline': [], 'ready': []}
    for name in ('baseline', 'baseline', 'ready', 'ready'):
        select(name)
        warm = api.cohort(cfg, key, pref)
        warm_groups[name].append(warm)
        api.h.save(root / 'prefill.warmup.json', dict(groups=warm_groups,
            performance_sample=False))
    api.h.save(root / 'correctness.warmup.json', verdict(warm_groups))
    blocks = []
    for index, name in enumerate(cfg['policy_order']):
        block = []
        for repeat in range(cfg['repeats']):
            select(name)
            measured = api.cohort(cfg, key, pref)
            block.append(measured)
            rounds[name].append(measured)
            api.h.save(root / f'prefill.block{index:02d}.json',
                       dict(policy=name, rounds=block, profiler_enabled=False))
        blocks.append(dict(policy=name, rounds=block))
    prefill_check = verdict(rounds)
    api.h.save(root / 'correctness.prefill.json', dict(**prefill_check,
        prefill_requests_checked=sum(len(g['requests']) for rr in rounds.values() for g in rr),
        same_loaded_model=True, comparison='exact generated token IDs after both policies warmed'))
    for name in rounds:
        result = api.summarize(rounds[name], '8 requests, zero cached context, 4096 new input tokens each, 1 output')
        result.update(policy=name, same_loaded_model=True, outputs_equal=prefill_check['passed'],
                      valid=prefill_check['passed'], diagnostic_only=not prefill_check['passed'],
                      correctness_verdict=prefill_check['verdict'], pp_loop_slots=cfg['pp_loop_slots'])
        api.h.save(root / f'metrics.prefill.{name}.json', result)
    api.h.save(root / 'prefill.blocks.json', dict(blocks=blocks))

    # One independent trace per policy, excluded from all performance metrics.
    for name in ('baseline', 'ready'):
        select(name)
        out = root / 'traces' / name
        out.mkdir(parents=True)
        api.profile_action(cfg, key, '/start_profile', dict(output_dir=str(out),
            activities=['CPU', 'GPU'], record_shapes=True, with_stack=False,
            profile_id='pp2-prefill-' + name, merge_profiles=False,
            detailed_annotations=True))
        group = api.cohort(cfg, key, pref)
        api.h.save(out / 'observation.json', dict(round=group, performance_sample=False,
            outputs_match_formal_reference=outputs(group)==outputs(rounds['baseline'][0])))
        api.profile_action(cfg, key, '/stop_profile')
        deadline = time.monotonic() + 90
        while len(list(out.glob('*.trace.json.gz'))) != 2:
            health()
            if time.monotonic() >= deadline:
                raise TimeoutError('Both PP stages must export a trace')
            time.sleep(1)
        api.h.save(out / 'manifest.json', dict(performance_sample=False, policy=name,
            files=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                   for p in sorted(out.glob('*.trace.json.gz'))]))
    # Independently re-prime real prefixes before EACH Decode cohort. Save every
    # observation before adjudicating, including mismatches. A/A controls test
    # whether exact text agreement is already unstable in the original path.
    select('baseline')
    api.cohort(cfg, key, pref)
    api.cohort(cfg, key, dec)
    decode_rounds = {'baseline': [], 'ready': []}
    hashes = {'baseline': [], 'ready': []}
    for index, name in enumerate(('baseline', 'ready', 'baseline', 'ready')):
        select(name)
        api.cohort(cfg, key, pref)
        measured = api.cohort(cfg, key, dec)
        if any((r['cached_tokens'] or 0) < cfg['prefix_tokens'] - 64 for r in measured['requests']):
            raise AssertionError('Decode prefix was not served from the native GPU cache')
        decode_rounds[name].append(measured)
        hashes[name].append(outputs(measured))
        api.h.save(root / f'decode.observation{index:02d}.json', dict(
            policy=name, round=measured, profiler_enabled=False,
            independently_reprimed_prefix=True))
    original_stable = hashes['baseline'][0] == hashes['baseline'][1]
    candidate_stable = hashes['ready'][0] == hashes['ready'][1]
    cross_equal = all(h == hashes['baseline'][0] for hh in hashes.values() for h in hh)
    verdict = 'passed' if cross_equal else 'rejected' if original_stable else 'inconclusive_original_unstable'
    for name in decode_rounds:
        result = api.summarize(decode_rounds[name], '8 requests, independently re-primed 4096 real cached tokens + 1 new input each, 128 outputs')
        result.update(policy=name, regression_check_only=True, outputs_equal=cross_equal,
            valid=cross_equal, diagnostic_only=not cross_equal, correctness_verdict=verdict)
        api.h.save(root / f'metrics.decode.{name}.json', result)
    api.h.save(root / 'correctness.decode.json', dict(passed=cross_equal,
        original_AA_stable=original_stable, candidate_BB_stable=candidate_stable,
        hashes=hashes, verdict=verdict, independently_reprimed_prefix=True))
