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
        return [r['text_sha256'] for r in group['requests']]

    for name in ('baseline', 'ready'):
        select(name)
        warm = api.cohort(cfg, key, pref)
        if reference is None:
            reference = outputs(warm)
        if outputs(warm) != reference:
            raise AssertionError('Prefill outputs differ during policy warmup')
    blocks = []
    for index, name in enumerate(cfg['policy_order']):
        block = []
        for repeat in range(cfg['repeats']):
            select(name)
            measured = api.cohort(cfg, key, pref)
            if outputs(measured) != reference:
                raise AssertionError('Prefill outputs changed across send policies')
            block.append(measured)
            rounds[name].append(measured)
            api.h.save(root / f'prefill.block{index:02d}.json',
                       dict(policy=name, rounds=block, profiler_enabled=False))
        blocks.append(dict(policy=name, rounds=block))
    for name in rounds:
        result = api.summarize(rounds[name], '8 requests, zero cached context, 4096 new input tokens each, 1 output')
        result.update(policy=name, same_loaded_model=True, outputs_equal=True,
                      pp_loop_slots=cfg['pp_loop_slots'])
        api.h.save(root / f'metrics.prefill.{name}.json', result)
    api.h.save(root / 'prefill.blocks.json', dict(blocks=blocks))

    # This is a regression check, not a replacement three-round Decode result.
    select('baseline')
    api.cohort(cfg, key, pref)
    api.cohort(cfg, key, dec)  # warmup, native model-produced GPU prefixes
    decode_reference = None
    for name in ('baseline', 'ready'):
        health()
        set_policy(policy, name == 'ready')
        measured = api.cohort(cfg, key, dec)
        if any((r['cached_tokens'] or 0) < cfg['prefix_tokens'] - 64 for r in measured['requests']):
            raise AssertionError('Decode prefix was not served from the native GPU cache')
        if decode_reference is None:
            decode_reference = outputs(measured)
        if outputs(measured) != decode_reference:
            raise AssertionError('Decode output hashes changed across send policies')
        result = api.summarize([measured], '8 requests, 4096 real cached tokens + 1 new input each, 128 outputs')
        result.update(policy=name, regression_check_only=True, outputs_equal=True)
        api.h.save(root / f'metrics.decode.{name}.json', result)

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
        if outputs(group) != reference:
            raise AssertionError('Profiled Prefill outputs changed')
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
    api.h.save(root / 'correctness.json', dict(passed=True,
        prefill_requests_checked=sum(len(g['requests']) for rr in rounds.values() for g in rr),
        full_text_sha256_equal=True, prefix_model_produced=True,
        same_loaded_model=True, method='original math; only producer readiness before CPU metadata changes'))
