"""Longer native Decode profiles; optional eager diagnostic kept separate.

Performance rounds never enable the profiler or eager diagnostic. Every phase
starts from the same model-produced 4096-token GPU prefix. No synthetic KV.
"""
from pathlib import Path
import hashlib
import os
import statistics


def make_tokens(cfg):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
    rows = []
    for index in range(cfg['requests']):
        head = [5000 + index] + tokenizer.encode(f'Document {index}. ', add_special_tokens=False)
        body = tokenizer.encode('A distributed service keeps a record of independent requests. '
            'Explain the computation and communication clearly.\n', add_special_tokens=False)
        row = (head + body * (cfg['prefix_tokens'] // len(body) + 2))[:cfg['prefix_tokens']]
        if len(row) != cfg['prefix_tokens']:
            raise RuntimeError('Frozen prompt length mismatch')
        rows.append(row)
    return rows


def set_diagnostic(cfg, enabled):
    path = cfg.get('diagnostic_eager_policy_path')
    if not path:
        if enabled:
            raise RuntimeError('No eager diagnostic is configured')
        return
    fd = os.open(path, os.O_WRONLY | os.O_NOFOLLOW)
    try:
        os.pwrite(fd, b'1' if enabled else b'0', 0)
    finally:
        os.close(fd)


def run_workloads(base, run, cfg, arm, key, health):
    tokens = make_tokens(cfg)
    pref = [base.packet(cfg, arm, t, 1, i) for i, t in enumerate(tokens)]
    decode = [base.packet(cfg, arm, t + [198], cfg['output_tokens'], i)
              for i, t in enumerate(tokens)]
    base.h.save(run / arm / 'prompt_fingerprints.json', dict(
        requests=len(tokens), prefix_tokens=cfg['prefix_tokens'],
        sha256=[hashlib.sha256(bytes(str(t), 'utf-8')).hexdigest() for t in tokens]))

    def cached(result):
        values = [r['cached_tokens'] for r in result['requests']]
        if values != [cfg['prefix_tokens']] * len(result['requests']):
            raise RuntimeError('Expected exact GPU prefix hits: ' + str(values))

    set_diagnostic(cfg, False)
    health()
    base.flush_cache(cfg, key)
    prime = base.cohort(cfg, key, pref)
    base.h.save(run / arm / 'prefix_preparation.json', dict(
        measured=False, real_model_prefix=True, cpu_kv_restore=False,
        result=prime, cached_context_per_request=cfg['prefix_tokens']))
    # Cover cached one-token Extend and Decode local shapes 1,2,3,4 before
    # full cohorts. An HTTP arrival grouping must not introduce first-use JIT
    # into a formal round. The final two full warmups retain output length128.
    for count in [2, 4, 6, 8, 8, 8]:
        health()
        selected = decode[:count]
        warm = base.cohort(cfg, key, selected)
        cached(warm)
    rounds = []
    for repeat in range(cfg['repeats']):
        base.progress(run, cfg, 'native_timing_' + arm, repeat=repeat + 1)
        health()
        result = base.cohort(cfg, key, decode)
        cached(result)
        rounds.append(result)
        base.h.save(run / arm / 'decode.partial.json', dict(rounds=rounds))
    measured = base.summarize(rounds,
        '8 requests; each 4096 real GPU-cached tokens + 1 new input -> 128 outputs')
    measured.update(execution='native', diagnostic_eager=False,
        graph=cfg['graph'], gpu_pair=[0, 1])
    base.h.save(run / arm / 'metrics.decode.json', measured)

    modes = [('native', cfg['profile_repeats'])]
    if cfg['hardware'] == 'H100':
        modes.append(('eager_diagnostic', cfg['diagnostic_eager_repeats']))
    try:
        for mode, count in modes:
            set_diagnostic(cfg, mode == 'eager_diagnostic')
            # Warm the selected native execution path outside all profile windows.
            cached(base.cohort(cfg, key, decode))
            for repeat in range(count):
                health()
                base.progress(run, cfg, 'profile_' + arm + '_' + mode, repeat=repeat + 1)
                directory = run / arm / 'traces' / f'{mode}_r{repeat + 1}'
                directory.mkdir(parents=True)
                base.profile_action(cfg, key, '/start_profile', dict(
                    output_dir=str(directory), num_steps=cfg['profile_steps'],
                    activities=['CPU', 'GPU'], record_shapes=True, with_stack=False,
                    profile_id=f'{arm}-{mode}-r{repeat + 1}', merge_profiles=False,
                    detailed_annotations=True))
                result = base.cohort(cfg, key, decode)
                cached(result)
                base.h.save(directory / 'cohort.json', dict(performance_sample=False,
                    execution=mode, result=result))
                files = base.await_exports(directory, health, timeout=240)
                if len(files) != 2:
                    raise RuntimeError('Every profile must export both actual ranks')
                base.h.save(directory / 'manifest.json', dict(performance_sample=False,
                    execution=mode, files=files, requested_forward_steps=cfg['profile_steps'],
                    initial_extend_excluded_in_analysis=True,
                    target_selected_decode_forwards_per_rank=cfg['selected_forwards']))
    finally:
        set_diagnostic(cfg, False)
    base.h.save(run / arm / 'completed.json', dict(passed=True, formal_rounds=len(rounds),
        native_profiles=cfg['profile_repeats'],
        eager_diagnostic_profiles=cfg['diagnostic_eager_repeats'] if cfg['hardware'] == 'H100' else 0))


def install(base):
    original_args = base.engine_args
    original_environment = base.environment

    def engine_args(cfg, arm):
        if cfg['hardware'] == 'H100':
            import hopper_reference
            return hopper_reference.engine_args(cfg, arm)
        return original_args(cfg, arm)

    def environment(cfg, arm, inherited):
        result = original_environment(cfg, arm, inherited)
        result['EP_STEADY_HARDWARE'] = cfg['hardware']
        if cfg['hardware'] == 'H100':
            result['PP_READY_POLICY_PATH'] = cfg['policy_path']
            result['STEADY_EAGER_POLICY_PATH'] = cfg['diagnostic_eager_policy_path']
        return result

    base.engine_args = engine_args
    base.environment = environment
    base.run_workloads = lambda *args: run_workloads(base, *args)
