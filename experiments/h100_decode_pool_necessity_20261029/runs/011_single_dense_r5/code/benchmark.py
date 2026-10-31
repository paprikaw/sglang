"""Matched Prefill/Decode timing plus untimed real-routing diagnostics."""
import json
import os
from pathlib import Path
import time


def policy(cfg, byte):
    fd = os.open(cfg['fragment_policy_path'], os.O_WRONLY)
    try:
        os.pwrite(fd, byte, 0)
    finally:
        os.close(fd)


def workloads(base, run, cfg, arm, key, health):
    from sampling import make_tokens, set_diagnostic
    tokens = make_tokens(cfg)
    pref = [base.packet(cfg, arm, t, 1, i) for i, t in enumerate(tokens)]
    dec = [base.packet(cfg, arm, t + [198], cfg['output_tokens'], i) for i, t in enumerate(tokens)]
    if cfg.get('validation_only'):
        policy(cfg, b'0')
        set_diagnostic(cfg, False)
        base.flush_cache(cfg, key)
        base.cohort(cfg, key, pref)
        for count in [2,4,6,8,8]:
            health()
            base.cohort(cfg, key, dec[:count])
        policy(cfg, b'V')
        rounds=[]
        for repeat in range(3):
            health()
            value = base.h.http_json(cfg['port'], '/generate', key, dict(
                input_ids=[t+[198] for t in tokens],
                sampling_params=dict(temperature=0,max_new_tokens=8,ignore_eos=True),
                stream=False, return_logprob=True, logprob_start_len=cfg['prefix_tokens']+1,
                top_logprobs_num=10), timeout=120)
            if not isinstance(value,list) or len(value)!=cfg['requests']:
                raise RuntimeError('Expected one result per native batched input')
            rows=[]
            for result in value:
                meta=result['meta_info']
                if meta['completion_tokens']!=8 or meta['cached_tokens']!=cfg['prefix_tokens']:
                    raise RuntimeError('Batched validation changed output or prefix cache')
                rows.append(dict(text=result.get('text'), output_tokens=meta['completion_tokens'],
                    validation_logits={k:meta[k] for k in ('output_token_logprobs','output_top_logprobs')}))
            rounds.append(dict(requests=rows))
        base.h.save(run/arm/'validation.probabilities.json',dict(rounds=rounds,output_tokens=8,
            performance_sample=False, protocol='One native batched HTTP request, atomic scheduler input, local microbatch4'))
        base.h.save(run/arm/'metrics.validation.json',dict(valid=True,performance_sample=False,
            validated_requests=3*cfg['requests'],output_tokens_per_request=8))
        base.h.save(run/arm/'completed.json',dict(passed=True,validation_only=True))
        policy(cfg,b'0')
        return
    def check(group, cached):
        counts = [r.get('cached_tokens') or 0 for r in group['requests']]
        if cached and min(counts) < cfg['prefix_tokens'] - 64:
            raise RuntimeError('Decode lost its real GPU prefix: ' + str(counts))
        if not cached and max(counts) > 64:
            raise RuntimeError('Cold Prefill unexpectedly reused context: ' + str(counts))
    policy(cfg, b'0')
    set_diagnostic(cfg, False)
    base.flush_cache(cfg, key)
    base.cohort(cfg, key, pref)
    rounds = []
    for i in range(cfg['repeats']):
        health()
        base.flush_cache(cfg, key)
        group = base.cohort(cfg, key, pref)
        check(group, False)
        rounds.append(group)
        base.h.save(run / arm / 'prefill.partial.json', dict(rounds=rounds))
    base.h.save(run / arm / 'metrics.prefill.json', base.summarize(rounds, '8 requests; cached0 + new4096 -> output1'))
    base.flush_cache(cfg, key)
    base.cohort(cfg, key, pref)
    for count in [2, 4, 6, 8, 8, 8]:
        health()
        base.cohort(cfg, key, dec[:count])
    rounds = []
    for i in range(cfg['repeats']):
        health()
        group = base.cohort(cfg, key, dec)
        check(group, True)
        rounds.append(group)
        base.h.save(run / arm / 'decode.partial.json', dict(rounds=rounds))
    base.h.save(run / arm / 'metrics.decode.json', base.summarize(rounds, '8 requests; cached4096 + new1 -> output128'))
    # No timed run above contains routing copies, microbenchmarks or profiler.
    if arm in ('ep2','pp2'):
        set_diagnostic(cfg, True)
        policy(cfg, b'D')
        base.cohort(cfg, key, dec)
        policy(cfg, b'P')
        base.flush_cache(cfg, key)
        base.cohort(cfg, key, pref)
        policy(cfg, b'0')
        set_diagnostic(cfg, False)
    for phase, packets in [('prefill', pref), ('decode', dec)]:
        health()
        if phase == 'prefill':
            base.flush_cache(cfg, key)
        else:
            base.flush_cache(cfg, key)
            base.cohort(cfg, key, pref)
            base.cohort(cfg, key, dec)
        root = run / arm / 'traces' / phase
        root.mkdir(parents=True, exist_ok=True)
        base.profile_action(cfg, key, '/start_profile', dict(output_dir=str(root),
            num_steps=1000000 if phase == 'prefill' else cfg['profile_steps'],
            activities=['CPU', 'GPU'], record_shapes=True,
            profile_id=arm + '-' + phase, with_stack=False,
            detailed_annotations=True, merge_profiles=False))
        group = base.cohort(cfg, key, packets)
        if phase == 'prefill':
            base.profile_action(cfg, key, '/stop_profile')
        traces = base.await_exports(root, health, timeout=240)
        base.h.save(root / 'manifest.json', dict(phase=phase, execution='native',
            selected_forwards=cfg['selected_forwards'], files=traces,
            timestamp=time.time()))
    base.h.save(run / arm / 'completed.json', dict(passed=True, formal_rounds=cfg['repeats']))
    # Short output distributions are evaluated separately on the same inputs;
    # do not interpret divergent long greedy continuations as exact arithmetic.
    base.flush_cache(cfg, key)
    base.cohort(cfg, key, pref)
    validation = []
    policy(cfg, b'V')
    for repeat in range(3):
        health()
        packets = [dict(base.packet(cfg, arm, t + [198], 8, i),
                        return_logprob=True, logprob_start_len=len(t) + 1, top_logprobs_num=10)
                   for i, t in enumerate(tokens)]
        from fair_validation import atomic_validation
        validation.append(atomic_validation(base,cfg,arm,key,tokens))
    base.h.save(run / arm / 'validation.probabilities.json', dict(
        performance_sample=False, rounds=validation, output_tokens=8,
        protocol='Same eight cached prefixes; original output-token and top-10 log probabilities; compare only identical histories'))
    policy(cfg, b'0')
    base.progress(run, cfg, 'completed_' + arm)


def install(base):
    previous = base.environment
    def environment(cfg, arm, inherited):
        env = previous(cfg, arm, inherited)
        env.update(FRAGMENT_POLICY_PATH=cfg['fragment_policy_path'], FRAGMENT_EXPERT_MODE=cfg['expert_mode'])
        return env
    base.environment = environment
    base.run_workloads = lambda run, cfg, arm, key, health: workloads(base, run, cfg, arm, key, health)
    previous_progress=base.progress
    def progress(run,cfg,stage,**extra):
        if cfg.get('validation_only'):
            base.h.save(run/'task/progress.json',dict(experiment_id=cfg['id'],task_id='comparison',
                stage=stage,timestamp=time.time(),controller_pid=os.getpid(),
                valid_points=len(list(run.glob('*/metrics.validation.json'))),expected=len(cfg['layouts']),**extra))
        else:
            previous_progress(run,cfg,stage,**extra)
    base.progress=progress
