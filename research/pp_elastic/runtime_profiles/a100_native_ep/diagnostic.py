"""Matched EP2 backend diagnosis, with native GPU prefixes and separate profiles."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import threading
import time


def byte(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_NOFOLLOW)
    try:
        os.pwrite(fd, bytes([value]), 0)
    finally:
        os.close(fd)


def install(base):
    original_environment = base.environment
    epoch = [0]

    def args(cfg, arm):
        row = next(r for r in cfg['layouts'] if r['id'] == arm)
        result = [
            '--model-path', cfg['model_path'], '--host', '127.0.0.1', '--port', str(cfg['port']),
            '--dist-init-addr', f"127.0.0.1:{cfg['dist_port']}",
            '--tp-size', '2', '--ep-size', '2', '--dp-size', '2', '--pp-size', '1',
            '--enable-dp-attention', '--dtype', 'bfloat16', '--kv-cache-dtype', 'bfloat16',
            '--context-length', '8192', '--mem-fraction-static', '0.90', '--page-size', '64',
            '--max-total-tokens', '196608', '--max-running-requests', '32',
            '--max-mamba-cache-size', '168', '--mamba-max-states-per-path', '1',
            '--mamba-ssm-dtype', 'float32', '--chunked-prefill-size', '8192',
            '--max-prefill-tokens', '8192', '--attention-backend', 'triton',
            '--linear-attn-backend', 'triton', '--moe-runner-backend', 'triton',
            '--moe-a2a-backend', row['backend'], '--disable-cuda-graph',
            '--cuda-graph-backend-prefill', 'disabled', '--disable-overlap-schedule',
            '--disable-custom-all-reduce', '--skip-server-warmup', '--enable-cache-report',
            '--json-model-override-args', '{"language_model_only":true}',
            '--enable-hierarchical-cache', '--hicache-size', '64',
            '--hicache-host-memory-mode', 'cache', '--hicache-write-policy', 'write_through',
            '--hicache-io-backend', 'kernel', '--hicache-mem-layout', 'page_first',
        ]
        if row['backend'] == 'deepep':
            result += ['--deepep-mode', 'normal', '--deepep-dispatcher-output-dtype', 'bf16']
        result.remove('--disable-cuda-graph')
        result += ['--cuda-graph-bs-decode', '16']
        return result

    def environment(cfg, arm, inherited):
        env = original_environment(cfg, arm, inherited)
        env.update(
            EP2_PP2_ARM='ep2', EP2_PP2_RUN=str(Path(cfg['code_root']).parent / arm),
            PP_FIXED_COHORT_CONTROL=cfg['fixed_cohort_control'],
            PP_FIXED_COHORT_SIZE=cfg['fixed_cohort_size'],
            DEEPEP_LL_NVLINK_PAIR_ONLY='0', EP_STEADY_HARDWARE='A100',
        )
        return env

    def packet(cfg, arm, tokens, output, index):
        return dict(input_ids=tokens, sampling_params=dict(temperature=0, max_new_tokens=output,
                    ignore_eos=True), stream=True, data_parallel_rank=index % 2)

    def cohort(cfg, key, payloads):
        epoch[0] = epoch[0] % 254 + 1
        byte(cfg['fixed_cohort_control'], epoch[0])
        gate = threading.Barrier(len(payloads))
        with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
            futures = [pool.submit(base.stream_one, cfg, key, p, gate) for p in payloads]
            rows = [f.result() for f in futures]
        info = base.h.http_json(cfg['port'], '/get_server_info', key, timeout=30)
        receipts = [s['experiment_prefill_ep_receipt'] for s in info['internal_states']
                    if 'experiment_prefill_ep_receipt' in s]
        assert len(receipts) == 2 and {s['dp_rank'] for s in receipts} == {0, 1}
        for s in receipts:
            assert s['epoch'] == epoch[0]
            assert s['ordered_markers'] == list(range(5000 + s['dp_rank'], 5000 + len(rows), 2))
        duration = max(r['finished'] for r in rows) - min(r['started'] for r in rows)
        return dict(requests=rows, makespan_s=duration,
                    output_tokens_per_s=sum(r['output_tokens'] for r in rows) / duration,
                    input_tokens_per_s=sum(len(p['input_ids']) for p in payloads) / duration,
                    admission_receipts=receipts)

    def progress(run, cfg, stage, **extra):
        base.h.save(run / 'task/progress.json', dict(
            experiment_id=cfg['id'], task_id='comparison', stage=stage,
            timestamp=time.time(), controller_pid=os.getpid(),
            valid_points=len(list(run.glob('*/workloads/*/completed.json'))),
            expected=cfg['expected_points'], **extra))

    def workloads(run, cfg, arm, key, health):
        from previous_workloads import prompts
        for case in cfg['workloads']:
            current = dict(cfg, requests=case['requests'])
            directory = run / arm / 'workloads' / case['id']
            directory.mkdir(parents=True, exist_ok=True)
            byte(cfg['fixed_cohort_size'], case['requests'])
            tokens = prompts(current, case)
            pref = [packet(current, arm, t, 1, i) for i, t in enumerate(tokens)]
            dec = [packet(current, arm, t + [198], cfg['output_tokens'], i)
                   for i, t in enumerate(tokens)]
            base.h.save(directory / 'protocol.json', dict(
                **case, cached_context_per_request=4096, new_input_tokens_per_request=1,
                output_tokens_per_request=cfg['output_tokens'], hicache_enabled=True,
                prefix_location='native_gpu_hot', cpu_checkpoint_replay=False,
                physical_gpus=[0, 1], formal_profiler=False, pp=1, ep=2, attention_tp=1,
                attention_dp=2, prompt_sha256=[hashlib.sha256(str(t).encode()).hexdigest() for t in tokens]))

            def verify(group, prepared=False):
                assert len(group['requests']) == case['requests']
                for r in group['requests']:
                    assert r['output_tokens'] == (1 if prepared else cfg['output_tokens'])
                    assert r['prompt_tokens'] == (4096 if prepared else 4097)
                    assert r['cached_tokens'] == (0 if prepared else 4096), r['cached_tokens']
                    assert len(r.get('output_ids') or []) == r['output_tokens']

            health()
            base.flush_cache(cfg, key)
            prime = cohort(current, key, pref)
            verify(prime, prepared=True)
            base.h.save(directory / 'prefix_preparation.json', dict(performance_sample=False, result=prime))
            for i in range(cfg['warmups']):
                progress(run, cfg, f'warmup_{arm}_{case["id"]}', repeat=i + 1)
                health()
                verify(cohort(current, key, dec))
            formal = []
            for i in range(cfg['repeats']):
                health()
                progress(run, cfg, f'timing_{arm}_{case["id"]}', repeat=i + 1)
                group = cohort(current, key, dec)
                verify(group)
                formal.append(group)
                base.h.save(directory / f'cohort.{i}.json', dict(performance_sample=True, result=group))
            base.h.save(directory / 'metrics.decode.json', base.summarize(formal, cfg['protocol']))
            info = base.h.http_json(cfg['port'], '/get_server_info', key, timeout=30)
            base.h.save(directory / 'server_info.json', base.h.safe_server_info(info))
            # Only B32 has a separate short profile. It is never a timing sample.
            if case['requests'] == 32 and cfg.get('profile_enabled', True):
                traces = directory / 'traces'
                traces.mkdir()
                health()
                progress(run, cfg, f'profile_{arm}_{case["id"]}')
                base.profile_action(cfg, key, '/start_profile', dict(
                    output_dir=str(traces), num_steps=16, activities=['CPU', 'GPU'],
                    record_shapes=True, with_stack=False, profile_id=arm,
                    merge_profiles=False, detailed_annotations=True))
                group = cohort(current, key, dec)
                verify(group)
                exports = base.await_exports(traces, health, timeout=240)
                assert len(exports) == 2
                base.h.save(traces / 'manifest.json', dict(performance_sample=False, files=exports,
                    drop_initial_forwards=4, selected_forwards=8))
            base.h.save(directory / 'completed.json', dict(passed=True, formal_cohorts=len(formal)))
            byte(cfg['fixed_cohort_control'], 0)
        base.h.save(run / arm / 'completed.json', dict(passed=True, workloads=len(cfg['workloads'])))

    base.environment, base.engine_args = environment, args
    base.packet, base.cohort = packet, cohort
    base.progress, base.run_workloads = progress, workloads
