"""Freeze an equal-four-GPU native EP comparison; preparation uses no GPU."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import secrets
import shutil

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
PREVIOUS = WORK / 'experiments/a100_ep2_backend_diagnosis_20261103/runs/ep2_native_graph_r1'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    path.chmod(0o600)


def source(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def prepare(run, overlap=True, order=None):
    assert not run.exists(), 'Use an independent new run; never overwrite measured inputs'
    run.mkdir(parents=True)
    shutil.copytree(PREVIOUS / 'code', run / 'code', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT / 'matched.py', run / 'code/matched.py')
    shutil.copy2(ROOT / 'prepare.py', run / 'code/prepare_matched.py')
    p = run / 'code/runner.py'
    text = p.read_text().replace('import diagnostic\ndiagnostic.install(sys.modules[__name__])',
                                 'import matched\nmatched.install(sys.modules[__name__])')
    text = text.replace("choices=['deepep_a','native','deepep_b']", "choices=['replicas','ep4']")
    text = text.replace("    args = ap.parse_args()", "    ap.add_argument('--instance', type=int, default=0)\n    args = ap.parse_args()")
    text = text.replace("        engine_entry(args.run, cfg, args.arm)",
                        "        cfg = matched.instance_config(cfg, args.arm, args.instance)\n        engine_entry(args.run, cfg, args.arm)")
    text = text.replace("cached_tokens=final['meta_info'].get('cached_tokens'),",
                        "cached_tokens=final['meta_info'].get('cached_tokens'), cached_tokens_details=final['meta_info'].get('cached_tokens_details'),")
    p.write_text(text)
    # Both instances admit the same global four-rank workload mapping. Actual
    # model math and scheduling remain native; only arrival ordering is held.
    p = run / 'code/prefill_admission.py'
    text = p.read_text().replace('5000 + dp_rank, 5000 + size, 2',
        '5000 + int(os.environ.get("MATCHED_RANK_OFFSET", "0")) + dp_rank, 5000 + size, 4')
    p.write_text(text)
    cfg = read(PREVIOUS / 'config.json')
    cfg.update(id=run.name, code_root=str(run / 'code'), admin_key_file=str(run / 'admin.key'),
               arm_order=order or ['replicas', 'ep4'], port=24571, dist_port=24572,
               expected_points=18, profile_enabled=False, overlap=overlap,
               warmups=1, repeats=3, batches=[8, 32, 64], prefix_tokens=4096,
               output_tokens=64, new_prefill_tokens=1024, startup_timeout_s=600,
               protocol='Qwen3.6-35B-A3B BF16, A100069 four GPUs; concurrent 2xEP2 versus EP4; '
               'native NCCL/DP-attention/Triton on both, Decode CUDA Graph bs2/8/16, '
               f'overlap_schedule={overlap}, same GPU rank/request mapping, HiCache64 decimal GB/rank; '
               'cold4096->1, cached4096+new1024->1, cached4096+new1->64; B8/32/64; '
               'one warmup plus three formal cohorts. Synthetic full-model phase isolation; no PD transfer.')
    for field, name in [('fixed_cohort_control', 'fixed_cohort.bin'), ('fixed_cohort_size', 'fixed_cohort_size.bin')]:
        cfg[field] = str(run / name)
        (run / name).write_bytes(b'\x00')
        (run / name).chmod(0o600)
    cfg['launch_env'] = {k:v for k,v in cfg['launch_env'].items()
                         if not (k.startswith(('PP_', 'BENCH_', 'MATRIX_', 'EP_')) or k == 'LD_PRELOAD')}
    cfg['launch_env'].update(PP_FIXED_COHORT_CONTROL=cfg['fixed_cohort_control'],
                             PP_FIXED_COHORT_SIZE=cfg['fixed_cohort_size'])
    cfg['layouts'] = []
    for arm, ep, groups in [('replicas', 2, [[0, 1], [2, 3]]), ('ep4', 4, [[0, 1, 2, 3]])]:
        uses = [dict(id=f'{run.name}.{arm}.i{i}', path=str(run / 'cache' / f'{arm}.i{i}.log'),
                     expected_ranks=[f'TP{r}' for r in range(ep)]) for i in range(len(groups))]
        cfg['layouts'].append(dict(id=arm, backend='none', ep=ep, tp=ep, dp=ep, pp=1,
                                   replicas=len(groups), gpu_groups=groups, usage_logs=uses))
        for i in range(len(groups)):
            (run / arm / f'i{i}' / 'ep2').mkdir(parents=True)
    cache = cfg['model_cache']
    cache['hicache_reserve_bytes_per_node'][cfg['node']] = 288 * 1024**3
    cache['usage_logs'][cfg['node']] = [u for row in cfg['layouts'] for u in row['usage_logs']]
    (run / 'cache').mkdir(exist_ok=True)
    shutil.copy2(PREVIOUS / 'cache/node.json', run / 'cache/node.json')
    cache['node_specs'][cfg['node']] = source(run / 'cache/node.json')
    (run / 'admin.key').write_text(secrets.token_urlsafe(32))
    (run / 'admin.key').chmod(0o600)
    save(run / 'config.json', cfg)
    spec = read(PREVIOUS / 'dashboard_spec.json')
    external = [x for x in spec['frozen_inputs'] if not Path(x['path']).is_relative_to(PREVIOUS)]
    frozen = [*external, source(run / 'config.json'), source(run / 'cache/node.json'),
              *[source(p) for p in sorted((run / 'code').glob('*.py'))]]
    spec.update(id=run.name, title='A100 four-GPU native 2xEP2 versus EP4: Prefill and Decode',
                port=24570, runtime_dir=str(run / 'dashboard_guard'), frozen_inputs=frozen,
                model_cache=cache, resource_priority='normal',
                tasks=[dict(id='comparison', label='4 GPU, matched P and D, 2xEP2 versus EP4',
                            nodes=[cfg['node']], root=str(run / 'task'),
                            controller_script=str(run / 'code/runner.py'), expected=18)])
    save(run / 'dashboard_spec.json', spec)
    save(run / 'task/progress.json', dict(experiment_id=run.name, task_id='comparison',
                                         stage='queued', valid_points=0, expected=18))
    save(run / 'launch_manifest.json', dict(id=run.name, protocol=cfg['protocol'],
        engine_root=cfg['engine_root'], engine_commit=cfg['engine_commit'], engine_tag=cfg['engine_tag'],
        node=cfg['node'], job_id=cfg['job_id'], frozen_inputs=frozen,
        acceptance='All8 rank startup qualifications, exact cohort/rank mapping, output count and cache split, '
                   'all18 cells, Guard/cache postcheck, owned cleanup and lease release',
        limits='This is not AgentX or a P-to-D handoff measurement; BF16 output trajectories may differ across layouts'))
    for p in (run / 'code').glob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
        p.chmod(0o600)
    assert len({x['path'] for x in frozen}) == len(frozen)
    assert all(source(Path(x['path'])) == x for x in frozen)
    return dict(run=str(run), points=18, concurrent_instances=[2, 1], overlap=overlap)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--no-overlap', action='store_true')
    args = ap.parse_args()
    print(json.dumps(prepare(args.run.absolute(), overlap=not args.no_overlap)))
