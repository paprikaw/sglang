"""Freeze one paired backend diagnosis; no GPU launch occurs here."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
OLD = WORK / 'experiments/a100_pp_ep_regression_20261027/runs/a10035_nvlink_r4'
E10 = WORK / 'experiments/agentx_a100_matrix_20261102/runs/a100_e10_r7'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    path.chmod(0o600)


def desc(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    run = args.run.absolute()
    if run.exists():
        raise RuntimeError('Use a fresh run directory')
    code = run / 'code'
    code.mkdir(parents=True)
    cfg = json.loads((E10 / 'config.json').read_text())['node_configs']['spartan-gpgpu069']
    for name in ['runner.py', 'process_helpers.py', 'experiment_plan.py', 'export_ready.py',
                 'previous_workloads.py', 'sampling.py', 'pp_schedule_opt.py', 'prefill_admission.py']:
        shutil.copy2(OLD / 'code' / name, code / name)
    for name in ['a100_runtime_adapter.py', 'runtime_setup.py']:
        shutil.copy2(E10 / 'code' / name, code / name)
    guard = Path(cfg['guard_scripts']).parent
    shutil.copy2(guard / 'scripts/artifact_visibility.py', code / 'artifact_visibility.py')
    shutil.copy2(ROOT / 'diagnostic.py', code / 'diagnostic.py')
    shutil.copy2(ROOT / 'prepare.py', code / 'prepare.py')
    p = code / 'runner.py'
    text = p.read_text()
    text = text.replace('import a100_benchmark\na100_benchmark.install(sys.modules[__name__])',
                        'import diagnostic\ndiagnostic.install(sys.modules[__name__])')
    text = text.replace("choices=['ep2','pp2']", "choices=['deepep_a','native','deepep_b']")
    text = text.replace("(run / arm / 'qualification').glob", "(run / arm / 'ep2/qualification').glob")
    # Qualify the adapter for native EP2 as well as the already supported EP4.
    p.write_text(text)
    p = code / 'a100_runtime_adapter.py'
    text = p.read_text().replace("self.ps.moe_ep_size == 4 and get_moe_a2a_backend().is_none()",
                                 "self.ps.moe_ep_size in (2, 4) and get_moe_a2a_backend().is_none()")
    text = text.replace("native_allgather = ps.moe_ep_size == 4 and all(",
                        "native_allgather = ps.moe_ep_size in (2, 4) and all(")
    text = text.replace('Native NCCL all-gather EP4; no DeepEP buffer',
                        'Native NCCL all-gather EP2/EP4; no DeepEP buffer')
    p.write_text(text)
    (code / 'runtime_adapter.py').write_text('''"""Shared E10 real-weight validation plus matched EP cohort admission."""
def install():
    import a100_runtime_adapter, runtime_setup, prefill_admission
    a100_runtime_adapter.install()
    runtime_setup.install()
    prefill_admission.install()
''')
    controls = {'fixed_cohort_control': ('fixed_cohort.bin', 0),
                'fixed_cohort_size': ('fixed_cohort_size.bin', 32)}
    for key, (name, value) in controls.items():
        cfg[key] = str(run / name)
        (run / name).write_bytes(bytes([value]))
        (run / name).chmod(0o600)
    cfg.update(id=run.name, code_root=str(code), admin_key_file=str(run / 'admin.key'),
               port=24541, dist_port=24542, arm_order=['deepep_a', 'native', 'deepep_b'],
               prefix_tokens=4096, output_tokens=64, warmups=1, repeats=3,
               expected_points=6, startup_timeout_s=600,
               protocol='E10 Qwen35B BF16 EP2, Attention DP2/TP1, A100069 NV12 GPU0/1; '
                        'DeepEP normal / native NCCL / DeepEP normal, same native Triton Expert; '
                        'B8 and B32 diverse, GPUhot4096+new1 ->64 outputs, HiCache64 decimal GB/rank, '
                        'one warmup and three profiler-off cohorts; separate B32 profile; no P/D handoff.',
               workloads=[dict(id=f'decode_b{b}_diverse', phase='decode', requests=b, family='diverse')
                          for b in [8, 32]])
    layouts = []
    for arm, backend in [('deepep_a', 'deepep'), ('native', 'none'), ('deepep_b', 'deepep')]:
        use = dict(id=f'{run.name}.{arm}', path=str(run / 'cache' / f'{arm}.log'),
                   expected_ranks=['TP0', 'TP1'])
        layouts.append(dict(id=arm, backend=backend, ep=2, tp=2, dp=2, pp=1,
                            replicas=1, gpu_groups=[[0, 1]], usage_logs=[use]))
        (run / arm / 'ep2').mkdir(parents=True)
    cfg['layouts'] = layouts
    cache = cfg['model_cache']
    spec_cache = json.loads((E10 / 'dashboard_spec.json').read_text())['model_cache']
    # Reuse exact existing canonical CPU weights on the still-running booked node.
    cache = {**spec_cache, 'node_specs': {}, 'usage_logs': {},
             'hicache_reserve_bytes_per_node': {'spartan-gpgpu069': 160 * 1024**3},
             'cpu_memory_budget': {'schema_version': 1, 'kv_banks_per_node': {'spartan-gpgpu069': []}}}
    (run / 'cache').mkdir()
    shutil.copy2(E10 / 'cache/spartan-gpgpu069.json', run / 'cache/node.json')
    cache['node_specs']['spartan-gpgpu069'] = desc(run / 'cache/node.json')
    cache['usage_logs']['spartan-gpgpu069'] = [r['usage_logs'][0] for r in layouts]
    cache['on_miss'] = 'fail'
    cfg['model_cache'] = cache
    (run / 'admin.key').write_text(secrets.token_urlsafe(32))
    (run / 'admin.key').chmod(0o600)
    save(run / 'config.json', cfg)
    old_spec = json.loads((E10 / 'dashboard_spec.json').read_text())
    # Preserve already pinned native/engine/model identities and the qualified Guard bundle.
    frozen = [x for x in old_spec['frozen_inputs']
              if not Path(x['path']).is_relative_to(E10)]
    frozen += [desc(p) for p in sorted(code.glob('*.py'))]
    frozen += [desc(run / 'config.json'), desc(run / 'cache/node.json')]
    spec = dict(old_spec, id=run.name, title='A100 EP2 backend diagnosis: DeepEP / NCCL / DeepEP',
                port=24540, runtime_dir=str(run / 'dashboard_guard'), max_active_nodes=1,
                model_cache=cache, frozen_inputs=frozen,
                tasks=[dict(id='comparison', label='EP2 GPU-hot Decode, same NVLink pair, ABA',
                            nodes=['spartan-gpgpu069'], root=str(run / 'task'),
                            controller_script=str(code / 'runner.py'), expected=6)])
    save(run / 'dashboard_spec.json', spec)
    save(run / 'task/progress.json', dict(experiment_id=run.name, task_id='comparison',
                                        stage='queued', valid_points=0, expected=6))
    save(run / 'launch_manifest.json', dict(id=run.name, engine_root=cfg['engine_root'],
        engine_commit=cfg['engine_commit'], engine_tag=cfg['engine_tag'], node=cfg['node'],
        job_id=cfg['job_id'], protocol=cfg['protocol'], private_adapters=[desc(p) for p in code.glob('*.py')],
        cache_qualification='Native GPU-hot prefix per fresh server; no matching E10 CPU KV checkpoint in registry',
        comparison='Backend execution paths include their native token layouts; same Triton math family, not bitwise kernel geometry',
        hypothesis='DeepEP normal host/dispatch overhead explains a material part of slow EP2',
        acceptance='All rank numerical checks, exact prefix/output count, actual DP cohort receipts, Guard/cache/cleanup; report ABA drift'))
    print(json.dumps(dict(run=str(run), guard=str(guard / 'scripts/dashboard_guard.py'), points=6)))


if __name__ == '__main__':
    main()
