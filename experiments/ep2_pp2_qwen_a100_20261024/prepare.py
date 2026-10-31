"""Freeze an exact single-node EP2/PP2 comparison; CPU only, never launch."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'qwen_a100_capacity_20261019_r1'
sys.path.insert(0, str(OLD))
import prepare_run as old


def build(run, node, arms=('pp2', 'ep2')):
    jobs = old.booked_jobs()
    if node not in jobs:
        raise RuntimeError('Chosen node is not in a current RUNNING booking')
    inputs = old.candidate_inputs() + old.engine_inputs() + old.memory_policy_sources()
    model, model_info, model_inputs = old.model_identity('large')
    inputs += model_inputs
    run = Path(run).absolute()
    run.mkdir(mode=0o700, parents=True)
    (run / 'task').mkdir(mode=0o700)
    code = run / 'code'
    code.mkdir(mode=0o700)
    for source, name in [(ROOT / 'runner.py', 'runner.py'),
                         (ROOT / 'runtime_adapter.py', 'runtime_adapter.py'),
                         (OLD / 'fp8_marlin_adapter.py', 'fp8_marlin_adapter.py'),
                         (OLD / 'qualification.py', 'process_helpers.py'),
                         (OLD / 'experiment_plan.py', 'experiment_plan.py')]:
        shutil.copyfile(source, code / name)
        os.chmod(code / name, 0o600)
        inputs.append(old.descriptor(code / name))
    native = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/runtime/deepep_ce_ep8_admission_20261010_r1/candidate/python/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so')
    expected = 'c3405cad906f56aeaebfa4f947bdb09e1d5505248f146879ee9fdacc4bbb2da3'
    if old.sha(native) != expected:
        raise RuntimeError('The pinned A100 native extension changed')
    inputs.append(old.descriptor(native))
    wrapper = old.WORKSPACE / '.third_party/deepep-token-routing-ep4-20261004'
    for path in sorted((wrapper / 'deep_ep').rglob('*.py')):
        inputs.append(old.descriptor(path))
    cache = run / 'cache'
    cache.mkdir(mode=0o700)
    node_spec = dict(node=node, expected_job_id=jobs[node], engine_root=str(old.ENGINE),
        python=str(old.PYTHON), source_commit=old.COMMIT,
        socket_path=f'/tmp/sglang-canonical-weights-{os.getuid()}/cache.sock',
        cache_scheme='canonical_full_weight', support_profile=old.PROFILE,
        model_path=str(model), max_bytes=192 * old.GIB, headroom_bytes=32 * old.GIB)
    old.write_json(cache / 'node.json', node_spec)
    layouts = []
    for name, tp, ep, dp, pp, ranks in [
        ('ep2', 2, 2, 2, 1, ['TP0', 'TP1']),
        ('pp2', 1, 1, 1, 2, ['PP0 TP0', 'PP1 TP0'])]:
        if name not in arms:
            continue
        (run / name).mkdir(mode=0o700)
        use = dict(id=run.name + '.' + name, path=str(cache / f'{name}.log'), expected_ranks=ranks)
        old.create_usage_log(use['path'])
        layouts.append(dict(id=name, tp=tp, ep=ep, dp=dp, pp=pp,
                            gpu_groups=[[0, 1]], replicas=1, usage_logs=[use]))
    contract = dict(policy='required', provider='sglang_canonical_full_weight_v1',
        support_profile=old.PROFILE, on_miss='build',
        node_specs={node: old.descriptor(cache / 'node.json')},
        usage_logs={node: [row['usage_logs'][0] for row in layouts]},
        hicache_reserve_bytes_per_node={node: 0},
        cpu_memory_budget=dict(schema_version=1, kv_banks_per_node={node: []}))
    key = run / 'admin.key'
    key.write_text(secrets.token_hex(32) + '\n')
    os.chmod(key, 0o600)
    cfg = dict(id=run.name, node=node, job_id=jobs[node], engine_root=str(old.ENGINE),
        engine_commit=old.COMMIT, python=str(old.PYTHON), model_path=str(model),
        model_revision=model_info['revision'], guard_scripts=str(old.SCRIPTS),
        admin_key_file=str(key), code_root=str(code), layouts=layouts, model_cache=contract,
        launch_env=old.a100_launch_env(old.PYTHON, qualification_jit=True),
        deepep_wrapper=str(wrapper), deepep_native=str(native), port=19421,
        startup_timeout_s=1800, arm_order=list(arms), pp_async_batch_depth=0,
        pp_loop_slots=2, requests=8, prefix_tokens=4096, prefill_tokens=4096,
        output_tokens=128, repeats=3, attention_backend='triton',
        precision='E4M3 block-FP8 serialized weights; W8A16 Marlin; BF16 activations/KV',
        graph=False, overlap_schedule=False, prefix_cache='native GPU radix, model-produced',
        protocol='ample-capacity; no CPU KV restore or synthetic KV; unprofiled then profiler')
    old.write_json(run / 'config.json', cfg)
    inputs += [old.descriptor(cache / 'node.json'), old.descriptor(run / 'config.json')]
    spec = dict(schema_version=1, id=run.name, title='Qwen122B A100 EP2 vs PP2: full model',
        engine='sglang', protocol='task-events-v2', template='parallel-tasks-v5',
        features=['parallel-tasks','task-progress','task-details','gpu-tab','gpu-telemetry','guard','cpu-model-cache'],
        port=19420, max_active_nodes=1, gpu_lease_ttl_s=90, cleanup_wait_s=90,
        runtime_dir=str(run / 'dashboard_guard'), gpu_registry_dir=str(old.GPU_REGISTRY),
        sources=[dict(worktree=str(old.ENGINE), commit=old.COMMIT, tag=old.TAG)],
        model_cache=contract, frozen_inputs=list({r['path']: r for r in inputs}.values()),
        tasks=[dict(id='comparison', label=' then '.join(arms) + '; same NVLink GPUs',
                    root=str(run / 'task'), expected=2 * len(layouts), nodes=[node],
                    controller_script=str(code / 'runner.py'))])
    old.validate_contract(spec)
    old.write_json(run / 'dashboard_spec.json', spec)
    old.write_json(run / 'task/progress.json', dict(experiment_id=run.name, task_id='comparison',
        stage='queued', timestamp=time.time(), valid_points=0, expected=2 * len(layouts)))
    command = [str(old.PYTHON), str(old.SCRIPTS / 'dashboard_guard.py'), 'run', '--spec',
               str(run / 'dashboard_spec.json'), '--exit-after-run', '--',
               str(old.PYTHON), str(code / 'runner.py'), '--run', str(run)]
    old.write_json(run / 'preparation.json', dict(command=command,
        required_environment=old.shared_environment(), node=node, job_id=jobs[node]))
    return dict(prepared=True, run=str(run), node=node, job_id=jobs[node], command=command)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--node', default='spartan-gpgpu069')
    ap.add_argument('--arms', nargs='+', choices=['pp2', 'ep2'], default=['pp2', 'ep2'])
    a = ap.parse_args()
    print(json.dumps(build(a.run, a.node, a.arms), indent=2))
