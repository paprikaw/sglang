"""Freeze the whole-node H100 comparison; CPU-only preparation."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
A100 = ROOT.parent / 'ep2_pp2_qwen_a100_20261024'
sys.path.insert(0, str(A100))
import prepare as base


def build(run, node, reuse_bank=False):
    result = base.build(run, node, ('ep2', 'pp2'))
    run = Path(run).absolute()
    cfg = json.loads((run / 'config.json').read_text())
    spec = json.loads((run / 'dashboard_spec.json').read_text())
    code = run / 'code'
    # Replace the copied A100-only bridge BEFORE freezing or starting Guard.
    for source, name in [(A100 / 'runner.py', 'base_runner.py'),
                         (ROOT / 'runner.py', 'runner.py'),
                         (ROOT / 'protocol.py', 'protocol.py'),
                         (ROOT / 'runtime_adapter.py', 'runtime_adapter.py'),
                         (ROOT.parent / 'qwen_pp_ready_send_20261025/pp_ready_send.py', 'pp_ready_send.py'),
                         (ROOT.parent / 'qwen_h100_capacity_20261017_r5/runtime_setup.py', 'runtime_setup.py')]:
        shutil.copyfile(source, code / name)
        os.chmod(code / name, 0o600)
    native = base.old.DEEPEP_NATIVE
    if base.old.sha(native) != base.old.DEEPEP_NATIVE_SHA256:
        raise RuntimeError('The qualified Hopper DeepEP extension changed')
    wrapper = base.old.DEEPEP_ROOT
    launch_env = base.old.a100_launch_env(base.old.PYTHON)
    launch_env.update(MAX_JOBS='4', TORCH_CUDA_ARCH_LIST='9.0',
        SGLANG_ENABLE_JIT_DEEPGEMM='1', SGLANG_JIT_DEEPGEMM_PRECOMPILE='0',
        # AUTO uses normal mode for Prefill. Only Decode uses the LL buffer;
        # its actual per-DP batch never exceeds eight rows in this protocol.
        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK='16',
        NCCL_IB_HCA='mlx5_bond_0', NCCL_SOCKET_IFNAME='bond0.3027',
        GLOO_SOCKET_IFNAME='bond0.3027', NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME='bond0.3027',
        NVSHMEM_HCA_LIST='mlx5_bond_0:1', NVSHMEM_IB_ADDR_FAMILY='AF_INET',
        NVSHMEM_IB_ADDR_RANGE='172.26.92.0/22',
        NVSHMEM_DISABLE_P2P='0', NVSHMEM_IBGDA_NIC_HANDLER='cpu_host_memory',
        NVSHMEM_QP_DEPTH='4096', DEEPEP_AH_ZERO_STATIC_RATE='1',
        LD_PRELOAD=str(base.old.VERBS_SHIM))
    for row in cfg['layouts']:
        row['gpu_groups'] = [[0, 1] if row['id'] == 'ep2' else [2, 3]]
    cfg.update(launch_env=launch_env, deepep_native=str(native), deepep_wrapper=str(wrapper),
        port=19631, dist_port=19641, parallel_pairs=True,
        attention_backend='fa3', moe_runner='deep_gemm', graph=True,
        overlap_schedule=True, pp_loop_slots=2, max_global_requests=16,
        precision='Native Hopper FP8 W8A8/DeepGEMM; BF16 KV and residuals; FP32 GDN states',
        protocol='Moderate: cold Prefill4096 and cached4096 Decode1->128; '
                 'near capacity: cached-context Prefill512 and Decode1->128, phases separated; '
                 'native model-produced GPU radix/GDN cache, no timed preparation or CPU restore; '
                 'actual capacity after native DeepEP and Graph allocation; 2 equal PP slots',
        policy_path=f'/tmp/{run.name}-pp-policy', startup_timeout_s=1800,
        stage_timeout_s=1800, repeats=3)
    if reuse_bank:
        cfg['model_cache']['on_miss'] = 'fail'
        spec['model_cache']['on_miss'] = 'fail'
    (run / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    os.chmod(run / 'config.json', 0o600)
    # Recompute descriptors after all private code/config changes. No completed
    # or running frozen run is ever modified by this preparer.
    inputs = [r for r in spec['frozen_inputs'] if not Path(r['path']).is_relative_to(code)
              and r['path'] != str(run / 'config.json')
              and not r['path'].endswith('deep_ep_cpp.cpython-312-x86_64-linux-gnu.so')
              and '/deep_ep/' not in r['path']]
    inputs += [base.old.descriptor(p) for p in code.glob('*.py')]
    inputs += [base.old.descriptor(p) for p in (wrapper / 'deep_ep').rglob('*.py')]
    inputs += [base.old.descriptor(native), base.old.descriptor(base.old.VERBS_SHIM),
               base.old.descriptor(run / 'config.json')]
    for name in ('managers/scheduler_pp_mixin.py', 'layers/quantization/fp8.py',
                 'layers/moe/moe_runner/deep_gemm.py'):
        inputs.append(base.old.descriptor(base.old.ENGINE / 'python/sglang/srt' / name))
    spec.update(title='H100 Qwen122B: parallel EP2 versus PP2, moderate and near-capacity',
                port=19630, frozen_inputs=list({r['path']: r for r in inputs}.values()))
    spec['tasks'][0].update(label='EP2 GPU0/1 and PP2 GPU2/3; concurrent', expected=14)
    base.old.validate_contract(spec)
    (run / 'dashboard_spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    os.chmod(run / 'dashboard_spec.json', 0o600)
    # This is the same reviewed Qwen-capable Guard bundle used by A100 runs,
    # with the current shared registry and background handoff unchanged.
    result['command'] = [str(base.old.PYTHON), str(base.old.SCRIPTS / 'dashboard_guard.py'),
        'run', '--spec', str(run / 'dashboard_spec.json'), '--exit-after-run', '--',
        str(base.old.PYTHON), str(code / 'runner.py'), '--run', str(run)]
    (run / 'preparation.json').write_text(json.dumps(dict(command=result['command'],
        required_environment=base.old.shared_environment(), node=node, job_id=cfg['job_id']), indent=2))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--node', default='spartan-gpgpu178')
    ap.add_argument('--reuse-bank', action='store_true')
    args = ap.parse_args()
    print(json.dumps(build(args.run, args.node, args.reuse_bank), indent=2))
