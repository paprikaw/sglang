"""CPU-only pinned helpers: actual v7 DeepEP transport, no model or GEMM."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
GUARD = Path('/home/bxb1/.codex/skills/llm-experiment-guard/scripts/dashboard_guard.py').resolve()
PYTHON = '/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python'
CUDA = '/apps/easybuild-2022/easybuild/software/Core/CUDA/12.8.0'
DEEP = WORKSPACE / '.third_party/deepep-token-routing-ep4-20261004'
NATIVE = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/runtime/glm47_ep8_tp8_ep4_tp4_c4_20261004_r1/native4/python/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so')
NATIVE_SHA = '9dc348878f7a3ad13fa8766a0d2738ca0d19ab116a3a5cc8e6e4e21b12988581'
SHIM = WORKSPACE / 'experiments/deepep_ah_diagnosis_h100_20260928/trace_verbs.so'
MODES = ('cpu_cuda_after_capture', 'cuda_only_after_capture', 'cuda_kernels_only_after_capture')
FLAGS = ('TEARDOWN_CUPTI', 'DISABLE_CUPTI_LAZY_REINIT')
TRANSPORT = dict(NVSHMEM_IBGDA_NIC_HANDLER='cpu_host_memory', NVSHMEM_QP_DEPTH='4096',
    NVSHMEM_DISABLE_P2P='0', DEEPEP_LL_NVLINK_PAIR_ONLY='1', DEEPEP_LL_PCIE_P2P_MAX_TOKENS='0',
    NVSHMEM_HCA_LIST='mlx5_bond_0:1', NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME='bond0.3027',
    NVSHMEM_IB_ADDR_FAMILY='AF_INET', NVSHMEM_IB_ADDR_RANGE='172.26.92.0/22',
    DEEPEP_AH_ZERO_STATIC_RATE='1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def environment(cfg, mode, run):
    if mode not in MODES:
        raise ValueError('Unknown frozen mode')
    env = {k:v for k,v in os.environ.items()
        if not k.startswith(('SGLANG_', 'VLLM_', 'NCCL_', 'NVSHMEM_', 'DEEPEP_', 'GLM_', 'HUMMING_'))
        and k not in (*FLAGS, 'PYTHONPATH', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES')}
    env.update(TRANSPORT)
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='1', CUDA_HOME=CUDA, CUDA_PATH=CUDA,
        PATH=f'{Path(cfg["python"]).parent}:{CUDA}/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
        LD_LIBRARY_PATH=f'/home/bxb1/conda/lib:{CUDA}/lib64:/apps/easybuild-2022/easybuild/software/Core/OpenSSL/1.1/lib',
        PYTHONPATH=os.pathsep.join(map(str, (ROOT/'runtime', DEEP, NATIVE.parent))),
        LD_PRELOAD=str(SHIM), GLOO_SOCKET_IFNAME='bond0.3027', NCCL_SOCKET_IFNAME='bond0.3027',
        NCCL_IB_HCA='mlx5_bond_0', NCCL_CUMEM_HOST_ENABLE='0', NCCL_DEBUG='WARN',
        TEARDOWN_CUPTI='0', DISABLE_CUPTI_LAZY_REINIT='1', CUDA_MODULE_LOADING='EAGER')
    return env


def routes(rank, local_tokens=12, world=8, experts=160, topk=8):
    """One assignment to every owner per token; expert counts differ at most one."""
    if (world, experts, topk, local_tokens) != (8, 160, 8, 12) or rank not in range(world):
        raise ValueError('Only the pinned real Verify geometry is admitted')
    local_experts = experts // world
    return [[owner * local_experts + ((rank * local_tokens + token + 3 * owner) % local_experts)
             for owner in range(world)] for token in range(local_tokens)]


def expert_counts():
    counts = [0] * 160
    for rank in range(8):
        for row in routes(rank):
            for expert in row:
                counts[expert] += 1
    return counts


def activity_config(mode, activity):
    if mode == MODES[0]:
        return [activity.CPU, activity.CUDA]
    if mode == MODES[1]:
        return [activity.CUDA]
    if mode == MODES[2]:
        return [{activity.CUDA: ['CONCURRENT_KERNEL']}]
    raise ValueError('Unknown activity mode')


def trace_counts(payload):
    events = payload.get('traceEvents', [])
    gpu = [e for e in events if e.get('cat') in ('kernel','cuda_kernel','gpu_memcpy','gpu_memset')]
    kernels = [e for e in gpu if e.get('cat') in ('kernel','cuda_kernel')]
    runtime = [e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver')]
    proofs = []
    for index in range(3):
        scopes = [e for e in events if e.get('cat') == 'user_annotation' and e.get('name') == f'stage-marker-proof:{index}']
        matched = []
        for scope in scopes:
            launches = [e for e in runtime if 'LaunchKernel' in e.get('name','') and
                (e.get('pid'),e.get('tid')) == (scope.get('pid'),scope.get('tid')) and
                scope['ts'] <= e['ts'] < scope['ts'] + scope.get('dur',0)]
            correlations = {e.get('args',{}).get('correlation') for e in launches}
            matched += [dict(name=e.get('name'),duration_us=e.get('dur')) for e in kernels
                if e.get('args',{}).get('correlation') in correlations]
        proofs.append(dict(index=index,scope_count=len(scopes),kernels=matched))
    names = sorted({e.get('name','') for e in kernels})
    return dict(gpu_events=len(gpu), gpu_kernels=len(kernels), kernel_names=names,
        cuda_runtime_events=sum(e.get('cat') == 'cuda_runtime' for e in events),
        cuda_driver_events=sum(e.get('cat') == 'cuda_driver' for e in events),
        cuda_graph_launches=sum('GraphLaunch' in e.get('name','') for e in runtime),
        observed_dispatch_kernels=sum('dispatch' in e.get('name','').lower() for e in kernels),
        observed_combine_kernels=sum('combine' in e.get('name','').lower() for e in kernels),
        observed_spin_kernels=sum('spin_kernel' in e.get('name','') for e in kernels),scoped_proofs=proofs)


def profile_valid(mode, counts):
    if counts['observed_dispatch_kernels'] < 4 or counts['observed_combine_kernels'] < 4 or counts['observed_spin_kernels'] < 3:
        return False
    if mode == MODES[2]:
        return counts['cuda_runtime_events'] == counts['cuda_driver_events'] == 0
    if counts['cuda_graph_launches'] < 3:
        return False
    if mode == MODES[0]:
        return all(p['scope_count'] == 1 and len(p['kernels']) == 1 for p in counts['scoped_proofs'])
    return True
