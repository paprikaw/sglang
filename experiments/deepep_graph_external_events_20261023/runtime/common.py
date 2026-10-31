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
MODES = ('external_events_after_capture',)
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


EVENT_NAMES = ('graph_start','dispatch_start','dispatch_end','combine_start','combine_end','graph_end')

def validate_event_sample(row):
    import math
    stamps = row.get('timestamps_from_origin_ms',{})
    if set(stamps) != set(EVENT_NAMES) or any(not isinstance(stamps[k],(int,float)) or not math.isfinite(stamps[k]) or stamps[k] <= 0 for k in EVENT_NAMES):
        raise ValueError('Missing, non-finite or nonpositive actual event timestamps')
    values = [stamps[k] for k in EVENT_NAMES]
    if any(b < a for a,b in zip(values,values[1:])):
        raise ValueError('External event nesting/order is invalid')
    for key,start,end in [('dispatch_ms','dispatch_start','dispatch_end'),('combine_ms','combine_start','combine_end'),('graph_ms','graph_start','graph_end')]:
        measured = row.get(key)
        if not isinstance(measured,(int,float)) or not math.isfinite(measured) or measured <= 0:
            raise ValueError('Missing or nonpositive actual GPU span: '+key)
        # Independent cudaEventElapsedTime calls can round at float32 precision.
        if abs(measured-(stamps[end]-stamps[start])) > max(.02,abs(measured)*.02):
            raise ValueError('Event span does not agree with origin-relative timestamps: '+key)
    if row['dispatch_ms'] + row['combine_ms'] > row['graph_ms'] + .02:
        raise ValueError('Nested communication spans exceed enclosing Graph')
    return True

def validate_freshness(samples):
    if len(samples) != 3 or [s.get('payload_phase') for s in samples] != [1,2,3]:
        raise ValueError('Exactly three changed-payload replays required')
    for sample in samples: validate_event_sample(sample)
    starts = [s['timestamps_from_origin_ms']['graph_start'] for s in samples]
    if any(b-a < 5 for a,b in zip(starts,starts[1:])):
        raise ValueError('Graph event timestamps did not refresh after deliberate 20ms host-only gap')
    if not all(s.get('payload_valid') is True for s in samples):
        raise ValueError('Each changed-payload replay needs exact GPU payload proof')
    return dict(passed=True,graph_start_timestamps_from_origin_ms=starts,
        elapsed_between_replay_starts_ms=[b-a for a,b in zip(starts,starts[1:])],
        host_sleep_between_samples_s=.02,not_capture_constant=True)
