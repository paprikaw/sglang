"""Pinned no-model A100 locality test, same native and API mode as the full Hy3 run."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WORKSPACE=Path('/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling')
MANAGER=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager')
DEEP=WORKSPACE/'.third_party/deepep-ce-glm-ep8-20261010'
DEEP_COMMIT='81a1b5960f5fc73bce33045a56e5b6970e2bcb3c'
HOST_COMMIT='7d1ebddc49649fe06d6b8ccd2f77876e2ba9b4ce'
DEEP_TAG='experiment/deepep/ce-glm-ep8-validated-20261010'
NATIVE_SO=MANAGER/'runtime/deepep_ce_ep8_admission_20261010_r1/candidate/python/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so'
NATIVE_SHA='c3405cad906f56aeaebfa4f947bdb09e1d5505248f146879ee9fdacc4bbb2da3'
CACHED_NATIVE_RECEIPT=NATIVE_SO.parents[1]/'build_result.json'
PYTHON='/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python'
CUDA='/apps/easybuild-2022/easybuild/software/Core/CUDA/12.8.0'
GUARD_SCRIPTS=Path('/home/bxb1/experiment-guard/skills/llm-experiment-guard/scripts')
GUARD=GUARD_SCRIPTS/'dashboard_guard.py'
DEFAULT_RUN=WORKSPACE/'experiments/single_layer_ep4_ep8_matched_20261024/comm_r1'
NODES=('spartan-gpgpu065','spartan-gpgpu167')
JOBS=('32715970','32715983')
TRACE_SHIM=WORKSPACE/'experiments/deepep_ah_diagnosis_h100_20260928/trace_verbs.so'
TRACE_SHIM_SHA='d0434685656281f372a6cda96f26e14778e325e77d8af96be1fc3698b58f6a68'
CAPACITY,HIDDEN,EXPERTS,TOPK,WARMUP,ITERATIONS=512,4096,192,8,3,3
TOKEN_SWEEP=(1,128)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)
def environment(physical_gpus):
    assert tuple(physical_gpus) in ((0,1),(0,1,2,3))
    env={k:v for k,v in os.environ.items() if not k.startswith(('SGLANG_','VLLM_','NVSHMEM_','DEEPEP_','NCCL_','HUMMING_','EXPERIMENT_GUARD_')) and k not in ('LD_PRELOAD','PYTHONPATH','CUDA_VISIBLE_DEVICES')}
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,physical_gpus)),PYTHONPATH=f'{DEEP}:{NATIVE_SO.parent}',
        LD_LIBRARY_PATH=f'/home/bxb1/conda/lib:{CUDA}/lib64:/apps/easybuild-2022/easybuild/software/Core/OpenSSL/1.1/lib',
        LD_PRELOAD=str(TRACE_SHIM),PATH=f'{Path(PYTHON).parent}:{CUDA}/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',CUDA_HOME=CUDA,CUDA_PATH=CUDA,
        GLOO_SOCKET_IFNAME='bond0.3027',NCCL_SOCKET_IFNAME='bond0.3027',NCCL_CUMEM_HOST_ENABLE='0',
        NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME='bond0.3027',NVSHMEM_HCA_LIST='mlx5_bond_0:1',NVSHMEM_IB_ADDR_FAMILY='AF_INET',
        NVSHMEM_IB_ADDR_RANGE='172.26.92.0/22',NVSHMEM_IBGDA_NIC_HANDLER='cpu_host_memory',NVSHMEM_DISABLE_P2P='0',
        NVSHMEM_DEBUG='INFO',
        NVSHMEM_QP_DEPTH='4096',DEEPEP_AH_ZERO_STATIC_RATE='1',DEEPEP_LL_NVLINK_PAIR_ONLY='1',DEEPEP_LL_PCIE_P2P_MAX_TOKENS='0',SGLANG_DEEPEP_DISABLE_NVLINK_LOW_LATENCY='0')
    return env
