"""Matched A100 GLM serving arms using the accepted CE candidate and optional same-native RDMA controls."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
SG = WORKSPACE / '.engine-worktrees/glm47-ep16-canonical-profile-20261004'
SG_COMMIT = '870269702285d94bdd926a50cc29ad1a20bcf60c'
SG_TAG = 'experiment/sglang/glm47-ep16-canonical-profile-20261004-cpu-validated'
DAEMON_SG = WORKSPACE / '.engine-worktrees/glm47-canonical-weight-cache-20261003'
DAEMON_COMMIT = '92e1823d518603bd9b1fb802503739f7370b68fa'
DAEMON_TAG = 'serving/sglang/glm47-awq/canonical-weight-cache-20261003'
DEEP = WORKSPACE / '.third_party/deepep-ce-glm-ep8-20261010'
DEEP_COMMIT = '81a1b5960f5fc73bce33045a56e5b6970e2bcb3c'
DEEP_TAG = 'experiment/deepep/ce-glm-ep8-candidate-20261010'
PYTHON = '/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python'
MODEL = Path('/data/gpfs/projects/punim2715/huggingface/GLM-4.7-AWQ')
CUDA = '/apps/easybuild-2022/easybuild/software/Core/CUDA/12.8.0'
GUARD_SCRIPTS = Path('/home/bxb1/.codex/skills/llm-experiment-guard/scripts').resolve()
GUARD = GUARD_SCRIPTS/'dashboard_guard.py'
ARMS = ('pp2ep4', 'ep8')
NATIVE = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/runtime/deepep_ce_ep8_admission_20261010_r1/candidate/python/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so')
NATIVE_SHA = 'c3405cad906f56aeaebfa4f947bdb09e1d5505248f146879ee9fdacc4bbb2da3'
HUMMING_ROOT = Path(PYTHON).parents[1]/'lib/python3.12/site-packages/humming'
SEED = WORKSPACE/'experiments/glm47_ep8_p2p_vs_tp8_full_20261002/humming_cache'
SHIM = WORKSPACE/'experiments/deepep_ah_diagnosis_h100_20260928/trace_verbs.so'
sys.path.insert(0, str(GUARD_SCRIPTS))
from artifact_visibility import wait_for_stable_json, ArtifactVisibilityTimeout

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def runtime_dir(run=None): return Path(run or os.environ['RUN_DIR']).resolve()
def cfg(run=None): return json.loads((runtime_dir(run)/'config.json').read_text())
def case(arm,run=None): return cfg(run)['cases'][arm]
def save(path,value):
    path=Path(path)
    if not path.is_absolute(): path=runtime_dir()/path
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n');tmp.replace(path)
def visible_json(path,receipt_dir=None,timeout=60):
    path=Path(path);receipts=Path(receipt_dir) if receipt_dir else runtime_dir()/'artifact_visibility'
    dest=receipts/(hashlib.sha256(str(path.absolute()).encode()).hexdigest()[:16]+'.json')
    try: value,receipt=wait_for_stable_json(path,timeout_s=timeout)
    except ArtifactVisibilityTimeout as error: save(dest,error.receipt);raise
    save(dest,receipt);return value
def native_info(arm=None,run=None):
    expected=cfg(run)['deepep_native']
    assert str(NATIVE)==expected['path'] and sha(NATIVE)==expected['sha256']==NATIVE_SHA
    return expected
def environment(arm,attempt,node_rank,run=None):
    run=runtime_dir(run);selected=case(arm,run)
    e={k:v for k,v in os.environ.items() if not k.startswith(('SGLANG_','VLLM_','NVSHMEM_','DEEPEP_','NCCL_','HUMMING_')) and k not in ('LD_PRELOAD','PYTHONPATH')}
    scratch=Path('/tmp')/f'{os.getuid()}-{run.name}'
    e.update(CUDA_VISIBLE_DEVICES='0,1,2,3',PYTHONPATH=f'{SG / "python"}:{DEEP}:{NATIVE.parent}',
        LD_LIBRARY_PATH=f'/home/bxb1/conda/lib:{CUDA}/lib64:/apps/easybuild-2022/easybuild/software/Core/OpenSSL/1.1/lib',
        CUDA_HOME=CUDA,CUDA_PATH=CUDA,CUDA_MODULE_LOADING='EAGER',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PATH=f'{Path(PYTHON).parent}:{CUDA}/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',MAX_JOBS='4',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HUMMING_COMPILER='nvcc',HUMMING_DISABLE_PARALLEL_BUILD='1',
        HUMMING_CACHE_DIR=str(scratch/'unused'),HUMMING_TMP_DIR=str(scratch/'unused_tmp'),
        TORCH_EXTENSIONS_DIR=str(scratch/'torch_extensions'),TRITON_CACHE_DIR=str(scratch/'triton'),
        SGLANG_ENABLE_JIT_DEEPGEMM='0',SGLANG_HUMMING_USE_F16_ACCUM='0',
        SGLANG_HUMMING_MOE_GEMM_TYPE='indexed',TORCH_CUDA_ARCH_LIST='8.0',
        NCCL_CUMEM_HOST_ENABLE='0',NCCL_IB_HCA='mlx5_bond_0',NCCL_SOCKET_IFNAME='bond0.3027',GLOO_SOCKET_IFNAME='bond0.3027',
        ARM=arm,ATTEMPT=attempt,NODE_RANK=str(node_rank),RUN_DIR=str(run),
        LD_PRELOAD=str(SHIM),DEEPEP_AH_ZERO_STATIC_RATE='1',NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME='bond0.3027',
        NVSHMEM_HCA_LIST='mlx5_bond_0:1',NVSHMEM_IB_ADDR_FAMILY='AF_INET',NVSHMEM_IB_ADDR_RANGE='172.26.92.0/22',
        NVSHMEM_IBGDA_NIC_HANDLER='cpu_host_memory',NVSHMEM_QP_DEPTH='4096',NVSHMEM_DISABLE_P2P='0',
        DEEPEP_LL_NVLINK_PAIR_ONLY='1',DEEPEP_LL_PCIE_P2P_MAX_TOKENS=str(selected['ep_capacity'] if selected['transport']=='ce' else 0),
        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=str(selected['ep_capacity']),
        SGLANG_DEEPEP_DISABLE_NVLINK_LOW_LATENCY='0')
    return e
def server_args(arm,node_rank,run=None):
    s=case(arm,run)
    args=['--model-path',str(MODEL),'--trust-remote-code','--dtype','bfloat16',
        '--quantization','humming','--moe-runner-backend','humming',
        '--tp-size',str(s['tp']),'--pp-size',str(s['pp']),'--nnodes','2','--node-rank',str(node_rank),
        '--dist-init-addr',f'{s["nodes"][0]}:{s["dist_port"]}','--host','0.0.0.0','--port',str(s['api_port']),
        '--attention-backend','triton','--context-length','16384',
        '--mem-fraction-static','0.80','--max-total-tokens','65536','--max-running-requests','16',
        '--chunked-prefill-size',str(s['global_prefill_chunk']),'--max-prefill-tokens','1024',
        '--disable-radix-cache','--disable-custom-all-reduce','--disable-shared-experts-fusion',
        '--disable-overlap-schedule','--random-seed','42',
        '--cuda-graph-backend-decode','disabled','--cuda-graph-backend-prefill','disabled',
        '--reasoning-parser','glm45','--tool-call-parser','glm47','--watchdog-timeout','600',
        '--dp-size',str(s['dp']),'--enable-dp-attention','--ep-size',str(s['ep']),
        '--moe-a2a-backend','deepep','--deepep-mode','low_latency','--deepep-dispatcher-output-dtype','bf16']
    if s['pp']>1:args+=['--pp-max-micro-batch-size','8']
    return args
