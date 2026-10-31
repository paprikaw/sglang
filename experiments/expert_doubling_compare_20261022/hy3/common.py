from pathlib import Path
import hashlib,json,os,sys
ROOT=Path(__file__).resolve().parent
WORKSPACE=Path('/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling')
SG=WORKSPACE/'.engine-worktrees/hy3-a100-qualification-20261018'
SG_COMMIT='77b6177a9dc963d791116e133b70b8755f8711c9'
SG_TAG='serving/sglang/hy3-bf16/dp-triton-metadata-cpu-tested-20261019'
MODEL=Path('/data/gpfs/projects/punim2565/models/Hy3-BF16')
PYTHON='/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python'
CUDA='/apps/easybuild-2022/easybuild/software/Core/CUDA/12.8.0'
GUARD_SCRIPTS=ROOT.parent/'guard/skills/llm-experiment-guard/scripts'
GUARD=GUARD_SCRIPTS/'dashboard_guard.py'
sys.path.insert(0,str(GUARD_SCRIPTS))
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
 temp.write_text(json.dumps(value,indent=2,default=str)+'\n');temp.replace(path)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def environment(arm,attempt,node_rank,run=None):
 env={k:v for k,v in os.environ.items() if not k.startswith(('SGLANG_','VLLM_','NVSHMEM_','DEEPEP_','NCCL_','HUMMING_')) and k not in ('LD_PRELOAD','PYTHONPATH')}
 scratch=Path('/tmp')/f'{os.getuid()}-hy3-expert-surface'/run.name
 env.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(SG/'python'),CUDA_HOME=CUDA,CUDA_PATH=CUDA,
 LD_LIBRARY_PATH=f'/home/bxb1/conda/lib:{CUDA}/lib64:/apps/easybuild-2022/easybuild/software/Core/OpenSSL/1.1/lib',
 PATH=f'{Path(PYTHON).parent}:{CUDA}/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
 PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',MAX_JOBS='4',
 SGLANG_ENABLE_JIT_DEEPGEMM='0',SGLANG_HUMMING_USE_F16_ACCUM='0',HUMMING_COMPILER='nvcc',
 HUMMING_DISABLE_PARALLEL_BUILD='1',HUMMING_CACHE_DIR=str(scratch/'humming-cache'),
 HUMMING_TMP_DIR=str(scratch/'humming-tmp'),TRITON_CACHE_DIR=str(scratch/'triton-cache'),RUN_DIR=str(run),ARM=arm)
 return env
def server_args(arm,node_rank,run=None): return ['--run',str(run)]
