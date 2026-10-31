"""CPU-only helpers for the bounded, no-model NCCL lifecycle diagnosis."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD = Path('/home/bxb1/.codex/skills/llm-experiment-guard/scripts/dashboard_guard.py').resolve()
PYTHON = '/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python'
CUDA = '/apps/easybuild-2022/easybuild/software/Core/CUDA/12.8.0'
MODES = ('cpu_cuda_after_capture', 'cuda_only_after_capture')
FLAGS = ('TEARDOWN_CUPTI', 'DISABLE_CUPTI_LAZY_REINIT')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def environment(cfg, mode, run):
    if mode not in MODES:
        raise ValueError('Unconfigured lifecycle mode')
    result = {key: value for key, value in os.environ.items()
              if not key.startswith(('SGLANG_', 'VLLM_', 'NCCL_', 'NVSHMEM_', 'DEEPEP_'))
              and key not in (*FLAGS, 'PYTHONPATH', 'LD_PRELOAD', 'LD_LIBRARY_PATH')}
    result.update(CUDA_VISIBLE_DEVICES='0,1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_HOME=CUDA, CUDA_PATH=CUDA,
        PATH=f'{Path(cfg["python"]).parent}:{CUDA}/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
        LD_LIBRARY_PATH=f'/home/bxb1/conda/lib:{CUDA}/lib64:/apps/easybuild-2022/easybuild/software/Core/OpenSSL/1.1/lib',
        NCCL_CUMEM_HOST_ENABLE='0', NCCL_DEBUG='WARN')
    result.update(TEARDOWN_CUPTI='0', DISABLE_CUPTI_LAZY_REINIT='1', CUDA_MODULE_LOADING='EAGER')
    return result


def trace_counts(payload):
    """Report raw collector evidence, without accepting guessed kernel names."""
    events = payload.get('traceEvents', [])
    gpu = [event for event in events if event.get('cat') in ('kernel', 'cuda_kernel', 'gpu_memcpy', 'gpu_memset')]
    kernels = [event for event in gpu if event.get('cat') in ('kernel', 'cuda_kernel')]
    runtime = [event for event in events if event.get('cat') == 'cuda_runtime']
    proof = []
    for index in range(3):
        label = f'stage-marker-proof:{index}'
        scope = [event for event in events if event.get('name') == label and event.get('cat') == 'user_annotation']
        evidence = []
        for window in scope:
            launches = [event for event in runtime if 'LaunchKernel' in event.get('name', '')
                and (event.get('pid'), event.get('tid')) == (window.get('pid'), window.get('tid'))
                and window['ts'] <= event['ts'] < window['ts'] + window.get('dur', 0)]
            correlations = {event.get('args', {}).get('correlation') for event in launches}
            observed = [event for event in kernels if event.get('args', {}).get('correlation') in correlations]
            evidence.extend(dict(name=event['name'], duration_us=event.get('dur'), device=event.get('args', {}).get('device'))
                            for event in observed)
        proof.append(dict(index=index, scope_count=len(scope), kernels=evidence))
    names = sorted(set(event.get('name', '') for event in kernels))
    return dict(gpu_events=len(gpu), gpu_kernels=len(kernels), kernel_names=names,
        cuda_runtime_events=len(runtime), cuda_launches=sum('LaunchKernel' in event.get('name', '') for event in runtime),
        cuda_graph_launches=sum('GraphLaunch' in event.get('name', '') for event in runtime),
        cuda_synchronizations=sum('Synchronize' in event.get('name', '') for event in runtime),
        scoped_proofs=proof,
        observed_spin_kernels=sum('spin_kernel' in event.get('name', '') for event in kernels),
        observed_nccl_kernels=sum('nccl' in event.get('name', '').lower() for event in kernels))
