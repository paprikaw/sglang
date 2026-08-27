#!/usr/bin/env bash
# Read-only dependencies from the working SGLang environment; no shared installs.
export QWEN_DEPLOY_ROOT=/data/gpfs/projects/punim2715/model_serving/qwen38-27b-uncensored
export QWEN_SGLANG_SOURCE=/data/gpfs/projects/punim2715/vllm_workbench/sglang_autoscaling/sglang-upstream-95f5ecd
export QWEN_PYTHON=/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python
export PYTHONPATH="$QWEN_SGLANG_SOURCE/python"
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.8.0
export CUDA_PATH="$CUDA_HOME"
export CUDA_ROOT="$CUDA_HOME"
export CUDAToolkit_ROOT="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
# The login module otherwise injects CUDA 12.4 headers into 12.8 JIT builds.
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH LIBRARY_PATH NVCC_PREPEND_FLAGS
export LD_LIBRARY_PATH="/home/bxb1/conda/lib:$CUDA_HOME/lib64"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MAX_JOBS=8
export KVCACHED_AUTOPATCH=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN
export NCCL_CUMEM_HOST_ENABLE=0
