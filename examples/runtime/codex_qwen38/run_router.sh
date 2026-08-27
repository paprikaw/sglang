#!/usr/bin/env bash
set -euo pipefail
umask 077
task_root=/data/gpfs/projects/punim2715/model_serving/qwen38-27b-uncensored
exec /data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/python3 -u "$task_root/router.py" >> "$task_root/logs/router.log" 2>&1
