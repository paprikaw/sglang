#!/usr/bin/env bash
set -euo pipefail
umask 077
task_root=/data/gpfs/projects/punim2715/model_serving/qwen38-27b-uncensored
bash "$task_root/start_server.sh" 2>&1 | tee -a "$task_root/logs/server-service.log"
