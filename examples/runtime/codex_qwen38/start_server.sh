#!/usr/bin/env bash
set -euo pipefail
umask 077
source /data/gpfs/projects/punim2715/model_serving/qwen38-27b-uncensored/environment.sh
cd "$QWEN_DEPLOY_ROOT"
exec "$QWEN_PYTHON" -u "$QWEN_DEPLOY_ROOT/serve.py"
