#!/bin/bash
# Usage: run-cfg.sh <config under recipe/configs> <log name> -- detached launch via the unified Python launcher.
set -euo pipefail
cd /dev/shm/home/qinghanc/work
. ./tpu_env.sh
export ENABLE_COMPILE_CACHE=1 TORCH_TPU_TIER2_NAME=qinghanc_glm_flash_recipe
cd glm-dsa-tpu
setsid nohup $HOME/venv/bin/python -m benchmark.tpu.launch_server --config "recipe/configs/$1" \
  > /dev/shm/home/qinghanc/work/glm-run/$2.log 2>&1 < /dev/null &
echo "started $1 -> glm-run/$2.log"
