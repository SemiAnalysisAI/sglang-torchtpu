#!/bin/bash
# Run the repo's GLM-5.3-Flash recipe unchanged on one tpuq host (env only).
set -euo pipefail
cd /dev/shm/home/qinghanc/work
. ./tpu_env.sh
export PYTHON=$HOME/venv/bin/python HOST=127.0.0.1 PORT=30000
export ENABLE_COMPILE_CACHE=1 TORCH_TPU_TIER2_NAME=qinghanc_glm_flash_recipe
exec bash glm-dsa-tpu/benchmark/tpu/recipe/glm_5_3_flash.sh
