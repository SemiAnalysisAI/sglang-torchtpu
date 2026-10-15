#!/bin/bash
# Usage: run-cfg-port.sh <config under recipe/configs> <log name> [tree, default glm-port] [venv, default venv-port]
# Detached launch of a ported tree under its own venv.
set -euo pipefail
TREE=${3:-glm-port}; VENV=/dev/shm/home/qinghanc/${4:-venv-port}
cd /dev/shm/home/qinghanc/work
. ./tpu_env.sh
# Compile cache on shared storage only (tier 3, TORCH_TPU_TIER3_ROOT in tpu_env.sh). The
# in-memory tier 2 lives in /dev/shm, which counts toward the pod's memory; turn it off.
export ENABLE_COMPILE_CACHE=1
cd $TREE
setsid nohup $VENV/bin/python -m benchmark.tpu.launch_server --config "recipe/configs/$1" \
  --runtime-envs '{"TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": null}' \
  > /dev/shm/home/qinghanc/work/glm-run/$2.log 2>&1 < /dev/null &
echo "started $1 -> glm-run/$2.log ($TREE, $VENV)"
