#!/bin/bash
# Usage: split-e2e.sh <tree under work/> <test path in tree> <tag> -- run a TPU e2e test file against that tree.
W=/dev/shm/home/qinghanc/work; T=$W/$1
exec > $W/glm-run/e2e-$3.out 2>&1
. $W/tpu_env.sh
bash $W/stop_wait.sh
export PYTHON=/dev/shm/home/qinghanc/venv-port-b/bin/python PYTHONPATH=$T/python:$T HEALTH_TIMEOUT=3600
cd $T/benchmark/tpu && $PYTHON $T/$2
echo "E2E_EXIT $?"
