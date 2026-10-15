#!/bin/bash
# Usage: bench-std.sh <tag> [concurrency] -- the repo's standard 8k1k bench_serving (random, range 0.8), detached.
set -euo pipefail
cd /dev/shm/home/qinghanc/work
. ./tpu_env.sh
export PYTHON=$HOME/venv/bin/python MODEL=/weights/GLM-5.3-Flash WORKLOAD=8k1k \
       BATCH_SIZE=${2:-1} OUTPUT_DIR=/dev/shm/home/qinghanc/work/results/$1 HEALTH_TIMEOUT=3600
setsid nohup bash glm-dsa-tpu/benchmark/tpu/bench_serving.sh > glm-run/bench-$1.log 2>&1 < /dev/null &
echo "bench $1 -> glm-run/bench-$1.log"
