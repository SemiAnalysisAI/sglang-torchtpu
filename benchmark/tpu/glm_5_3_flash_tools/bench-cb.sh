#!/bin/bash
# Usage: bench-cb.sh <tag> <concurrency> -- cookbook-parity 8k1k: random, range ratio 1, 8 (c1) / 32 (c16) prompts, 2 warmups.
cd /dev/shm/home/qinghanc/work
. ./tpu_env.sh
export PYTHON=$HOME/venv/bin/python MODEL=/weights/GLM-5.3-Flash WORKLOAD=8k1k RANDOM_RANGE_RATIO=1 \
       BATCH_SIZE=$2 NUM_PROMPTS=$([ $2 = 1 ] && echo 8 || echo 32) WARMUP_REQUESTS=2 OUTPUT_DIR=/dev/shm/home/qinghanc/work/results/$1-c$2 HEALTH_TIMEOUT=3600
bash glm-dsa-tpu/benchmark/tpu/bench_serving.sh > glm-run/bench-$1-c$2.log 2>&1
echo "== $1 c$2"; grep -E "Successful|Output token thr|Total token thr|Accept|Mean TTFT|Mean TPOT|Median TPOT" glm-run/bench-$1-c$2.log
