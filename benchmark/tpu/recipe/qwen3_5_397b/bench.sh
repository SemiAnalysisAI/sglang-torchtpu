#!/usr/bin/env bash
# Sample usage: ISL=8192 OSL=1024 CONCUR=4 ./bench.sh
#
# Report MEDIAN ITL and MEDIAN TTFT from this run. Benchmark duration and output
# tok/s swing +-7% across repeats on one live server, while median ITL holds to
# 0.25%, so the aggregate numbers are not usable for A/B comparison on this model.
set -uo pipefail

export ISL=${ISL:-8192}
export OSL=${OSL:-1024}
CONCUR=${CONCUR:-4}
# 10 prompts per concurrency slot, matching the InferenceX benchmark client.
export NUM_PROMPTS=${NUM_PROMPTS:-$((CONCUR * 10))}

export MODEL=${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}
export DATASET_NAME=random-ids
export TOKENIZE_PROMPT=1
export BATCH_SIZE=$CONCUR

cd "$(dirname "$0")/../.."
exec ./bench_serving.sh "$@"
