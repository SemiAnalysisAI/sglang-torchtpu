#!/usr/bin/env bash
# Perf benchmark: MiMo-V2.5-Pro, 128k context, against a RUNNING TPU server (:30000).
# Thin wrapper over ../bench_serving.sh recording the exact 128k workload used for
# the concurrency sweep (see results_mimo_pro_128k_conc2.md and RUNBOOK.md).
#
# Prereq: server already launched per RUNBOOK.md and healthy on :30000.
#   - On a FRESH server, run ../warmup.sh once first to compile all shape buckets
#     (otherwise the first measured requests pay the cold-compile cost in TTFT).
#   - Between repeated runs on the same server, warmup is NOT needed.
#
# Usage:
#   ./run_bench_128k.sh [CONCURRENCY]        # default concurrency = 2
# Override any knob via env, e.g.:
#   NUM_PROMPTS=16 ./run_bench_128k.sh 4
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONC="${1:-2}"

export PYTHON="${PYTHON:-/cache/work/sglang-torchtpu/.venv/bin/python}"
export MODEL="${MODEL:-/gcs/models/MiMo-V2.5-Pro}"

cd "$HERE/.."   # benchmark/tpu — where bench_serving.sh and warmup.sh live

# 128k context: ISL + OSL == context_length (131072). range_ratio=1.0 => every
# request is exactly 128k (no length jitter). NUM_PROMPTS defaults to 4 waves.
ISL="${ISL:-130048}" \
OSL="${OSL:-1024}" \
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-1.0}" \
BATCH_SIZE="$CONC" \
NUM_PROMPTS="${NUM_PROMPTS:-$((CONC * 4))}" \
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}" \
OUTPUT_DIR="${OUTPUT_DIR:-$HERE/results/mimo_pro_128k_c${CONC}}" \
exec ./bench_serving.sh
