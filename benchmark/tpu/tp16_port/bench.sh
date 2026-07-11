#!/usr/bin/env bash
# Serving benchmark: 1k in / 1k out, concurrency 4, detailed metrics (TTFT/ITL/
# TPOT/E2E percentiles). Run on the RANK-0 host after the server is "fired up".
#
#   ./bench.sh                       # local rank-0 (127.0.0.1:30000)
#   TARGET_HOST=<host0-ip> ./bench.sh   # from another machine
#
# Override the shape: ISL=2048 OSL=512 CONC=8 NUM_PROMPTS=32 ./bench.sh
set -uo pipefail
cd "$(dirname "$0")"
source ./common_env.sh

TARGET_HOST="${TARGET_HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
CONC="${CONC:-4}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
OUT="${OUT:-bench_isl${ISL}_osl${OSL}_c${CONC}.json}"

exec "$PYTHON" -m sglang.bench_serving \
    --backend sglang \
    --host "$TARGET_HOST" --port "$PORT" \
    --model "$MODEL" --tokenizer "$MODEL" \
    --dataset-name random \
    --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio 1.0 \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONC" \
    --warmup-requests "$CONC" \
    --output-details \
    --output-file "$OUT"
