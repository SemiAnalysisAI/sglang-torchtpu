#!/usr/bin/env bash
# Original two-host TP16 / EP16 FP8-KV evaluation, without DP attention.
set -euo pipefail
export NNODES=2
export TP_SIZE=16
export EP_SIZE=16
export DP_SIZE=1
export KV_CACHE_DTYPE=fp8_e4m3
export MAX_RUNNING=128
export TPU_COMPILE_BS="64 128"
export ENABLE_PROFILER_ENDPOINT=0
exec bash "$(dirname "$0")/deepseek_r1_0528_mh.sh" "$@"
