#!/usr/bin/env bash
# One 2-host TPU v7 2x2x2 instance: TP16 / EP16 / DP2, attention TP8.
set -euo pipefail

export NNODES=2
export TP_SIZE=16
export EP_SIZE=16
export DP_SIZE=2
export MAX_RUNNING=4
export TORCH_COMPILE_MAX_BS=4
# Cover the single-request tail as well as concurrent decode.
export TPU_COMPILE_BS="1 2 4"
# The server divides the global prefill budget by DP_SIZE: 4096 per group.
export CHUNKED_PREFILL_SIZE=8192
export MAX_PREFILL_TOKENS=8192
export TPU_COMPILE_TOKENS="4096"

# Reuse the non-DPA recipe's checkpoint, quantization and cache settings.
exec bash "$(dirname "$0")/deepseek_r1_0528_mh.sh" "$@"
