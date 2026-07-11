#!/usr/bin/env bash
# MiMo-V2.5 *Pro* (1T), 128k context, TP=16 across 2x tpu7x-8 hosts.
#   tp=16, dp=2  ->  attention_tp = tp/dp = 16/2 = 8
#   (=> each of the 2 attention-DP replicas = one full host's 8 devices)
#   torch_compile_max_bs = max_running(2) * dp_size(2) = 4  (DP-padding rule; here 8 for headroom)
#
# !!! PRECONDITION: attention_tp=8 requires the Pro config's num_key_value_heads
#     AND swa_num_key_value_heads to BOTH be multiples of 8. The 310B MiMo-V2.5
#     has GA num_key_value_heads=4 and will FAIL this. Verify first (see README).
#
# RUN THE SAME COMMAND ON BOTH HOSTS, only NODE_RANK differs:
#   Host 0 (rank 0, serves HTTP :30000):   ./launch_tp16_pro_128k.sh 0
#   Host 1 (rank 1):     RANK0_HOST=<host0-internal-ip> ./launch_tp16_pro_128k.sh 1
# GKE tpu7x-16 slice: run with NO arg on every worker (rank auto from TPU_WORKER_*).
set -uo pipefail
cd "$(dirname "$0")"
# Point MODEL at the Pro checkpoint's LOCAL path (never /gcs) before sourcing.
MODEL="${MODEL:-/gcs/models/MiMo-V2.5-Pro}"
source ./common_env.sh
cd "$LAUNCHER_DIR"

NODE_RANK="${1:-${NODE_RANK:-}}"

MODEL="$MODEL" \
NNODES=2 ${NODE_RANK:+NODE_RANK=$NODE_RANK} ${RANK0_HOST:+RANK0_HOST=$RANK0_HOST} \
CHIPS_PER_HOST=8 DIST_PORT="${DIST_PORT:-10011}" \
HOST=0.0.0.0 PORT="${PORT:-30000}" \
TP_SIZE=16 EP_SIZE="${EP_SIZE:-16}" \
QUANTIZATION= KV_CACHE_DTYPE=fp8_e4m3 \
CONTEXT_LEN=131072 MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-140288}" \
MAX_RUNNING="${MAX_RUNNING:-2}" TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-8}" \
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.3}" \
CHUNKED_PREFILL_SIZE=4096 MAX_PREFILL_TOKENS=4096\
TPU_COMPILE_TOKENS="64 128 256 512 1024 2048 4096" \
ENABLE_TORCH_COMPILE=1 \
JSON_MODEL_OVERRIDE='{"vision_config":null,"audio_config":null}' \
EXTRA_ARGS="--dp-size 2 --enable-dp-attention --enable-dp-lm-head --trust-remote-code ${EXTRA_ARGS:-}" \
exec ./launch_server.sh
