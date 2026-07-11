#!/usr/bin/env bash
# MULTI-HOST TP=16 across 2x tpu7x-8 hosts (2 hosts x 8 devices = tp16).
# MiMo-V2.5 DP-attention scaled up from the single-host tp8/dp2 baseline:
#   tp=16, ep=16, dp=4   ->  attention_tp = tp/dp = 16/4 = 4 = num_kv_heads (fused-qkv OK)
#   torch_compile_max_bs = max_running(4) * dp_size(4) = 16   (DP-padding bucket rule)
#
# RUN THE SAME COMMAND ON *BOTH* HOSTS, only NODE_RANK differs.
#
#   Host 0 (rank 0 — serves the HTTP endpoint on :30000):
#       ./launch_tp16_multihost.sh 0
#   Host 1 (rank 1):
#       RANK0_HOST=<host0-internal-ip> ./launch_tp16_multihost.sh 1
#
# GKE TPU pod slice (tpu7x-16): env TPU_WORKER_ID / TPU_WORKER_HOSTNAMES are set
# automatically, so you can instead run this with NO arg on every worker — rank,
# peer list and tp=16 all auto-derive.
#
# Prereqs on BOTH hosts (see README):
#   - identical sglang branch (fix 312cb2bf5 present), torch_tpu pin, libtpu 0.0.40
#   - MODEL present at the same local path (stage to /cache, never /gcs)
#   - rendezvous port (DIST_PORT, default 10011) open host0<->host1
set -uo pipefail
cd "$(dirname "$0")"
source ./common_env.sh
cd "$LAUNCHER_DIR"

# NODE_RANK from $1, else env, else GKE auto-detect inside launch_server.sh.
NODE_RANK="${1:-${NODE_RANK:-}}"

MODEL="$MODEL" \
NNODES=2 ${NODE_RANK:+NODE_RANK=$NODE_RANK} ${RANK0_HOST:+RANK0_HOST=$RANK0_HOST} \
CHIPS_PER_HOST=8 DIST_PORT="${DIST_PORT:-10011}" \
HOST=0.0.0.0 PORT="${PORT:-30000}" \
EP_SIZE=16 \
QUANTIZATION= KV_CACHE_DTYPE=fp8_e4m3 \
CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=16384 \
MAX_RUNNING=4 TORCH_COMPILE_MAX_BS=16 \
MEM_FRACTION_STATIC=0.5 \
CHUNKED_PREFILL_SIZE=2048 MAX_PREFILL_TOKENS=2048 \
TPU_COMPILE_TOKENS="64 128 256 512 1024 2048" \
ENABLE_TORCH_COMPILE=1 \
JSON_MODEL_OVERRIDE='{"vision_config":null,"audio_config":null}' \
EXTRA_ARGS="--dp-size 4 --enable-dp-attention --enable-dp-lm-head --trust-remote-code --load-format dummy" \
exec ./launch_server.sh
