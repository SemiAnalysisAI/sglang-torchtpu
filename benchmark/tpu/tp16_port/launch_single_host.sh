#!/usr/bin/env bash
# SINGLE-HOST (1x tpu7x-8, tp=8) MiMo-V2.5 DP-attention — the exact config
# validated on this host (tp8 / ep8 / dp2, attention_tp = 8/2 = 4 = num_kv_heads).
# This is the baseline the TP=16 2-host run scales up from.
#
#   ./launch_single_host.sh
#
# Serving shape sized for the 1k/1k concurrency<=4 bench. dummy weights
# (--load-format dummy); drop that from EXTRA_ARGS to load a real checkpoint.
set -uo pipefail
cd "$(dirname "$0")"
source ./common_env.sh
cd "$LAUNCHER_DIR"

MODEL="$MODEL" \
TP_SIZE=8 EP_SIZE=8 \
QUANTIZATION= KV_CACHE_DTYPE=fp8_e4m3 \
CONTEXT_LEN=4096 MAX_TOTAL_TOKENS=16384 \
MAX_RUNNING=4 TORCH_COMPILE_MAX_BS=8 \
MEM_FRACTION_STATIC=0.5 \
CHUNKED_PREFILL_SIZE=2048 MAX_PREFILL_TOKENS=2048 \
TPU_COMPILE_TOKENS="64 128 256 512 1024 2048" \
ENABLE_TORCH_COMPILE=1 \
JSON_MODEL_OVERRIDE='{"vision_config":null,"audio_config":null}' \
EXTRA_ARGS="--dp-size 2 --enable-dp-attention --enable-dp-lm-head --trust-remote-code --load-format dummy" \
exec ./launch_server.sh
