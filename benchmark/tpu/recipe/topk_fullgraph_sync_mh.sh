#!/usr/bin/env bash
# Two-host TP16 validation for compile-static distributed TopK in the intact graph.
set -euo pipefail

cd "$(dirname "$0")/.."  # recipe/ -> benchmark/tpu

# Match the reported shell command: CI otherwise injects a 300-second soft
# watchdog that can tear down this large model while it is still precompiling.
exec env \
  MODEL="${MODEL:-/weights/Qwen3-Coder-480B-A35B-Instruct-FP8}" \
  TP_SIZE=16 \
  EP_SIZE=16 \
  CONC=64 \
  MAX_RUNNING=128 \
  CONTEXT_LEN=8448 \
  MAX_TOTAL_TOKENS=1081344 \
  PAGE_SIZE=256 \
  CHUNKED_PREFILL_SIZE=8192 \
  MAX_PREFILL_TOKENS=8192 \
  TPU_COMPILE_TOKENS="1024 4096 8192" \
  TPU_COMPILE_BS="1 16 32 64 128" \
  TORCH_COMPILE_MAX_BS=128 \
  MEM_FRACTION_STATIC=0.8 \
  ATTENTION_BACKEND=pallas_tpu_rpa_batched \
  QUANTIZATION=fp8 \
  KV_CACHE_DTYPE=fp8_e4m3 \
  SGLANG_IS_IN_CI=false \
  SGLANG_TPU_CHECK_RECOMPILATION=1 \
  EXTRA_ARGS="--trust-remote-code --tpu-compile-tracer dynamo --tpu-fp8-runtime channelwise --sampling-defaults openai --enable-fused-qk-norm-rope" \
  bash launch_server.sh
