#!/usr/bin/env bash
# GLM-5.3-Flash text-only TP8/EP8 recipe. DSA is exactly dense within the
# configured context (2048 selected tokens); KDA runs on the GDN v3 Pallas kernel.
# Decode speed flags (8k/1k c1 TPOT, measured on v7x TP8): route-meta MoE for
# one token (threshold just above m_pad * topk = 16, so larger batches keep the
# existing route), single-kernel W8A8 dense linears, f32 MoE scales (avoids a
# per-step scale relayout).
set -euo pipefail
cd "$(dirname "$0")/.."
exec env \
  MODEL="${MODEL:-/weights/GLM-5.3-Flash}" \
  TP_SIZE="${TP_SIZE:-8}" \
  EP_SIZE="${EP_SIZE:-8}" \
  CONTEXT_LEN="${CONTEXT_LEN:-2048}" \
  CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-128}" \
  MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-128}" \
  MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}" \
  PAGE_SIZE="${PAGE_SIZE:-128}" \
  MAX_RUNNING="${MAX_RUNNING:-2}" \
  TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-2}" \
  TPU_COMPILE_TOKENS="${TPU_COMPILE_TOKENS:-128}" \
  TPU_COMPILE_BS="${TPU_COMPILE_BS:-1 2}" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}" \
  KV_CACHE_DTYPE=bfloat16 \
  DISABLE_RADIX_CACHE=1 \
  DISABLE_OVERLAP_SCHEDULE=1 \
  ENABLE_PROFILER_ENDPOINT=0 \
  SGLANG_MAMBA_SSM_DTYPE=float32 \
  SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD="${SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD:-17}" \
  SGLANG_TPU_FP8_LINEAR_SMALL_M_KERNEL="${SGLANG_TPU_FP8_LINEAR_SMALL_M_KERNEL:-1}" \
  SGLANG_TPU_FP8_MOE_FP32_SCALES="${SGLANG_TPU_FP8_MOE_FP32_SCALES:-1}" \
  EXTRA_ARGS="--language-model-only --watchdog-timeout 1800 --max-mamba-cache-size 4 --disable-shared-experts-fusion ${EXTRA_ARGS:-}" \
  bash launch_server.sh
