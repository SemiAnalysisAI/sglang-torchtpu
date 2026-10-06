#!/usr/bin/env bash
# GLM-5.3 text launch on a 2-host v7x slice (TP16/EP16). Same bounded-context
# contract as glm_5_3.sh; only the parallelism differs. Launch on BOTH hosts:
# the launcher reads TPU_WORKER_ID / TPU_WORKER_HOSTNAMES for rank + rendezvous.
# Overlap schedule is ON here: it hides the ~10 ms/step host gap and takes
# bs=1 decode from 32.8 to 22.8 ms/token (TP16, measured 2026-10-01).
# TensorCore gather for tiny MoE batches reduced the matched bs=1 latency from
# 22.55 to 19.36 ms/token (2026-10-02), with identical checked token logprobs.
# Set SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD=0 for the prior SC path.
# Original single-host recipe: Within 2048 tokens the indexer selects every
# causal key, so ordinary MLA is exact and needs no sparse-indexer kernel.
set -euo pipefail
cd "$(dirname "$0")/.."
glm_context=${CONTEXT_LEN:-2048}
if (( glm_context < 1 || glm_context > 2048 )); then
  echo 'GLM-5.3 reference recipe requires 1 <= CONTEXT_LEN <= 2048.' >&2
  exit 2
fi
exec env \
  MODEL="${MODEL:-/weights/GLM-5.3}" \
  TP_SIZE="${TP_SIZE:-16}" \
  EP_SIZE="${EP_SIZE:-16}" \
  ATTENTION_BACKEND=pallas_tpu_mla \
  CONTEXT_LEN="$glm_context" \
  CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-128}" \
  MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-128}" \
  MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}" \
  PAGE_SIZE="${PAGE_SIZE:-128}" \
  MAX_RUNNING="${MAX_RUNNING:-2}" \
  TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-2}" \
  TPU_COMPILE_TOKENS="${TPU_COMPILE_TOKENS:-128}" \
  TPU_COMPILE_BS="${TPU_COMPILE_BS:-1 2}" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.95}" \
  KV_CACHE_DTYPE=bfloat16 \
  DISABLE_RADIX_CACHE=1 \
  DISABLE_OVERLAP_SCHEDULE="${DISABLE_OVERLAP_SCHEDULE:-0}" \
  SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD="${SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD:-3}" \
  ENABLE_PROFILER_ENDPOINT=0 \
  EXTRA_ARGS="--json-model-override-args {\"tpu_dense_dsa_max_context\":2048} --watchdog-timeout 1800 --disable-shared-experts-fusion ${EXTRA_ARGS:-}" \
  bash launch_server.sh
