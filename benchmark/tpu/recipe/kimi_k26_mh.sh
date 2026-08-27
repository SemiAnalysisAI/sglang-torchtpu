#!/usr/bin/env bash
# Kimi-K2.6 (1.1T MoE, native INT4 compressed-tensors) + MoonViT multi-host
# launch recipe -- TEXT PATH VALIDATED, VISION PATH NOT.
#
# STATUS: run end-to-end against the K2.6 checkpoint on a 2-host v7x slice
# (2026-08-26): weights load, the int4 MoE serves, and GPQA-Diamond scores under
# the card protocol. The MoonViT tower is untested -- its precompile ladder is
# disabled below and no vision gate exists yet, so treat only the text path as
# covered.
#
# Sizing rationale:
#   1.1T total, int4 experts -> ~37GB/device at tp=16 over 2 hosts, confirmed on
#   the K2.6 checkpoint. That is 5GB/device less than R1, so the memory knobs
#   below can match the R1 recipe outright; see MEM_FRACTION_STATIC.
#   ep=16 matches tp. EP=8 leaves tp-within-expert=2, whose per-rank intermediate
#   has no 128-aligned divisor for the gmm tile ("pad the dim" ValueError) -- do
#   not use.
#
# QUANTIZATION= must stay explicitly EMPTY: launch_server.sh defaults to
# ${QUANTIZATION-fp8}, and passing fp8 silently mis-loads the int4 experts.
# Empty lets sglang auto-detect compressed-tensors from the checkpoint config.
#
# A prefill shape outside the precompiled buckets takes the recompile path that
# crashes torch_tpu, so TPU_COMPILE_TOKENS must cover every shape the traffic
# produces -- including, once vision is enabled, the 256 tokens a 448x448 image
# becomes after the 4:1 merge.
#
# Weights: /weights/Kimi-K2.6 on the shared model-weights-rox ROX Hyperdisk.
set -uo pipefail

export MODEL=${MODEL:-/weights/Kimi-K2.6}

# --- parallelism ---
export NNODES=${NNODES:-2}
export TP_SIZE=${TP_SIZE:-16}
export EP_SIZE=${EP_SIZE:-16}

# --- precision ---
# Native int4 (compressed-tensors pack-quantized). Both MUST be empty; see above.
export QUANTIZATION=
export KV_CACHE_DTYPE=

# --- attention ---
# Kimi is an MLA model.
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-pallas_tpu_mla}

# --- memory / shapes ---
# Sized for the K2.6 card protocol (context 262144, reasoning generations up to
# 98304 tokens) under CI's cold-compile mode, and matched to the R1 recipe
# wherever the two models have no reason to differ.
#
# Precompile host memory is driven by the number of compiled graphs, not by the
# batch cap. Measured on a 963Gi node (2026-08-26): bs=64 with the 8-rung token
# ladder grew past 612GB mid-EXTEND and got the pod OOM-killed, while a single
# 8192-token graph peaked at 257GB. Chunked prefill caps every prefill at 8192
# and shorter shapes pad up to it, so one token bucket is full coverage.
#
# The batch grid is one bucket for the same reason, and it is not a trim either:
# the bucketer returns the smallest bucket >= n, so a single bucket at the cap
# covers every batch width by padding up to it. Two buckets is what breaks --
# TPU_COMPILE_BS="64 128" left no graph live at 128 and the first decode batch
# that wide tripped fail_on_recompile and killed the server (2026-08-26). Keep
# TPU_COMPILE_BS a single value equal to MAX_RUNNING.
# The batch cap is a ceiling, not a reservation: the scheduler admits against
# free KV and retracts under pressure, so it self-limits. Measured on the card
# protocol (2026-08-26), 32 concurrent GPQA requests held the pool at 0.98 with
# zero retractions. Kimi is MLA with a single latent head, so the KV pool is
# replicated per rank rather than sharded -- ~65KB per token per core, ~660k
# tokens at mfs 0.85. Capping generation at 32768 (see CONTEXT_LEN) bounds the
# per-request footprint the way R1's does, which is what lets the memory knobs
# match deepseek_r1_0528_mh.sh outright.
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
export MAX_RUNNING=${MAX_RUNNING:-128}
export CONTEXT_LEN=${CONTEXT_LEN:-40960}  # 32768 gen + 8192 prompt headroom
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
export MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
export TPU_COMPILE_TOKENS=${TPU_COMPILE_TOKENS:-"8192"}
export TPU_COMPILE_BS=${TPU_COMPILE_BS:-"128"}
export TORCH_COMPILE_MAX_BS=${TORCH_COMPILE_MAX_BS:-$MAX_RUNNING}

# MoonViT's precompile ladder is a second graph family on top of the text ones,
# and text evals never reach it. Set-but-empty disables it; unset the variable
# (not just blank it) for vision runs.
export SGLANG_TPU_VIT_PRECOMPILE_SIZES=${SGLANG_TPU_VIT_PRECOMPILE_SIZES-""}

# Custom config/tokenizer/processor code (configuration_kimi_k25.py,
# kimi_k25_processor.py) ships in the checkpoint.
# Each host page-caches the ~278GiB of shards its ranks read; held alongside
# compile memory that is enough to evict the pod on a 963Gi node.
export EXTRA_ARGS="--trust-remote-code --weight-loader-drop-cache-after-load${EXTRA_ARGS:+ $EXTRA_ARGS}"

# sglang's VL warmup exercises the multimodal embed-merge path. That aborted
# XLA:TPU's fusion emitter until main's int32 `masked_scatter_` ATen kernel
# landed (hardware_backend/tpu/patches.py). SKIP_WARMUP=0 once a green run
# confirms the kernel makes it safe.
: "${SKIP_WARMUP:=1}"
[[ "$SKIP_WARMUP" == "1" ]] && export EXTRA_ARGS="--skip-server-warmup${EXTRA_ARGS:+ $EXTRA_ARGS}"

export ENABLE_PROFILER_ENDPOINT=${ENABLE_PROFILER_ENDPOINT:-0}

cd "$(dirname "$0")/.."
exec ./launch_server.sh "$@"
