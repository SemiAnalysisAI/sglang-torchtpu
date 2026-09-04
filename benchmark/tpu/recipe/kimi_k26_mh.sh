#!/usr/bin/env bash
# Kimi-K2.6 (1.1T MoE, native INT4 compressed-tensors) + MoonViT multi-host
# launch recipe -- text path validated; vision path smoke-validated with VISION=1.
#
# STATUS: run end-to-end against the K2.6 checkpoint on a 2-host v7x slice
# (2026-08-26): weights load, the int4 MoE serves, and GPQA-Diamond scores under
# the card protocol. Vision (2026-09-03, VISION=1): the MoonViT ladder compiles,
# single- and two-image MMMU-Pro questions answer correctly through
# /v1/chat/completions; a full MMMU-Pro pass is the remaining gate.
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
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE-""}   # bf16 default; fp8_e4m3 halves KV bytes (MLA fp8 KV supported)

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
# The batch cap is set by the KV pool, and 128 is past it. Kimi is MLA with a
# single latent head, so the pool is replicated per rank rather than sharded:
# ~65KB per token per core, ~660k tokens at mfs 0.85. Under the card protocol
# its reasoning chains average ~13.5k tokens, so the pool holds ~48 of them.
#
# Measured on a 2-host v7x slice (2026-08-26). At 128 the pool pins at 1.00 and
# the scheduler retracts to make room: 816 retraction events against 78
# completions, each retracted request losing its KV and re-prefilling. Forward
# progress collapsed -- 30 completions in the first half hour, then 23, 8, 10,
# 5, 2 -- while aggregate decode throughput looked healthy at ~345 tok/s. The
# throughput number is the trap: the tokens were going into re-prefilling work
# already done. At 32 against a 443k pool the same eval held 0.64 q/min with
# zero retractions.
#
# So size the cap to pool/avg_chain, not to what the scheduler will admit. It
# admits 128 happily and then thrashes.
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
export MAX_RUNNING=${MAX_RUNNING:-48}
export CONTEXT_LEN=${CONTEXT_LEN:-40960}  # 32768 gen + 8192 prompt headroom
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
export MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
export TPU_COMPILE_TOKENS=${TPU_COMPILE_TOKENS:-"8192"}
export TPU_COMPILE_BS=${TPU_COMPILE_BS:-"48"}
export TORCH_COMPILE_MAX_BS=${TORCH_COMPILE_MAX_BS:-$MAX_RUNNING}

# MoonViT's precompile ladder is a second graph family on top of the text ones,
# and text evals never reach it, so it stays off unless VISION=1. Buckets are
# pre-merge patches per image (the TPU policy runs each image at its own
# bucket); these 13 squares cover MMMU-Pro, whose largest image is 16,900 =
# 130^2 patches at the checkpoint's in_patch_limit 16384, with ~1.3x padding.
# An image past the top bucket runs eagerly rather than recompiling.
if [[ "${VISION:-0}" == "1" ]]; then
  export SGLANG_TPU_VIT_PRECOMPILE_SIZES=${SGLANG_TPU_VIT_PRECOMPILE_SIZES:-"256,400,576,784,1024,1600,2304,3136,4096,6400,9216,12544,16900"}
else
  export SGLANG_TPU_VIT_PRECOMPILE_SIZES=""
fi

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
