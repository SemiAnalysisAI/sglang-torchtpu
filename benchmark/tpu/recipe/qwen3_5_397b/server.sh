#!/usr/bin/env bash
# Sample usage: ISL=8192 OSL=1024 CONCUR=4 ./server.sh
#
# Qwen3.5-397B-A17B-FP8 on tpu7x, TP8/EP8. A hybrid model: 45 of 60 layers are
# GDN linear attention with a fixed-size recurrent state, 15 are full attention.
# That shapes almost every choice below.
set -uo pipefail

ISL=${ISL:-8192}
OSL=${OSL:-1024}
CONCUR=${CONCUR:-4}
export CONTEXT_LEN=${CONTEXT_LEN:-$((ISL + OSL + 20))}

# GDN slot headroom, and why MAX_TOTAL_TOKENS is not CONCUR*CONTEXT_LEN.
#
# `max_mamba_cache_size` defaults to `max_running_requests`
# (model_runner_kv_cache_mixin.py), so setting max_running equal to the client
# concurrency leaves exactly one GDN slot per in-flight request and zero spare.
# A replacement request then cannot take *any* free slot, only the one released
# by a specific finishing request, and waits a full generation. Measured at
# CONCUR=4: median queue 0.4 ms but p99 TTFT 10797 ms; with 8 slots, p99 TTFT
# 1108 ms and +8% throughput. Median TTFT is unchanged either way -- it is pure
# queueing, not compute.
#
# A few spare slots is the whole fix -- what matters is that a replacement finds
# *some* free slot rather than waiting on one specific request. `+4` reproduces
# the measured CONCUR=4 point (8 slots) and stays affordable as concurrency
# grows, where a multiplier would not: each slot also costs ~2048 tokens of the
# shared KV budget below.
export MAX_RUNNING=${MAX_RUNNING:-$((CONCUR + 4))}

# A mamba slot's bytes bill against the SAME token budget as KV
# (UnifiedMambaTokenToKVPoolAllocator), ~1690 token-equivalents per slot on this
# model. Sizing the pool at exactly CONCUR*CONTEXT_LEN therefore over-commits and
# only CONCUR-1 requests fit. Budget KV for the requests actually in flight, plus
# the slot charge for every slot that exists.
#
# This is the configuration measured at CONCUR=4 (53328 here vs 54352 measured):
# p99 TTFT 1108 ms, median ITL 8.10, 396 tok/s. Sizing instead for MAX_RUNNING
# full contexts would be ~2x larger and is untested.
export MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-$((CONCUR * CONTEXT_LEN + MAX_RUNNING * 2048))}

CHUNK=${CHUNK:-4096}

export MODEL=${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}
export TP_SIZE=8
export EP_SIZE=8
export QUANTIZATION=fp8
export KV_CACHE_DTYPE=fp8_e4m3
export ATTENTION_BACKEND=pallas_tpu_rpa_v3
export PAGE_SIZE=256
export MEM_FRACTION_STATIC=0.85
export CHUNKED_PREFILL_SIZE=$CHUNK
export MAX_PREFILL_TOKENS=$CHUNK
export ENABLE_TORCH_COMPILE=1
export TORCH_COMPILE_MAX_BS=$MAX_RUNNING

# XLA / libtpu. Two deliberate departures from common_env.sh's default set:
#
# 1. scoped_vmem 32768, not 65536. The larger budget starves memory-space
#    assignment on this model: 16.84 vs 9.64 ms median ITL.
#
# 2. SparseCore *gather* offload left ON. common_env.sh disables it as part of
#    the DP-attention hardening set, which arrived with the MiMo recipe rather
#    than from any Qwen3.5 measurement. With it off, the s32[40960]
#    inverse-permutation gather runs on TensorCore as a 401 us fusion x ~60
#    layers = 24.2 ms/step; with it on, XLA offloads it to SparseCore and it
#    nearly vanishes. Measured: 4096-token prefill chunk 177.4 -> 153.0 ms at
#    16k context, 190.4 -> 166.4 ms at 64k.
#
#    The three sparse-core *collective* offload disables are KEPT. They are the
#    part of the hardening set that guards collective correctness/halts, this
#    recipe has not validated enabling them, and doing so showed a possible
#    small ITL regression. Only the gather flag is a measured win.
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-"--xla_tpu_scoped_vmem_limit_kib=32768 \
--xla_tpu_enable_async_collective_fusion=true \
--xla_tpu_overlap_compute_collective_tc=true \
--xla_tpu_enable_async_collective_fusion_multiple_steps=true \
--xla_tpu_enable_latency_hiding_scheduler=true \
--xla_latency_hiding_scheduler_rerun=10 \
--xla_tpu_all_gather_collective_matmul_mode=post_spmd_conservative \
--xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd_conservative \
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=false \
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"}

# Threshold BELOW which the MoE permute uses a dense one-hot matmul instead of a
# gather. Decode (m = batch) stays under it and takes the cheap dense path; a
# 4096-token prefill chunk is far above it and takes the SparseCore gather.
export SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD=${SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD:-1024}

# Build the RoPE cache at the served context, not the checkpoint's 262144
# positions. Safe here because this recipe serves text only; MRoPE video
# positions are structural and can exceed the context, so do not set this for
# multimodal serving.
export SGLANG_TPU_ENABLE_ROPE_CACHE_SLICE=${SGLANG_TPU_ENABLE_ROPE_CACHE_SLICE:-1}

# --language-only means the ViT never executes, but its 29 patch buckets are
# precompiled anyway: 88 s warm / 296 s cold of pure waste.
export SGLANG_TPU_VIT_PRECOMPILE_SIZES=${SGLANG_TPU_VIT_PRECOMPILE_SIZES-}

extra_args=(
  --language-only
  --tpu-fp8-runtime channelwise
  # NOT set, deliberately:
  #  --enable-fused-qk-norm-rope   the Pallas op is validated for head_dim 128
  #                                and this model is 256, so the flag is a
  #                                silent no-op (can_use_tpu_fused_qk_norm_rope)
  #  --moe-runner-backend pallas_tpu_fused_rs
  #                                capped at routed-M <= 2048 while a prefill
  #                                chunk is 4096*topk = 40960, and large M
  #                                core-halts with RPA-v3 (sglang-torchtpu#308)
)
export EXTRA_ARGS="${extra_args[*]}${EXTRA_ARGS:+ $EXTRA_ARGS}"

cd "$(dirname "$0")/../.."
exec ./launch_server.sh "$@"
