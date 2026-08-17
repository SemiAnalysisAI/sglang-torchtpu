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
# Remembered before the default is applied, so the DP block below can tell an
# explicit caller value from this derived one and leave the former alone.
_MAX_TOTAL_FROM_CALLER=${MAX_TOTAL_TOKENS:-}
export MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-$((CONCUR * CONTEXT_LEN + MAX_RUNNING * 2048))}

CHUNK=${CHUNK:-4096}

# ------------------------------------------------------------------ DP attention
# DP_SIZE>1 gives each rank its own attention shard while the MoE stays EP8. It is
# how this model is served at high concurrency; DP_SIZE=1 leaves everything below
# inert and the recipe behaves exactly as before.
#
# Two knobs are load-bearing here, and neither default is right on TPU.
#
# 1. `--chunked-prefill-size` is a GLOBAL budget that sglang divides by dp_size.
#    Passing 4096 under DP8 chunks each rank at 512 -- half of vLLM's baseline and
#    an 8192-token request then needs 16 chunks per rank instead of 2. Scale it by
#    DP_SIZE so CHUNK means per-rank, as it does on the TP path.
#
# 2. `--prefill-delayer-max-delay-passes` defaults to 30 and wants ~250 here.
#    Under DP attention every rank runs the same forward mode each step, and
#    `DpPaddingMode.MAX_LEN` (forced on TPU for static shapes) pads every rank to
#    the global max -- so a prefill step where only k of 8 ranks have work still
#    costs the full 8-rank shape, wasting (8-k)/8. The delayer holds prefill
#    admission until ranks line up, but its `"mixed"` branch (only some ranks
#    prefillable -- exactly the misaligned case) is bounded ONLY by this pass
#    count. 30 passes x ~18 ms/decode step = ~530 ms of patience, while a rank
#    receives new prefill work only when one of its requests completes:
#    1024 tokens x ~29 ms ITL / ~15 running = ~1.96 s. It was ~4x too impatient.
#
#    Measured at ISL 8192 / OSL 1024 / CONCUR 128, DP8_EP8, 4 chips:
#      passes=30   2140.4 tok/s   52.5% of prefill steps have all 8 ranks active
#      passes=100  2379.5 tok/s   88.9%
#      passes=250  2424.4 tok/s   99.2%   (n=2: 2416.8 / 2432.0)
#    +13.3%, with median ITL 29.13 -> 27.66 and median TTFT unchanged. Alignment
#    saturates at 250, so raising it further buys nothing.
#
#    `--prefill-delayer-max-delay-ms` bounds only the `"all"` branch (the already
#    aligned case), which is why sweeping it measured null.
#
# NOT set, deliberately: `--load-balance-method total_tokens`. Measured -4.0%
# throughput and +157% median TTFT. round_robin spreads 8 consecutive arrivals
# over 8 distinct ranks, which is what alignment needs; dispatching to the
# momentarily-lightest rank piles arrivals onto one rank and drops all-8
# alignment 99.2% -> 92.0%.
DP_SIZE=${DP_SIZE:-1}
dp_args=()
if [ "$DP_SIZE" -gt 1 ]; then
  CHUNK=$((CHUNK * DP_SIZE))
  # ...but not past a whole context per rank, and keep it page-aligned: the
  # scaled-up global budget is meaningless beyond what one rank can hold.
  _ctx_cap=$((CONTEXT_LEN * DP_SIZE))
  [ "$CHUNK" -gt "$_ctx_cap" ] && CHUNK=$_ctx_cap
  CHUNK=$((CHUNK / 256 * 256))
  [ "$CHUNK" -lt 256 ] && CHUNK=256

  _per_rank_bs=$(((MAX_RUNNING + DP_SIZE - 1) / DP_SIZE))
  [ "$_per_rank_bs" -lt 1 ] && _per_rank_bs=1
  _per_rank_conc=$(((CONCUR + DP_SIZE - 1) / DP_SIZE))
  [ "$_per_rank_conc" -lt 1 ] && _per_rank_conc=1

  # `--max-total-tokens` is a PER-RANK cap -- `_apply_token_constraints` min()s it
  # against each rank's own profiled capacity -- unlike --max-running-requests and
  # --max-mamba-cache-size, which are global and divided by dp_size internally. So
  # the global budget computed above has to be divided here, or every rank sizes a
  # KV pool for the whole cluster's load. At CONCUR 128 / DP8 that is 1509888
  # tokens per rank instead of 188736, which leaves so little HBM that the server
  # dies during warmup: sglang's startup warmup sends a dummy multimodal request,
  # the one-time ViT compile it triggers asks for 1.36 GB, and only ~279 MB is
  # free. Measured -- the recipe could not launch at its own headline operating
  # point until this line existed.
  export MAX_TOTAL_TOKENS=${_MAX_TOTAL_FROM_CALLER:-$((_per_rank_conc * CONTEXT_LEN + _per_rank_bs * 2048))}

  # The compiled decode batch bucket is per rank, so a global MAX_RUNNING here
  # makes every rank precompile buckets it can never fill (bs up to 160 at
  # CONCUR 128, against a real per-rank batch of ~20) -- pure warmup time and HBM.
  export TORCH_COMPILE_MAX_BS=$_per_rank_bs
  dp_args=(
    --enable-dp-attention
    --dp-size "$DP_SIZE"
    --enable-dp-lm-head
    --enable-prefill-delayer
    --prefill-delayer-queue-min-ratio 0.5
    --prefill-delayer-max-delay-ms 2000
    --prefill-delayer-max-delay-passes "${DELAYER_PASSES:-250}"
  )
fi

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
# Already set to the per-rank batch by the DP block above when DP_SIZE>1.
export TORCH_COMPILE_MAX_BS=${TORCH_COMPILE_MAX_BS:-$MAX_RUNNING}

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

# Threshold BELOW which the MoE permute/unpermute use a dense one-hot matmul
# instead of a gather. Decode should be under it and take the cheap dense path;
# a prefill chunk should be far above it and take the SparseCore gather.
#
# It must SCALE WITH CONCURRENCY, and a fixed 1024 silently breaks at scale.
# fused_moe_op.py gates the two halves on different quantities:
#
#     use_dense_permute   = m       < threshold
#     use_dense_unpermute = m*topk  < threshold
#
# so with topk=10 a decode batch above 102 keeps the dense *permute* but drops the
# *unpermute* onto the gather path -- including its `argsort(argsort_idx)`. Measured
# at CONC=128 (batch 128, m*topk=1280), ISL 8192/OSL 1024, TP8/EP8:
#
#     threshold 1024 -> median ITL 36.28, TPOT 80.76   (unpermute on gather)
#     threshold 2048 -> median ITL 33.44, TPOT 77.83   (both dense)
#     threshold 4096 -> median ITL 33.42, TPOT 77.78   (both dense)
#
# i.e. **ITL -7.9%, TPOT -3.7%**, and 2048 vs 4096 are indistinguishable (0.04 ms
# across three samples) -- what matters is clearing `m*topk`, not the magnitude. No
# TTFT effect (median TTFT noise at this concurrency is ~2.5%).
#
# Rule: above `MAX_RUNNING * topk`, and at most CHUNK. The upper bound is `<=`, not
# `<`: prefill takes the dense permute iff `m < threshold`, so `threshold == CHUNK`
# still leaves a full chunk on the gather (this is the measured 4096 case above).
# Exceeding CHUNK is what must be avoided -- it would build a [chunk*topk, chunk]
# one-hot for prefill (168M elements at 4096) instead of a gather.
# topk is 10 for this checkpoint (`num_experts_per_tok`).
_TOPK=10
_DENSE_MIN=$(( MAX_RUNNING * _TOPK + 1 ))
_DENSE_THR=1024
while [ "$_DENSE_THR" -lt "$_DENSE_MIN" ] && [ $(( _DENSE_THR * 2 )) -le "$CHUNK" ]; do
  _DENSE_THR=$(( _DENSE_THR * 2 ))
done
if [ "$_DENSE_THR" -lt "$_DENSE_MIN" ]; then
  echo "WARNING: MAX_RUNNING=$MAX_RUNNING (x topk $_TOPK = $_DENSE_MIN) does not fit" \
       "under CHUNK=$CHUNK; MoE decode unpermute will use the gather path." >&2
fi
export SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD=${SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD:-$_DENSE_THR}

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
  "${dp_args[@]}"
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
