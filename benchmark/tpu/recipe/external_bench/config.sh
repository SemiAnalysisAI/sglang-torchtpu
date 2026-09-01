#!/usr/bin/env bash
# Single source of truth for the external-cluster TPU benchmark matrix
# (GPT-OSS-120B and Qwen3-Coder-480B x three serving recipes x two workloads).
# Sourced by serve_group.sh / bench_leg.sh / run_sweep.sh so every leg of every
# run uses identical settings.
#
# The matrix (task: "Add more benchmark tests for GPT-OSS models"):
#   BENCH_MODEL   gpt_oss_120b | qwen3_coder_480b
#   BENCH_RECIPE  1host_pc               1 host, tp=8, prefix caching
#                 1host_pc_spec_kvoff    + EAGLE3 spec + KV offload (raiden)
#                 pd_2p2d_pc_spec_kvoff  4 hosts: 2-host prefill + 2-host decode
#                                        engines (tp=16 each) over raiden P/D,
#                                        prefix caching + spec + KV offload
#   WORKLOAD      agentic  multi-turn agentic (the 480B nightly's shape:
#                          4k in + 1k out per round, sessions to 16k; long
#                          sessions ramp to LONG_TARGET_CONTEXT)
#                 8k1k     single-turn 8k in / 1k out; "long requests" are
#                          single-turn LONG_TARGET_CONTEXT-input requests
#   PROFILE       ci     full sweep grid {1,16,64} x {0,1,4} (long>bs skipped)
#                 e2e    one point: concurrency 64 x 4 long requests
#                 smoke  tiny plumbing check
#
# Both workloads run through gen_agentic_workload.py + the agentic-trace
# dataset: the 8k1k shape simply sets the per-round input to the full target so
# every session is a single round. That reuses the whole measurement path
# (per-leg Prometheus deltas incl. prefix-cache hit rate, dump_meta records,
# summarize.py) instead of maintaining a second harness.
set -euo pipefail

EXT_BENCH_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TPU_BENCH_DIR=$(cd -- "$EXT_BENCH_DIR/../.." && pwd)   # benchmark/tpu
REPO_ROOT=$(cd -- "$TPU_BENCH_DIR/../.." && pwd)
# Reuse the agentic nightly's tools (trace generator, metrics probe, summary).
NIGHTLY_TOOLS_DIR="$TPU_BENCH_DIR/recipe/qwen3_coder_480b/nightly"

BENCH_MODEL=${BENCH_MODEL:?set BENCH_MODEL=gpt_oss_120b|qwen3_coder_480b}
BENCH_RECIPE=${BENCH_RECIPE:?set BENCH_RECIPE=1host_pc|1host_pc_spec_kvoff|pd_2p2d_pc_spec_kvoff}
WORKLOAD=${WORKLOAD:-agentic}
PROFILE=${PROFILE:-ci}

PYTHON=${PYTHON:-python3}

# ---------------------------------------------------------------------------
# Sweep axes (the task's grid). e2e keeps only the point the 4-host bring-up
# is validated with; smoke is a plumbing check.
# ---------------------------------------------------------------------------
case "$PROFILE" in
    ci)
        SWEEP_BS=${SWEEP_BS:-"1 16 64"}
        SWEEP_LONG=${SWEEP_LONG:-"0 1 4"}
        MAX_LEG_SECONDS=${MAX_LEG_SECONDS:-900}
        ;;
    e2e)
        SWEEP_BS=${SWEEP_BS:-"64"}
        SWEEP_LONG=${SWEEP_LONG:-"4"}
        MAX_LEG_SECONDS=${MAX_LEG_SECONDS:-1800}
        ;;
    smoke)
        SWEEP_BS=${SWEEP_BS:-"1 2"}
        SWEEP_LONG=${SWEEP_LONG:-"0 1"}
        MAX_LEG_SECONDS=${MAX_LEG_SECONDS:-900}
        ;;
    *)
        echo "ERROR: unknown PROFILE='$PROFILE' (want ci|e2e|smoke)" >&2
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# Workload shape. Both are expressed in the agentic generator's terms.
# ---------------------------------------------------------------------------
# LONG_TARGET_CONTEXT=65536 for both models keeps the c=64 working set at or
# under ~1.25M tokens (60x16k normal + 4x65k long), which the per-model KV
# pools below are sized to hold without retracting -- a retraction voids the
# leg (see the retraction gate in the tests). GPT-OSS's native window is 128k,
# so no rope override is needed for either model at this target.
LONG_TARGET_CONTEXT=${LONG_TARGET_CONTEXT:-65536}
case "$WORKLOAD" in
    agentic)
        TURN_INPUT_TOKENS=${TURN_INPUT_TOKENS:-4096}
        TURN_OUTPUT_TOKENS=${TURN_OUTPUT_TOKENS:-1024}
        TARGET_CONTEXT=${TARGET_CONTEXT:-16384}          # 3 rounds
        # ci ramps long sessions at 16k/round (~4 rounds to 64k) to bound wall
        # clock, same trade the 480B nightly's ci profile makes.
        LONG_RAMP_TOKENS=${LONG_RAMP_TOKENS:-16384}
        ;;
    8k1k)
        TURN_INPUT_TOKENS=${TURN_INPUT_TOKENS:-8192}
        TURN_OUTPUT_TOKENS=${TURN_OUTPUT_TOKENS:-1024}
        # target == per-round input -> every session is exactly one round, so
        # the leg is a plain 8k-in/1k-out serving benchmark.
        TARGET_CONTEXT=${TARGET_CONTEXT:-8192}
        # long requests are likewise SINGLE-turn: one request with the whole
        # long context as input.
        LONG_RAMP_TOKENS=${LONG_RAMP_TOKENS:-$LONG_TARGET_CONTEXT}
        ;;
    *)
        echo "ERROR: unknown WORKLOAD='$WORKLOAD' (want agentic|8k1k)" >&2
        exit 2
        ;;
esac

NORMAL_REPLAYS=${NORMAL_REPLAYS:-auto}
MAX_NORMAL_REPLAYS=${MAX_NORMAL_REPLAYS:-8}
TRACE_CACHE_DIR=${TRACE_CACHE_DIR:-${XDG_CACHE_HOME:-/tmp}/sglang_tpu_ext_bench_traces}

# ---------------------------------------------------------------------------
# Model fragments. MODEL / DRAFT_MODEL default to the gcsfuse bucket layout
# the external JobSet mounts; the template's injected MODEL env wins.
# ---------------------------------------------------------------------------
GCS_WEIGHTS_ROOT=${GCS_WEIGHTS_ROOT:-/gcs-weights/weights}
case "$BENCH_MODEL" in
    gpt_oss_120b)
        MODEL=${MODEL:-$GCS_WEIGHTS_ROOT/gpt-oss-120b}
        DRAFT_MODEL=${DRAFT_MODEL:-$GCS_WEIGHTS_ROOT/EAGLE3-gpt-oss-120b-bf16}
        # MXFP4 is declared by the checkpoint's own quantization_config.
        QUANTIZATION=
        KV_CACHE_DTYPE=${KV_CACHE_DTYPE-fp8_e4m3}
        # rpa_v3: the only TPU backend with attention sinks (GPT-OSS needs
        # them) AND the HiCache reload-commit hook KV offload requires.
        ATTENTION_BACKEND=pallas_tpu_rpa_v3
        PAGE_SIZE=${PAGE_SIZE:-64}
        CONTEXT_LEN=${CONTEXT_LEN:-131072}
        # ~15.2 KB KV/token/rank at tp=8 (fp8 + hybrid-SWA second pool). 10x
        # the context length, per the gpt_oss_120b recipe's HBM guidance:
        # leaves the EXTEND-graph temporaries their headroom at mem 0.6.
        # Scaled by TP below (per-rank bytes stay constant).
        MAX_TOTAL_TOKENS_TP8=1310720
        MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.6}
        # SparseCore ragged-gather MoE cannot lower for GPT-OSS's EXTEND graph.
        export SGLANG_TPU_FUSED_MOE_SC_GATHER=${SGLANG_TPU_FUSED_MOE_SC_GATHER:-0}
        MODEL_EXTRA_ARGS="--reasoning-parser gpt-oss"
        ;;
    qwen3_coder_480b)
        MODEL=${MODEL:-$GCS_WEIGHTS_ROOT/Qwen3-Coder-480B-A35B-Instruct-FP8}
        DRAFT_MODEL=${DRAFT_MODEL:-$GCS_WEIGHTS_ROOT/SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI}
        QUANTIZATION=fp8
        KV_CACHE_DTYPE=${KV_CACHE_DTYPE-fp8_e4m3}
        # rpa_v3 for KV-offload compatibility and because rpa_batched
        # mis-computes non-first segments of packed EXTEND batches
        # (/cache/share/tpu-batched-decode-corruption-fix.md).
        ATTENTION_BACKEND=pallas_tpu_rpa_v3
        PAGE_SIZE=${PAGE_SIZE:-128}
        # Native window is 262144; 131072 needs no YaRN and matches GPT-OSS.
        CONTEXT_LEN=${CONTEXT_LEN:-131072}
        # ~15.9 KB KV/token/rank at tp=8 (62 layers x 1 kv head x 128 x K+V).
        # Scaled by TP below (per-rank bytes stay constant).
        MAX_TOTAL_TOKENS_TP8=1179648
        MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
        MODEL_EXTRA_ARGS="--tpu-fp8-runtime channelwise"
        ;;
    *)
        echo "ERROR: unknown BENCH_MODEL='$BENCH_MODEL'" >&2
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# Recipe fragments: parallelism + features.
# ---------------------------------------------------------------------------
SPEC_ARGS="--speculative-algorithm EAGLE3 \
--speculative-draft-model-path $DRAFT_MODEL \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4"

case "$BENCH_RECIPE" in
    1host_pc)
        RECIPE_HOSTS=1; TP_SIZE=8; EP_SIZE=8
        PREFIX_CACHE=1; KV_OFFLOAD=0; SPEC_DECODE=0; PD_DISAGG=0
        ;;
    1host_pc_spec_kvoff)
        RECIPE_HOSTS=1; TP_SIZE=8; EP_SIZE=8
        PREFIX_CACHE=1; KV_OFFLOAD=1; SPEC_DECODE=1; PD_DISAGG=0
        ;;
    pd_2p2d_pc_spec_kvoff)
        # FOUR single-host tp=8 engines (2 prefill + 2 decode) behind the
        # mini-LB -- the raiden-validated topology. Two-host tp=16 engines
        # subdivided from the slice bring up, but their raiden KV pulls never
        # complete (see serve_group.sh).
        RECIPE_HOSTS=4; TP_SIZE=8; EP_SIZE=8     # per engine, 1 host each
        PREFIX_CACHE=1; KV_OFFLOAD=1; SPEC_DECODE=1; PD_DISAGG=1
        ;;
    *)
        echo "ERROR: unknown BENCH_RECIPE='$BENCH_RECIPE'" >&2
        exit 2
        ;;
esac

# Spec verify/draft graph capture is expensive and host-RAM hungry when the
# grid is unbounded; bound it to the sweep's shapes (incl. bs=64 -- the EAGLE3
# bs-64 crash is fixed by #546) and allow spec recompiles for the tail.
if [[ "$SPEC_DECODE" == "1" ]]; then
    export TPU_COMPILE_BS=${TPU_COMPILE_BS:-"1 2 4 16 64"}
    export SGLANG_TPU_ALLOW_SPEC_RECOMPILE=${SGLANG_TPU_ALLOW_SPEC_RECOMPILE:-1}
fi

# KV pool, scaled with TP so the per-rank HBM cost stays what the tp=8 recipe
# guidance was measured at (scaling a tp=8 engine's pool past that ate the
# ~45 GB the 8k EXTEND-graph capture needs and OOM'd bring-up -- iteration 5).
# In the P/D recipe each single-host decode engine holds ~half the running
# batch (two decodes behind the LB), so the tp=8 pool covers the c64 working
# set with ~2x headroom.
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS-$((MAX_TOTAL_TOKENS_TP8 * TP_SIZE / 8))}
MAX_TOTAL_TOKENS_PREFILL=${MAX_TOTAL_TOKENS_PREFILL-$MAX_TOTAL_TOKENS_TP8}

# P/D: a long request's KV handoff is chunky (a 64k-token context is ~1 GB
# per engine at fp8) and 4 of them land together at the c64-long4 point, so
# the default 300s disaggregation waiting timeout trips at the leg tail
# ("Raiden transfer failed on another attention TP/CP rank", observed
# 2026-09-01 on the first 4-host E2E). sglang's own log message recommends
# raising it; 900s bounds a wedged transfer without failing healthy ones.
if [[ "$PD_DISAGG" == "1" ]]; then
    # Two separate knobs bound the two sides of a handoff: the decode's wait
    # for its KV (WAITING) and the prefill's wait for the decode to ask
    # (BOOTSTRAP -- raiden is consumer-pull, so the prefill's handshake window
    # is fail_decode_silence(bootstrap_timeout), raiden/conn.py:2529). Raising
    # only the first still failed prefill-side at 300s (iteration 7).
    export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-900}
    export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-900}
fi

# KV offload (HiCache) io backend: raiden skips the fused pool's whole-pool D2H
# precompile staging (which needs ~2x the KV pool in HBM); ratio 1 keeps the
# host pool inside the pod memory request.
HICACHE_IO_BACKEND=${HICACHE_IO_BACKEND:-raiden}
HICACHE_RATIO=${HICACHE_RATIO:-1}
HICACHE_MEM_LAYOUT=page_first_fused

# Server admission: sized for the sweep's largest concurrency; the client's
# semaphore is what varies per leg (one server serves every leg).
_max_bs=1
for _b in $SWEEP_BS; do (( _b > _max_bs )) && _max_bs=$_b; done
MAX_CONCURRENCY=$_max_bs
MAX_RUNNING=${MAX_RUNNING:-$MAX_CONCURRENCY}
TORCH_COMPILE_MAX_BS=${TORCH_COMPILE_MAX_BS:-$MAX_CONCURRENCY}

# One cold 8k prompt per prefill pass; a cache-hit round's remainder still
# lands in one chunk. 8192 is also the EXTEND-graph HBM ceiling on tpu7x.
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}

# ---------------------------------------------------------------------------
# Ports / endpoints. The external JobSet's rank-0 health-gates and benches
# 127.0.0.1:30000, so that port is the single-host server, or the mini-LB in
# the P/D recipe (engines then sit on the PD_* ports).
# ---------------------------------------------------------------------------
PORT=${PORT:-30000}
PD_PREFILL_PORT=${PD_PREFILL_PORT:-30001}
PD_DECODE_PORT=${PD_DECODE_PORT:-30002}
PD_BOOTSTRAP_PORT=${PD_BOOTSTRAP_PORT:-8998}

HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-3600}
ENABLE_METRICS=1        # required: the per-leg deltas are how a leg proves
ENABLE_CACHE_REPORT=1   # prefix caching / KV offload actually fired

# ---------------------------------------------------------------------------
# Sanity.
# ---------------------------------------------------------------------------
for _b in $SWEEP_BS; do
    case "$_b" in ''|*[!0-9]*) echo "ERROR: SWEEP_BS has non-numeric '$_b'" >&2; exit 2;; esac
done
for _l in $SWEEP_LONG; do
    case "$_l" in ''|*[!0-9]*) echo "ERROR: SWEEP_LONG has non-numeric '$_l'" >&2; exit 2;; esac
done
_round_headroom=$(( (LONG_RAMP_TOKENS + TURN_OUTPUT_TOKENS) / 2 ))
if (( LONG_TARGET_CONTEXT + _round_headroom > CONTEXT_LEN )); then
    echo "ERROR: LONG_TARGET_CONTEXT=$LONG_TARGET_CONTEXT (+$_round_headroom" \
         "rounding headroom) exceeds CONTEXT_LEN=$CONTEXT_LEN" >&2
    exit 2
fi

MODEL_SLUG=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/__*/_/g; s/^_//; s/_$//')

# ---------------------------------------------------------------------------
# log_config — dumped into every run dir / server log so a result can never be
# read without the settings that produced it.
# ---------------------------------------------------------------------------
log_config() {
    cat <<EOF
==== external tpu benchmark config ====
  bench_model / recipe   = $BENCH_MODEL / $BENCH_RECIPE
  workload / profile     = $WORKLOAD / $PROFILE
  model                  = $MODEL
  quantization           = ${QUANTIZATION:-<checkpoint>}   kv_cache_dtype = $KV_CACHE_DTYPE
  tp / ep (per engine)   = $TP_SIZE / $EP_SIZE   hosts = $RECIPE_HOSTS
  attention_backend      = $ATTENTION_BACKEND   page_size = $PAGE_SIZE
  context_len            = $CONTEXT_LEN   max_total_tokens = ${MAX_TOTAL_TOKENS:-<derived>}
  mem_fraction_static    = $MEM_FRACTION_STATIC   chunked_prefill = $CHUNKED_PREFILL_SIZE
  max_running / compile_bs = $MAX_RUNNING / $TORCH_COMPILE_MAX_BS
  --- features ---
  prefix_caching         = $([[ "$PREFIX_CACHE" == "1" ]] && echo ENABLED || echo disabled)
  kv_offload (hicache)   = $([[ "$KV_OFFLOAD" == "1" ]] && echo "ENABLED (io=$HICACHE_IO_BACKEND ratio=$HICACHE_RATIO)" || echo disabled)
  spec_decode (EAGLE3)   = $([[ "$SPEC_DECODE" == "1" ]] && echo "ENABLED (draft=$DRAFT_MODEL)" || echo disabled)
  pd_disaggregation      = $([[ "$PD_DISAGG" == "1" ]] && echo "ENABLED (2P+2D raiden, tp=$TP_SIZE/engine)" || echo disabled)
  --- workload ---
  turn input / output    = $TURN_INPUT_TOKENS / $TURN_OUTPUT_TOKENS tokens
  target context         = $TARGET_CONTEXT (normal) / $LONG_TARGET_CONTEXT (long)
  long ramp tokens/round = $LONG_RAMP_TOKENS
  concurrency sweep      = $SWEEP_BS
  long-request sweep     = $SWEEP_LONG
  max_leg_seconds        = $MAX_LEG_SECONDS   health_timeout = $HEALTH_TIMEOUT
=======================================
EOF
}
