#!/usr/bin/env bash
# Single source of truth for the TPU benchmark matrix
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
# THE TIME-BUDGET CHAIN. Every timing knob in this harness must respect
#
#   worst-case sweep  <=  SWEEP_TIMEOUT  <  TIMEOUT_PER_FILE  <  JobSet deadline
#
# where worst-case sweep = (number of attempted legs) x MAX_LEG_SECONDS.
# TIMEOUT_PER_FILE (5400) is declared literally in every test file's
# MULTIHOST_LAUNCH (the CI scanner reads it via AST, so it cannot be computed
# there); test/registered/unit/test_bench_matrix.py ASSERTS the whole
# chain, so a change to any link that breaks another fails CI instead of
# silently truncating a benchmark.
# ---------------------------------------------------------------------------
# Deadline for one leg's bench client. Not a typical duration (healthy c64
# legs measured 150-500s) -- the cap only bites hung legs, and it must leave
# the ci profile's worst case (8 attempted legs) inside SWEEP_TIMEOUT. Sized
# for the gpt-oss 120k long target (7 ramp-rounds per long session).
MAX_LEG_SECONDS_CI=900
# e2e runs a single leg, so it can afford a far larger cap for bring-up-time
# experiments on that one point.
MAX_LEG_SECONDS_SINGLE_POINT=1800
# Poll cadence for /health waits (server bring-up is minutes; finer polling
# is noise) and the per-probe curl budget.
HEALTH_POLL_INTERVAL=10
CURL_MAX_TIME=10
# SIGKILL grace after a leg's deadline SIGTERM (a stuck aiohttp client
# ignores TERM).
LEG_KILL_GRACE=60

# ---------------------------------------------------------------------------
# Sweep axes (the task's grid). e2e keeps only the point the 4-host bring-up
# is validated with; smoke is a plumbing check.
# ---------------------------------------------------------------------------
case "$PROFILE" in
    ci)
        SWEEP_BS=${SWEEP_BS:-"1 16 64"}
        SWEEP_LONG=${SWEEP_LONG:-"0 1 4"}
        MAX_LEG_SECONDS=${MAX_LEG_SECONDS:-$MAX_LEG_SECONDS_CI}
        ;;
    e2e)
        SWEEP_BS=${SWEEP_BS:-"64"}
        SWEEP_LONG=${SWEEP_LONG:-"4"}
        MAX_LEG_SECONDS=${MAX_LEG_SECONDS:-$MAX_LEG_SECONDS_SINGLE_POINT}
        ;;
    smoke)
        SWEEP_BS=${SWEEP_BS:-"1 2"}
        SWEEP_LONG=${SWEEP_LONG:-"0 1"}
        MAX_LEG_SECONDS=${MAX_LEG_SECONDS:-$MAX_LEG_SECONDS_CI}
        ;;
    *)
        echo "ERROR: unknown PROFILE='$PROFILE' (want ci|e2e|smoke)" >&2
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# Workload shape. Both are expressed in the agentic generator's terms.
# ---------------------------------------------------------------------------
# The long-context target is PER MODEL: it is capped by the native window
# minus output/rounding/template headroom, and by what the model's KV pool
# holds retraction-free at c64-long4 (a retraction voids the leg -- see the
# retraction gate in the tests). Model validity itself is enforced by the
# model-fragments case below.
case "$BENCH_MODEL" in
    gpt_oss_120b)
        # 7 ramp-rounds -> final context ~122k, the largest the 131072 window
        # fits with headroom (the asked-for "128k" is unreachable: the last
        # round's output plus rounding must also fit). Requires the enlarged
        # pool in the model fragment (~1.41M-token worst case at c64-long4).
        LONG_TARGET_CONTEXT=${LONG_TARGET_CONTEXT:-120832}
        ;;
    *)
        # qwen3_coder_480b stays at 64k: its tp8 pool already sits at the
        # mem-fraction budget's edge (~21 GB), so a 120k target first needs a
        # pool + mem-fraction re-measurement on cells not yet E2E-run.
        LONG_TARGET_CONTEXT=${LONG_TARGET_CONTEXT:-65536}
        ;;
esac
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

# How many FRESH normal sessions each non-long concurrency slot plays
# back-to-back, so mixed-length legs keep all slots busy while the long
# sessions run (otherwise the aggregate would average a full batch with a
# mostly-idle tail while still claiming concurrency BS). "auto" sizes it from
# the long/normal round ratio; capped so trace size stays bounded. Note: this
# is a round-ratio heuristic -- it equalizes durations only when per-round
# cost is similar (true for the agentic shape; the 8k1k long requests still
# outlive their batch). The 480B nightly calls the same knob NORMAL_REPLAYS.
SESSIONS_PER_SLOT=${SESSIONS_PER_SLOT:-${NORMAL_REPLAYS:-auto}}
MAX_SESSIONS_PER_SLOT=${MAX_SESSIONS_PER_SLOT:-${MAX_NORMAL_REPLAYS:-8}}
TRACE_CACHE_DIR=${TRACE_CACHE_DIR:-${XDG_CACHE_HOME:-/tmp}/sglang_tpu_bench_matrix_traces}

# ---------------------------------------------------------------------------
# Model fragments. MODEL / DRAFT_MODEL default to the gcsfuse bucket layout
# the CI JobSet mounts; the template's injected MODEL env wins.
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
        # ~15.2 KB KV/token/rank at tp=8 (fp8 + hybrid-SWA second pool).
        # Sized for the 120k long target: c64-long4 worst case is
        # 60 x ~15.4k + 4 x ~122k ~= 1.41M tokens, plus margin. Costs ~24 GB
        # per rank and leaves ~41 GB for the EXTEND-graph capture -- above
        # the measured OOM point (38.3 GB, 2026-09-01 iteration 5) but below
        # the comfortable 45 GB line; if the capture OOMs, halve
        # CHUNKED_PREFILL_SIZE rather than shrinking this (a retraction voids
        # legs; a smaller chunk only slows cold prefills).
        # Scaled by TP below (per-rank bytes stay constant).
        MAX_TOTAL_TOKENS_TP8=1572864
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
# --watchdog-timeout as in the validated spec gates: the verify/draft graph
# captures keep the scheduler busy past the 300s default, which trips the
# (soft) watchdog into repeated stack-dump attempts mid-compile.
SPEC_ARGS="--speculative-algorithm EAGLE3 \
--speculative-draft-model-path $DRAFT_MODEL \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4 \
--watchdog-timeout 7200"

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
        # mini-LB -- the raiden-validated topology. (Two-host tp=16 engines
        # subdivided from the slice also bring up; see serve_group.sh for why
        # they are untested rather than known-broken.)
        RECIPE_HOSTS=4; TP_SIZE=8; EP_SIZE=8     # per engine, 1 host each
        PD_PREFILL_ENGINES=2   # the mini-LB pairs requests with a random one
        PREFIX_CACHE=1; KV_OFFLOAD=1; SPEC_DECODE=1; PD_DISAGG=1
        ;;
    *)
        echo "ERROR: unknown BENCH_RECIPE='$BENCH_RECIPE'" >&2
        exit 2
        ;;
esac
# Prefill engines a request can be routed to; the agentic prefix-hit gate
# scales its reachable rate by 1/N because each engine caches on its own.
PD_PREFILL_ENGINES=${PD_PREFILL_ENGINES:-1}

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

# P/D disaggregation timeouts: stock 300s. The 2026-09-01 external runs in
# which every request failed exactly at the timeout boundary (216.5 tok/s at
# 300s, 72.7 tok/s at 900s, empty response bodies) were the IPv6 ZMQ
# rendezvous bug fixed in PR #606 -- the pods here are IPv6-only and the
# raiden connector's PULL_REQ/BLOCKS were never delivered -- not slow
# transfers, so the window is a safety net only. Overridable for experiments.
#
# Two separate knobs bound the two sides of a handoff: the decode's wait for
# its KV (WAITING) and the prefill's wait for the decode to ask (BOOTSTRAP --
# raiden is consumer-pull; fail_decode_silence(bootstrap_timeout),
# raiden/conn.py:2529).
if [[ "$PD_DISAGG" == "1" ]]; then
    [[ -n "${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-}" ]] && export SGLANG_DISAGGREGATION_WAITING_TIMEOUT
    [[ -n "${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-}" ]] && export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT
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
# Ports / endpoints. The CI JobSet's rank-0 health-gates and benches
# 127.0.0.1:30000, so that port is the single-host server, or the mini-LB in
# the P/D recipe (engines then sit on the PD_* ports).
# ---------------------------------------------------------------------------
PORT=${PORT:-30000}
PD_PREFILL_PORT=${PD_PREFILL_PORT:-30001}
PD_DECODE_PORT=${PD_DECODE_PORT:-30002}
PD_BOOTSTRAP_PORT=${PD_BOOTSTRAP_PORT:-8998}
# raiden data-plane base port: each rank listens at base+rank (must not
# collide with any of the HTTP/bootstrap ports above).
RAIDEN_KV_BASE_PORT=${RAIDEN_KV_BASE_PORT:-31000}


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
# A long session's ACTUAL final context is rounds x (ramp + output) -- the
# same nearest-round arithmetic as bench_leg.sh / gen_agentic_workload.py --
# plus a small per-round chat-template overhead. Assert on that rather than a
# fixed half-round headroom, which over-blocks single-shot (1-round) longs.
_long_growth=$(( LONG_RAMP_TOKENS + TURN_OUTPUT_TOKENS ))
_long_rounds=$(( (LONG_TARGET_CONTEXT + _long_growth / 2) / _long_growth ))
(( _long_rounds < 1 )) && _long_rounds=1
_TEMPLATE_OVERHEAD_PER_ROUND=32   # measured ~10 tok/round on the 480B shape; padded
_long_final=$(( _long_rounds * (_long_growth + _TEMPLATE_OVERHEAD_PER_ROUND) ))
if (( _long_final > CONTEXT_LEN )); then
    echo "ERROR: a long session reaches ~$_long_final tokens" \
         "($_long_rounds rounds x ($_long_growth+$_TEMPLATE_OVERHEAD_PER_ROUND))" \
         "> CONTEXT_LEN=$CONTEXT_LEN; lower LONG_TARGET_CONTEXT or LONG_RAMP_TOKENS" >&2
    exit 2
fi

MODEL_SLUG=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/__*/_/g; s/^_//; s/_$//')

# ---------------------------------------------------------------------------
# log_config — dumped into every run dir / server log so a result can never be
# read without the settings that produced it.
# ---------------------------------------------------------------------------
log_config() {
    cat <<EOF
==== tpu benchmark matrix config ====
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
