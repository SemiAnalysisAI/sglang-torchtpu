#!/usr/bin/env bash
# Reproduce the measured Qwen3.5-397B-A17B-FP8 numbers on a fresh host.
#
#   CONCUR=4   DP_SIZE=1 bash repro.sh     # ~429 tok/s, median ITL 7.65
#   CONCUR=128 DP_SIZE=8 bash repro.sh     # ~2424 tok/s, median ITL 27.7
#
# server.sh alone is NOT sufficient to reproduce. Three things live outside it
# and each one silently changes the result:
#
#   1. THE CHIP PIN. Every published number was measured on a 4-chip 2x2x1 slice
#      (worker 0 of a 2-host pod). launch_server.sh deliberately *unsets*
#      TPU_VISIBLE_CHIPS / TPU_CHIPS_PER_HOST_BOUNDS / TPU_HOST_BOUNDS when
#      TP_SIZE>1 (it would mask chiplets and break the slicebuilder handshake),
#      so the single-host decision has to ride on TPU_WORKER_HOSTNAMES /
#      TPU_PROCESS_ADDRESSES, which it does not unset. On a host that is already
#      a single 4-chip node this block is harmless; on a pod it is essential.
#      Set PIN_CHIPS=0 to serve the whole slice instead.
#
#   2. THE COMPILE CACHE IS NOT KEYED ON CONFIG. The decoder comes back through
#      inductor's codecache as a pre-serialised executable, so a cache warmed
#      under different LIBTPU_INIT_ARGS -- or a different torch_tpu build --
#      is replayed verbatim and you measure the OLD build. Caches are namespaced
#      per config below; wipe them whenever torch_tpu changes.
#
#   3. MAX_RUNNING. The recipe derives several limits from it, and the DP and TP
#      paths want different headroom over CONCUR (see below).
#
# torch_tpu version matters: dev20260707090224 measured ~8% faster on decode
# than dev20260804175944 for this model (median ITL 7.65 vs 8.31 at CONCUR=4),
# with libtpu held identical. Pin it explicitly before comparing anything.
set -euo pipefail

CONCUR=${CONCUR:-4}
DP_SIZE=${DP_SIZE:-1}
ISL=${ISL:-8192}
OSL=${OSL:-1024}
PORT=${PORT:-8000}
PIN_CHIPS=${PIN_CHIPS:-1}

RECIPE_DIR=$(cd "$(dirname "$0")" && pwd)
SGL_REPO=${SGL_REPO:-$(cd "$RECIPE_DIR/../../../.." && pwd)}
PYTHON=${PYTHON:-$SGL_REPO/.venv/bin/python}
CACHE_ROOT=${CACHE_ROOT:-${TMPDIR:-/tmp}/qwen3_5_397b_repro}

# Bench client. Defaults to this recipe's own bench.sh (sglang's bench_serving,
# random-ids dataset), which is what makes this script portable. The published
# numbers above were produced with the InferenceX client instead; point BENCH at
# its bench.sh to compare against them directly. Either client is fine for an
# A/B as long as both arms use the same one.
#
# BENCH_VENV is for the InferenceX client only: it shells out to `python3`, and
# these images have no system python3, so it needs a venv on PATH.
BENCH=${BENCH:-$RECIPE_DIR/bench.sh}
BENCH_VENV=${BENCH_VENV:-}

# Model. MODEL_DIR wins if set; otherwise pass the HF repo id straight through
# and let the loader resolve it out of the usual HF cache.
MODEL_DIR=${MODEL_DIR:-Qwen/Qwen3.5-397B-A17B-FP8}

# MAX_RUNNING: CONCUR+4 on the TP path (measured), CONCUR+32 under DP -- a DP
# rank needs slack to stay busy while its peers finish their prefill.
if [ "$DP_SIZE" -gt 1 ]; then
  export MAX_RUNNING=${MAX_RUNNING:-$((CONCUR + 32))}
else
  export MAX_RUNNING=${MAX_RUNNING:-$((CONCUR + 4))}
fi

# (1) single-host 4-chip pin.
#
# These are assigned UNCONDITIONALLY, not with `${VAR:-default}`. A pod host
# already exports all of them describing the whole slice -- e.g.
# TPU_WORKER_HOSTNAMES listing both workers of a tpu7x-16 -- and
# launch_server.sh derives NNODES from the number of entries there
# (launch_server.sh:151). Deferring to the inherited value therefore makes the
# pin a silent no-op: the server comes up as node 0 of 2 and blocks forever in
# "Init torch distributed begin" waiting for a peer that was never started.
# PIN_CHIPS=0 is the way to opt out; a half-applied pin is not a useful state.
if [ "$PIN_CHIPS" = "1" ]; then
  export TPU_HOST_BOUNDS=1,1,1
  export TPU_TOPOLOGY=2x2x1
  export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
  export TPU_VISIBLE_CHIPS=0,1,2,3
  export TPU_WORKER_ID=0
  export TPU_WORKER_HOSTNAMES=localhost
  export TPU_PROCESS_ADDRESSES=localhost:8471
fi

# (2) per-config compile cache. Changing torch_tpu invalidates these; delete them.
CACHE_NS=${CACHE_NS:-c${CONCUR}_dp${DP_SIZE}}
export TORCH_TPU_TIER3_ROOT=${TORCH_TPU_TIER3_ROOT:-$CACHE_ROOT/torch_tpu_$CACHE_NS}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/inductor_$CACHE_NS}
mkdir -p "$TORCH_TPU_TIER3_ROOT" "$TORCHINDUCTOR_CACHE_DIR"

echo "=== repro: CONCUR=$CONCUR DP_SIZE=$DP_SIZE ISL=$ISL OSL=$OSL MAX_RUNNING=$MAX_RUNNING"
echo "=== repo=$SGL_REPO commit=$(git -C "$SGL_REPO" rev-parse --short HEAD 2>/dev/null)"
echo "=== torch_tpu=$("$PYTHON" -c 'import importlib.metadata as m;print(m.version("torch_tpu"))' 2>/dev/null)"
echo "=== cache_ns=$CACHE_NS bench=$BENCH"

# setsid so the trap can take down the whole server process group; killing the
# launcher shell alone leaves the engine workers holding the chips.
setsid env MODEL="$MODEL_DIR" CONCUR="$CONCUR" DP_SIZE="$DP_SIZE" \
  CONTEXT_LEN=$((ISL + OSL + 20)) PORT="$PORT" PYTHON="$PYTHON" \
  bash "$RECIPE_DIR/server.sh" &
SERVER_PID=$!
trap 'kill -TERM -- -"$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true' EXIT

echo "=== waiting for /health (cold start is ~25 min)"
until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died" >&2; exit 1; }
  sleep 20
done

bench_env=(
  MODEL="$MODEL_DIR" BASE_URL="http://127.0.0.1:$PORT" PYTHON="$PYTHON"
  ISL="$ISL" OSL="$OSL" PORT="$PORT"
  # CONCUR is this recipe's name for it, CONC is InferenceX's.
  CONCUR="$CONCUR" CONC="$CONCUR"
)
if [ -n "$BENCH_VENV" ]; then
  bench_env+=(VIRTUAL_ENV="$BENCH_VENV" PATH="$BENCH_VENV/bin:$PATH")
fi
env "${bench_env[@]}" bash "$BENCH"
