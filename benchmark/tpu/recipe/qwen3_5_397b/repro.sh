#!/usr/bin/env bash
# Reproduce the measured Qwen3.5-397B-A17B-FP8 numbers on a fresh host.
#
#   CONCUR=4   DP_SIZE=1 bash repro.sh     # ~429 tok/s, median ITL 7.65
#   CONCUR=128 DP_SIZE=8 bash repro.sh     # ~2424 tok/s, median ITL 27.7
#
# server.sh alone is NOT sufficient to reproduce. Three things live outside it
# and each one silently changes the result:
#
#   1. THE CHIP PIN. Every number was measured on a 4-chip 2x2x1 slice (worker 0
#      of a 2-host pod). launch_server.sh deliberately *unsets*
#      TPU_VISIBLE_CHIPS / TPU_CHIPS_PER_HOST_BOUNDS / TPU_HOST_BOUNDS when
#      TP_SIZE>1 (it would mask chiplets and break the slicebuilder handshake),
#      so the single-host decision has to ride on TPU_WORKER_HOSTNAMES /
#      TPU_PROCESS_ADDRESSES, which it does not unset. On a host that is already
#      a single 4-chip node this block is harmless; on a pod it is essential.
#
#   2. THE COMPILE CACHE IS NOT KEYED ON CONFIG. The decoder comes back through
#      inductor's codecache as a pre-serialised executable, so a cache warmed
#      under different LIBTPU_INIT_ARGS -- or a different torch_tpu build --
#      is replayed verbatim and you measure the OLD build. Caches are namespaced
#      per config below; wipe them whenever torch_tpu changes.
#
#   3. THE BENCH CLIENT. There is no system python3 on these images; the
#      InferenceX client runs out of a separate venv.
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

SGL_REPO=${SGL_REPO:-$(cd "$(dirname "$0")/../../../.." && pwd)}
MODEL_DIR=${MODEL_DIR:-$(ls -d /cache/hf/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/*/ 2>/dev/null | head -1)}
BENCH_VENV=${BENCH_VENV:-/cache/work/vllm_env}
IX_BENCH=${IX_BENCH:-/cache/work/tpu-inference/tests/e2e/benchmarking/inferencex/qwen3.5/bench.sh}
INFERENCEX_REPO=${INFERENCEX_REPO:-/cache/work/InferenceX}
export INFERENCEX_REPO

[ -n "$MODEL_DIR" ] || { echo "set MODEL_DIR (Qwen3.5-397B-A17B-FP8 snapshot dir)" >&2; exit 1; }

# (1) single-host 4-chip pin
export TPU_HOST_BOUNDS=1,1,1 TPU_TOPOLOGY=2x2x1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_VISIBLE_CHIPS=0,1,2,3 TPU_WORKER_ID=0
export TPU_WORKER_HOSTNAMES=localhost TPU_PROCESS_ADDRESSES=localhost:8471

# (2) per-config compile cache. Changing torch_tpu invalidates these; delete them.
CACHE_NS=${CACHE_NS:-c${CONCUR}_dp${DP_SIZE}}
export TORCH_TPU_TIER3_ROOT=${TORCH_TPU_TIER3_ROOT:-/cache/compile_cache/torch_tpu_${CACHE_NS}}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/cache/compile_cache/inductor_${CACHE_NS}}
mkdir -p "$TORCH_TPU_TIER3_ROOT" "$TORCHINDUCTOR_CACHE_DIR"

# MAX_RUNNING: CONCUR+4 on the TP path (measured), CONCUR+32 under DP8.
if [ "$DP_SIZE" -gt 1 ]; then
  export MAX_RUNNING=${MAX_RUNNING:-$((CONCUR + 32))}
else
  export MAX_RUNNING=${MAX_RUNNING:-$((CONCUR + 4))}
fi

echo "=== repro: CONCUR=$CONCUR DP_SIZE=$DP_SIZE ISL=$ISL OSL=$OSL MAX_RUNNING=$MAX_RUNNING"
echo "=== repo=$SGL_REPO commit=$(git -C "$SGL_REPO" rev-parse --short HEAD 2>/dev/null)"
echo "=== torch_tpu=$("$SGL_REPO/.venv/bin/python" -c 'import importlib.metadata as m;print(m.version("torch_tpu"))' 2>/dev/null)"
echo "=== cache_ns=$CACHE_NS"

MODEL="$MODEL_DIR" CONCUR="$CONCUR" DP_SIZE="$DP_SIZE" \
  CONTEXT_LEN=$((ISL + OSL + 20)) PORT="$PORT" PYTHON="$SGL_REPO/.venv/bin/python" \
  bash "$(dirname "$0")/server.sh" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "=== waiting for /health (cold start is ~25 min)"
until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; do
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died" >&2; exit 1; }
  sleep 20
done

# (3) bench client from its own venv -- there is no system python3
ISL="$ISL" OSL="$OSL" CONC="$CONCUR" PORT="$PORT" \
  HF_HOME=${HF_HOME:-/cache/hf} HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/cache/hf/hub} \
  VIRTUAL_ENV="$BENCH_VENV" PATH="$BENCH_VENV/bin:$PATH" \
  bash "$IX_BENCH"
