#!/bin/bash
# V1<->V2 per-request token-parity matrix for TPU EAGLE3 SpecV2.
#
# Launches a V1 server (no --enable-spec-v2-tpu) and a V2 server (--enable-spec-v2-tpu),
# and for each concurrency in PARALLELS captures the fixed greedy prompt set from BOTH,
# then strict-compares V1 vs V2 per request and reports the divergence SIGNATURE
# (how many requests are byte-identical, and the first-divergence positions).
#
# Acceptance:
#   * bs=1 (parallel=1): V1 and V2 must be byte-identical (strict). The eager bs=1
#     V1==V2 byte-identity is the standing losslessness proof (Round 0).
#   * bs>1: V1 and V2 build DIFFERENT compiled bf16 graphs, so greedy output can differ
#     by bf16 near-tie argmax flips. The STRUCTURED near-tie proof (the user-accepted
#     amendment) is implemented as a fail-closed comparator: parity_v1_v2.py
#     --allow-near-tie + the CPU sensitivity fixture test_parity_harness.py. It requires
#     a per-emitted-token top-2 logit trace from the server. That trace CANNOT be
#     captured from the live V2 bs>1 overlap path on torch_tpu: any top-2 computation
#     there either deadlocks the overlap pipeline (host sync) or poisons the lazy graph
#     (deferred), so the bs>1 margin evidence is architecturally unavailable on this
#     stack (documented in work_dirs/specv2/results/AC2_MATRIX.md). This script therefore
#     reports the bs>1 divergence signature for review rather than claiming a pass.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
DRAFT_MODEL="${DRAFT_MODEL:-AngelSlim/Qwen3-1.7B_eagle3}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-30010}"
PY="${PY:-.venv/bin/python3}"
MAXNEW="${MAXNEW:-48}"
PARALLELS="${PARALLELS:-2 3 5 7 8}"
# bs buckets to compile (must cover the concurrencies in PARALLELS); shrink to speed
# up cold runs, e.g. COMPILE_BS="1 2 3" with PARALLELS="2 3".
COMPILE_BS="${COMPILE_BS:-1 2 3 5 7 8}"
OUTDIR="$SCRIPT_DIR/results/v1v2_matrix"
mkdir -p "$OUTDIR"

export SGLANG_PLATFORM=tpu SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export PARITY_REQUEST_TIMEOUT="${PARITY_REQUEST_TIMEOUT:-1800}"  # cold compiles are slow
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/mnt/disks/pyc-v6e4-disk/.cache/torchinductor_specv2}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/disks/pyc-v6e4-disk/.cache/triton_specv2}"
unset TPU_VISIBLE_CHIPS TPU_CHIPS_PER_HOST_BOUNDS TPU_HOST_BOUNDS 2>/dev/null || true

cleanup() {
  ps -eo pid,args | grep -iE "launch_server|sglang::" | grep -v grep \
    | grep -v "bash -c" | grep -v shell-snapshots | awk '{print $1}' \
    | while read -r p; do kill -9 "$p" 2>/dev/null; done
}
fail() { echo "MATRIX FAIL: $*"; cleanup; exit 1; }
trap cleanup EXIT

launch() {  # variant_label extra_flags...
  local NAME=$1; shift; local LOG=/tmp/v1v2_matrix_$NAME.log
  cleanup; sleep 3; rm -f "$LOG"
  nohup "$PY" -u -m sglang.launch_server \
    --model-path "$MODEL" --context-length 4096 --device tpu --tp-size "$TP_SIZE" --dtype bfloat16 \
    --decode-log-interval 200 --host 127.0.0.1 --port "$PORT" --watchdog-timeout 3600 --skip-server-warmup \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path "$DRAFT_MODEL" \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.4 "$@" \
    --enable-torch-compile --tpu-compile-bs $COMPILE_BS --tpu-compile-tokens 1024 2048 --chunked-prefill-size 2048 \
    > "$LOG" 2>&1 &
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] && return 0
    grep -qiE "Scheduler hit an exception|SIGQUIT received|Initialization failed" "$LOG" && fail "$NAME crashed at startup"
    sleep 5
  done
  fail "$NAME never became ready"
}

capture_all() {  # variant_label
  local NAME=$1
  # parallel=1 pre-warm so the model graphs compile incrementally before any bs>1 request.
  "$PY" "$SCRIPT_DIR/parity_v1_v2.py" --url "http://127.0.0.1:$PORT" --out /tmp/_prewarm.json --parallel 1 --max-new "$MAXNEW" >/dev/null 2>&1 || true
  for n in $PARALLELS; do
    "$PY" "$SCRIPT_DIR/parity_v1_v2.py" --url "http://127.0.0.1:$PORT" --out /tmp/_warm.json --parallel "$n" --max-new "$MAXNEW" >/dev/null 2>&1
    "$PY" "$SCRIPT_DIR/parity_v1_v2.py" --url "http://127.0.0.1:$PORT" --out "$OUTDIR/${NAME}_p${n}.json" --parallel "$n" --max-new "$MAXNEW" \
      || fail "$NAME capture at parallel=$n failed"
  done
}

echo "== launching V1 (no --enable-spec-v2-tpu) =="
launch V1
capture_all v1
echo "== launching V2 (--enable-spec-v2-tpu) =="
launch V2 --enable-spec-v2-tpu
capture_all v2
cleanup

echo ""
echo "== V1 vs V2 per-request parity matrix =="
DIFFS=0
for n in $PARALLELS; do
  echo "--- parallel=$n ---"
  if "$PY" "$SCRIPT_DIR/parity_v1_v2.py" --compare "$OUTDIR/v1_p${n}.json" "$OUTDIR/v2_p${n}.json" >/dev/null 2>&1; then
    echo "parallel=$n: STRICT PASS (V1==V2 byte-identical)"
  else
    DIFFS=$((DIFFS + 1))
    "$PY" - "$OUTDIR/v1_p${n}.json" "$OUTDIR/v2_p${n}.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
ident = pos = 0; positions = []
for ra, rb in zip(a, b):
    ia, ib = ra.get("output_ids"), rb.get("output_ids")
    if ia == ib: ident += 1
    else:
        m = min(len(ia or []), len(ib or []))
        positions.append(next((k for k in range(m) if ia[k] != ib[k]), m))
print(f"parallel signature: {ident}/{len(a)} byte-identical, "
      f"divergence positions {sorted(positions)} (earliest {min(positions) if positions else '-'})")
PY
  fi
done

echo ""
if [ "$DIFFS" = "0" ]; then
  echo "MATRIX PASS: V1==V2 byte-identical for all parallel in {$PARALLELS}"
  exit 0
fi
echo "MATRIX: $DIFFS/$(echo $PARALLELS | wc -w) concurrencies diverge (bf16 near-tie signature expected)."
echo "  Structured near-tie PROOF requires the server top-2 logit trace, which is"
echo "  architecturally unavailable from the live V2 bs>1 overlap path on torch_tpu"
echo "  (see header + AC2_MATRIX.md). The fail-closed near-tie gate itself is the"
echo "  comparator parity_v1_v2.py --allow-near-tie + CPU fixture test_parity_harness.py."
echo "  AC-2 losslessness rests on eager bs=1 V1==V2 byte-identity (Round 0)."
exit 2
