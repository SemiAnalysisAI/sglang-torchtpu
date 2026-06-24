#!/bin/bash
# Standing gate for SpecV2 EAGLE3 + pallas_tpu_rpa_v3 on TPU. Exits NONZERO on any
# failure (fail-closed). Covers:
#   * overlap+EAGLE3+rpa_v3 correctness with two in-flight batches (parallel=2), a
#     bs/token bucket boundary crossing, long decode, and allocator page reuse. The
#     correctness signal is DETERMINISM: two identical bs=2 runs must be byte-identical
#     (same server + greedy => any divergence is real corruption, not a bf16 near-tie).
# The bs2-vs-bs1 cross-batch parity is reported as a diagnostic only (it may differ by
# documented bf16 batching near-ties); it does NOT gate.
#
# Usage:  ./bench_specv2_gate.sh        (launches its own server)
#   MODEL/DRAFT_MODEL/TP_SIZE/PORT overridable; see bench_eagle_eval.sh defaults.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
DRAFT_MODEL="${DRAFT_MODEL:-AngelSlim/Qwen3-1.7B_eagle3}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-30010}"
PY="${PY:-.venv/bin/python3}"
LOG=/tmp/specv2_gate.log
OUTDIR="$SCRIPT_DIR/results/specv2_gate"
mkdir -p "$OUTDIR"

export SGLANG_PLATFORM=tpu SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/mnt/disks/pyc-v6e4-disk/.cache/torchinductor_specv2}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/disks/pyc-v6e4-disk/.cache/triton_specv2}"
unset TPU_VISIBLE_CHIPS TPU_CHIPS_PER_HOST_BOUNDS TPU_HOST_BOUNDS 2>/dev/null || true

fail() { echo "GATE FAIL: $*"; cleanup; exit 1; }
cleanup() {
  ps -eo pid,args | grep -iE "launch_server|sglang::" | grep -v grep \
    | grep -v "bash -c" | grep -v shell-snapshots | awk '{print $1}' \
    | while read -r p; do kill -9 "$p" 2>/dev/null; done
}
trap cleanup EXIT

cleanup; sleep 2; rm -f "$LOG"
# bs buckets 1 and 2 (covered set); token buckets exercise a prefill bucket boundary.
nohup "$PY" -u -m sglang.launch_server \
  --model-path "$MODEL" --context-length 4096 --device tpu --tp-size "$TP_SIZE" --dtype bfloat16 \
  --decode-log-interval 5 --host 127.0.0.1 --port "$PORT" --watchdog-timeout 3600 --skip-server-warmup \
  --speculative-algorithm EAGLE3 --speculative-draft-model-path "$DRAFT_MODEL" \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-spec-v2-tpu --mem-fraction-static 0.4 \
  --enable-torch-compile --tpu-compile-bs 1 2 --tpu-compile-tokens 1024 2048 --chunked-prefill-size 2048 \
  > "$LOG" 2>&1 &

for i in $(seq 1 200); do
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && break
  grep -qiE "Scheduler hit an exception|SIGQUIT received|Initialization failed" "$LOG" 2>/dev/null && fail "server crashed during startup"
  sleep 5
done
[ "$(curl -s -o /dev/null -w "%{http_code}" -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] \
  || fail "server never became ready"

gen() {  # prompt max_new -> raw response
  curl -s -m 800 "http://127.0.0.1:$PORT/generate" -H "Content-Type: application/json" \
    -d "{\"text\":$1,\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":$2}}" 2>/dev/null
}

# ---- WARMUP: compile the EXACT workload's shapes (bs=1 + bs=2 over the same prompt
# set the steady-state pass re-runs), so steady-state runs at COVERED buckets. The
# warmup captures are also the bs=1 baseline. max_new=48 grows KV across decode. ----
echo "== warmup (compile the workload's bs=1 + bs=2 shapes) =="
"$PY" "$SCRIPT_DIR/parity_v1_v2.py" --url "http://127.0.0.1:$PORT" --out "$OUTDIR/bs1.json" --parallel 1 --max-new 48 \
  || fail "bs=1 warmup/baseline capture failed"
"$PY" "$SCRIPT_DIR/parity_v1_v2.py" --url "http://127.0.0.1:$PORT" --out "$OUTDIR/warm_bs2.json" --parallel 2 --max-new 48 \
  || fail "bs=2 warmup capture failed"
[ "$(curl -s -o /dev/null -w "%{http_code}" -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] \
  || fail "server died during warmup"

# ---- STEADY-STATE bs>1 overlap: RE-RUN the same bs=2 workload TWICE (two in-flight
# batches, long decode w/ allocator page reuse). The two identical runs must be
# byte-identical (determinism is the fail-closed overlap-correctness signal: same
# server + greedy, so any divergence is real KV/slice corruption or a draft->verify
# race, NOT a bf16 tie). ----
echo "== steady-state bs>1 overlap (re-run same workload, pass A) =="
"$PY" "$SCRIPT_DIR/parity_v1_v2.py" --url "http://127.0.0.1:$PORT" --out "$OUTDIR/bs2_a.json" --parallel 2 --max-new 48 \
  || fail "bs>1 capture A failed (server hung/crashed under overlap)"
echo "== steady-state bs>1 overlap (re-run same workload, pass B) =="
"$PY" "$SCRIPT_DIR/parity_v1_v2.py" --url "http://127.0.0.1:$PORT" --out "$OUTDIR/bs2_b.json" --parallel 2 --max-new 48 \
  || fail "bs>1 capture B failed (server hung/crashed under overlap)"
[ "$(curl -s -o /dev/null -w "%{http_code}" -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] \
  || fail "server died under sustained bs>1 overlap (corruption/deadlock)"

# ---- overlap correctness (FAIL-CLOSED): two identical bs=2 runs must match
# byte-for-byte. This is deterministic (same graph/server), so a nonzero comparator
# exit is real corruption -- it calls fail, never exits 0 on a "near-tie" excuse. ----
echo "== overlap determinism (bs=2 pass A vs pass B, must be byte-identical) =="
"$PY" "$SCRIPT_DIR/parity_v1_v2.py" --compare "$OUTDIR/bs2_a.json" "$OUTDIR/bs2_b.json" | tee "$OUTDIR/overlap_determinism.txt"
[ "${PIPESTATUS[0]}" = "0" ] || fail "overlap non-deterministic: two identical bs=2 runs diverged (KV/slice corruption or draft->verify race)"

# ---- bs>1 vs bs=1 cross-batch parity (diagnostic): may differ ONLY by bf16 batching
# near-ties (documented amendment). Reported, not gating; the gating correctness signal
# above is determinism, which admits no bf16 excuse. ----
echo "== cross-batch parity (bs2 vs bs1, diagnostic; bf16 near-ties allowed) =="
"$PY" "$SCRIPT_DIR/parity_v1_v2.py" --compare "$OUTDIR/bs1.json" "$OUTDIR/bs2_a.json" | tee "$OUTDIR/crossbatch_parity.txt" || true

echo ""
echo "overlap determinism: PASS (bs=2 byte-identical across runs)"
echo "GATE PASS"
exit 0
