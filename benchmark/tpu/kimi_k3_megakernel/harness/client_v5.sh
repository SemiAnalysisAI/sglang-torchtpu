#!/usr/bin/env bash
# client_v5.sh (client_v4 + PROFILE_SPECS xprof captures) <variant> <variant_dir> <port>: wait for the HTTP head, run the eval
# phases, GPQA Diamond for variants in GPQA_VARIANTS, the TPOT sweep if SWEEP is set,
# then write <variant_dir>/shutdown.
# Env (from the plan via run_variants.sh): CLIENT_CODE, MODEL, SWEEP, NHOSTS, JS_NAME, POD_NAMESPACE.
set +e
V=$1; VDIR=$2; PORT=$3
SCRIPTS=/mnt/workspace/dagent/e2e/scripts
source "$PLAN"
STARTUP_TIMEOUT_S=${STARTUP_TIMEOUT_S:-7200}
T0=$(date +%s)
echo "[Client] $V waiting for the server on port $PORT ($(date -u +%T))"
SRV_HOST=""
find_head() {
  # Main normalizes node ranks topologically, so the HTTP head need not be
  # JobSet host 0; the others answer /health from a dummy server.
  for i in $(seq 0 $((NHOSTS - 1))); do
    h="${JS_NAME}-ray-node-${i}-0.${JS_NAME}.${POD_NAMESPACE}.svc.cluster.local"
    [ -n "${E2E_TEST_HOST:-}" ] && h=$E2E_TEST_HOST  # local dry runs only
    if curl -sf -m 5 "http://${h}:${PORT}/v1/models" >/dev/null 2>&1; then SRV_HOST="$h"; return 0; fi
  done
  return 1
}
until find_head; do
  if [ -f "$VDIR/shutdown" ]; then echo "[Client] $V shutdown requested before the server came up"; exit 1; fi
  if [ $(( $(date +%s) - T0 )) -gt "$STARTUP_TIMEOUT_S" ]; then echo "[Client] $V server never came up" | tee -a "$VDIR/shutdown"; exit 1; fi
  sleep 10
done
UP_S=$(( $(date +%s) - T0 ))
echo "$UP_S" > "$VDIR/up"
echo "=== [Client] $V server UP on ${SRV_HOST} after ${UP_S}s ==="
BASE="http://${SRV_HOST}:${PORT}"
curl -s "$BASE/server_info" > "$VDIR/server_info.json"

export PYTHONPATH="${CLIENT_CODE}/python:${CLIENT_CODE}"
python3 "$SCRIPTS/k3_e2e_eval.py" "$VDIR" "$BASE" "$MODEL" "${E2E_PHASES1:-validate,greedy,gsm8k}" 2>&1

if [[ " ${GPQA_VARIANTS:-} " == *" $V "* ]]; then
  echo "=== [GPQA] $V GPQA Diamond, all questions ($(date -u +%T)) ==="
  python3 -c 'import sgl_eval' 2>/dev/null \
    || pip install -q "antlr4-python3-runtime==4.9.3" "sgl-eval==0.1.0" > "$VDIR/sgleval_pip.log" 2>&1 \
    || { tail -20 "$VDIR/sgleval_pip.log"; echo "[GPQA] sgl-eval install failed"; }
  # The shared HF token cannot read the gated GPQA repo: seed sgl-eval's cache with
  # the 198-row Diamond split built from the public CSV (k3acc/make_gpqa_cache.py).
  mkdir -p "$HOME/.cache/sgl_eval/gpqa"
  cp /mnt/workspace/dagent/k3acc/gpqa_diamond.jsonl "$HOME/.cache/sgl_eval/gpqa/diamond.jsonl"
  echo "[GPQA] seeded $(wc -l < "$HOME/.cache/sgl_eval/gpqa/diamond.jsonl") rows"
  python3 "$SCRIPTS/gpqa_eval.py" "$VDIR" "$BASE" "$MODEL" 2>&1
  echo "=== [GPQA] $V done ($(date -u +%T)) ==="
fi

if [ -n "${SWEEP:-}" ]; then
echo "=== [Bench] $V SWEEP=${SWEEP} ==="
(
  cd "$CLIENT_CODE" && MODEL="$MODEL" BASE_URL="$BASE" \
    DATASET_NAME=random-ids TOKENIZE_PROMPT=1 RANDOM_RANGE_RATIO=1.0 \
    PROMPTS_PER_CONCURRENCY=1 WARMUP_REQUESTS=1 DUMP_META=0 \
    OUTPUT_DIR="$VDIR/bench" SWEEP="${SWEEP}" \
    bash benchmark/tpu/bench_serving.sh 2>&1
)
echo "=== [Bench] rc=$? ==="
for f in "$VDIR"/bench/*.jsonl; do echo "[RESULT] $f"; cat "$f"; echo; done

fi

if [ -n "${PROFILE_SPECS:-}" ]; then
  # PROFILE_SPECS: space-separated <batch>:<prompt_len> captures. The requests start first;
  # the profile starts PROFILE_DELAY_S later (after prefill) and records PROFILE_STEPS decode
  # steps. Traces go to the attached SSD (PROFILE_ROOT), not the root-disk workspace.
  PROFILE_ROOT=${PROFILE_ROOT:-/mnt/kimik3-ssd/profiles}
  for spec in $PROFILE_SPECS; do
    B=${spec%%:*}; L=${spec#*:}
    PDIR="$PROFILE_ROOT/${JS_NAME:-run}/$V/bs${B}_isl${L}"; mkdir -p "$PDIR"
    echo "=== [Profile] $V bs=$B isl=$L steps=${PROFILE_STEPS:-20} -> $PDIR ($(date -u +%T)) ==="
    python3 - "$BASE" "$B" "$L" <<'PY' &
import json, random, sys, threading, urllib.request
base, b, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
def one(i):
    rnd = random.Random(i)
    body = {"input_ids": [rnd.randrange(1000, 100000) for _ in range(n)],
            "sampling_params": {"max_new_tokens": max(64, min(1500, 9300 - n)), "temperature": 0, "ignore_eos": True}}
    req = urllib.request.Request(base + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=1800))
    print("[Profile] req", i, {k: out.get("meta_info", {}).get(k) for k in ("prompt_tokens", "completion_tokens", "e2e_latency")}, flush=True)
ts = [threading.Thread(target=one, args=(i,)) for i in range(b)]
[t.start() for t in ts]; [t.join() for t in ts]
PY
    CLI=$!
    sleep "${PROFILE_DELAY_S:-8}"
    curl -s -X POST "$BASE/start_profile" -H "Content-Type: application/json" \
      -d "{\"output_dir\": \"$PDIR\", \"num_steps\": ${PROFILE_STEPS:-20}, \"activities\": [\"TPU\"]}" > "$PDIR/start_profile.json"
    wait $CLI
    sleep 30
    echo "=== [Profile] files in $PDIR ==="; find "$PDIR" -type f | head -40; du -sh "$PDIR"
  done
fi

python3 "$SCRIPTS/k3_e2e_eval.py" "$VDIR" "$BASE" "$MODEL" "${E2E_PHASES2:-image,logscan}" 2>&1
echo "=== [Client] $V done after $(( $(date +%s) - T0 ))s ==="
echo "client done" >> "$VDIR/shutdown"
