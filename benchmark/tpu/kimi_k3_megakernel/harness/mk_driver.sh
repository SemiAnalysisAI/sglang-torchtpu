#!/usr/bin/env bash
# Per-host driver for the Inferact Kimi-K3 megakernel (standalone, target-only, greedy) on one
# 2x2x4 slice: for each config in $PLAN, start distributed_entry.py on all 4 hosts, drive it from
# host 0 with mk_client.py, record the server's device-timed 'target_batch:' lines, tear down.
# Env: PLAN, OUT, NHOSTS, JOBSET_JOB_INDEX, JS_NAME, POD_NAMESPACE.
set -u
IDX=${JOBSET_JOB_INDEX}
SCRIPTS=/mnt/workspace/dagent/e2e/scripts
# shellcheck disable=SC1090
source "$PLAN"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/driver.host${IDX}.log") 2>&1
echo "[mk] host ${IDX} plan=${PLAN} configs='${CONFIGS}' $(date -u)"
cp "$PLAN" "$OUT/plan.sh" 2>/dev/null || true
env | grep -E "^(TPU_|JOBSET|MEGASCALE|HOSTNAME)" | sort

barrier() {
  touch "$OUT/barrier.$1.host${IDX}"
  local t0 n; t0=$(date +%s)
  while :; do
    n=$(ls "$OUT" 2>/dev/null | grep -c "^barrier\.$1\.host")
    [ "$n" -ge "$NHOSTS" ] && break
    if [ $(( $(date +%s) - t0 )) -gt "${2:-3600}" ]; then echo "[mk] barrier $1 timed out at $n/$NHOSTS"; break; fi
    sleep 5
  done
  echo "[mk] barrier $1 passed ($(date -u +%T))"
}

kill_mk() {
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}; [ "$pid" = "$$" ] && continue
    c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)
    case "$c" in *distributed_entry.py*) kill -9 "$pid" 2>/dev/null && echo "[mk] killed $pid";; esac
  done
}

pip install -q "jax[tpu]==${MK_JAX:-0.11.1}" "tiktoken>=0.14.0" "tokenizers>=0.21" "safetensors>=0.6" "huggingface-hub>=0.34" "numpy>=2" > "$OUT/pip.host${IDX}.log" 2>&1 \
  || { echo "[mk] pip install failed"; tail -20 "$OUT/pip.host${IDX}.log"; }
python3 - <<'PY'
import importlib.metadata as m
for d in ("jax", "jaxlib", "libtpu", "numpy", "tiktoken", "tokenizers", "safetensors"):
    try: print("[mk] ", d, m.version(d))
    except Exception: print("[mk] ", d, "-")
PY

HOSTS=""
for i in $(seq 0 $((NHOSTS - 1))); do HOSTS="${HOSTS:+$HOSTS,}${JS_NAME}-ray-node-${i}-0.${JS_NAME}.${POD_NAMESPACE}.svc.cluster.local"; done
HOST0=${HOSTS%%,*}
PORT=8000

for CFG in $CONFIGS; do
  IFS=: read -r NAME B CTX ISL OSL ROUNDS <<< "$CFG"
  CDIR="$OUT/$NAME"; mkdir -p "$CDIR"
  echo "[mk] === config $NAME batch=$B context=$CTX isl=$ISL osl=$OSL rounds=$ROUNDS $(date -u +%T)"
  kill_mk
  rm -rf /tmp/jaxcache_$NAME
  barrier "pre_$NAME"
  (
    cd "$MK_SRC" || exit 97
    export SLURM_PROCID=$IDX TP32_COORDINATOR="${HOST0}:12377" TPU_WORKER_HOSTNAMES="$HOSTS"
    export TPU_PROCESS_BOUNDS=1,1,4 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_CHIPS=0,1,2,3
    export JAX_PLATFORMS=tpu HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$MK_SRC"
    export JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache_$NAME XLA_FLAGS="--xla_allow_excess_precision=false"
    exec "$(command -v setsid)" python3 -B distributed_entry.py demo_kimi_dspark.py \
      --serve 0.0.0.0:$PORT --target-only --target-batch-size "$B" --steps-per-call 64 \
      --context "$CTX" --max-tokens "$OSL" --chat none --progress log \
      --checkpoint /mnt/kimik3-ssd/kimik3 --draft /mnt/kimik3-ssd/kimik3
  ) > "$CDIR/host${IDX}.log" 2>&1 &
  SRV=$!
  T0=$(date +%s)
  if [ "$IDX" = "0" ]; then
    (
      until curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        kill -0 $SRV 2>/dev/null || { echo "[mk] server exited during startup"; echo "server died" >> "$CDIR/done"; exit 1; }
        [ $(( $(date +%s) - T0 )) -gt "${STARTUP_TIMEOUT_S:-5400}" ] && { echo "[mk] startup timeout"; echo "timeout" >> "$CDIR/done"; exit 1; }
        sleep 10
      done
      echo "[mk] $NAME server up after $(( $(date +%s) - T0 ))s" | tee "$CDIR/up"
      python3 "$SCRIPTS/mk_client.py" "$CDIR" "http://127.0.0.1:$PORT" "$B" "$ISL" "$OSL" "$ROUNDS" validate
      sleep 3
      grep -h "target_batch:\|time to first\|median device call\|prompt tokens (prefill" "$CDIR/host0.log" | tail -20
      echo "client done" >> "$CDIR/done"
    ) > "$CDIR/client.log" 2>&1 &
  fi
  while :; do
    [ -f "$CDIR/done" ] && break
    if ! kill -0 $SRV 2>/dev/null; then echo "[mk] $NAME server on host $IDX exited"; echo "host $IDX server exited" >> "$CDIR/done"; break; fi
    [ $(( $(date +%s) - T0 )) -gt "${CONFIG_TIMEOUT_S:-7200}" ] && { echo "config timeout" >> "$CDIR/done"; break; }
    sleep 15
  done
  kill -TERM -- -"$SRV" 2>/dev/null; sleep 10; kill -KILL -- -"$SRV" 2>/dev/null; kill_mk
  barrier "post_$NAME"
done
echo "[mk] all configs done $(date -u)"
