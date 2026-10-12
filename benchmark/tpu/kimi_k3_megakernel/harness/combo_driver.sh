#!/usr/bin/env bash
# One TPU run, several phases on the same 2x2x4 slice, in order of $PHASES:
#   sglang: run_variants_v5.sh with $SG_PLAN (SGLang TPOT + mid-decode xprof, image/main stack)
#   spike:  mk_spike_driver.sh with $SPIKE_PLAN (megakernel inside torch_tpu workers, synthetic weights)
#   mk:     mk_driver.sh with $MK_PLAN (Inferact megakernel standalone; installs jax 0.11.1, so last)
# Each phase gets its own OUT subdirectory, a timeout, a process cleanup and a cross-host barrier.
# Env: PLAN, OUT, NHOSTS, JOBSET_JOB_INDEX, JS_NAME, POD_NAMESPACE.
set -u
IDX=${JOBSET_JOB_INDEX}
SCRIPTS=${SCRIPTS:-/mnt/workspace/dagent/e2e/scripts}
# shellcheck disable=SC1090
source "$PLAN"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/combo.host${IDX}.log") 2>&1
echo "[combo] host ${IDX} plan=${PLAN} phases='${PHASES}' $(date -u)"
cp "$PLAN" "$OUT/plan.sh" 2>/dev/null || true

barrier() {
  touch "$OUT/barrier.$1.host${IDX}"
  local t0 n; t0=$(date +%s)
  while :; do
    n=$(ls "$OUT" 2>/dev/null | grep -c "^barrier\.$1\.host")
    [ "$n" -ge "$NHOSTS" ] && break
    if [ $(( $(date +%s) - t0 )) -gt "${2:-3600}" ]; then echo "[combo] barrier $1 timed out at $n/$NHOSTS"; break; fi
    sleep 5
  done
  echo "[combo] barrier $1 passed ($(date -u +%T))"
}

ancestors() {
  local pid=$$
  while [ -n "$pid" ] && [ "$pid" != "0" ]; do
    echo "$pid"
    pid=$(awk '/^PPid:/{print $2}' "/proc/$pid/status" 2>/dev/null)
  done
}

cleanup() {
  # Leftover servers/workers from a phase that timed out would hold the TPU. Never touch
  # this driver or its ancestors.
  local keep; keep=" $(ancestors | tr '\n' ' ') "
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    case "$keep" in *" $pid "*) continue;; esac
    c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)
    case "$c" in
      *sglang.launch_server*|sglang::*|*launch_server.sh*|*torchrun*|*mk_bridge_spike.py*|*distributed_entry.py*|*run_variants_v5.sh*|*mk_spike_driver.sh*|*mk_driver.sh*)
        kill -9 "$pid" 2>/dev/null && echo "[combo] killed $pid: ${c:0:100}";;
    esac
  done
  rm -rf /dev/shm/torch_tpu_cache /tmp/tt3_* /tmp/sglcache_* 2>/dev/null
}

for P in $PHASES; do
  echo "[combo] === phase $P $(date -u +%T)"
  cleanup
  barrier "pre_$P" 1800
  case "$P" in
    sglang) PLAN=$SG_PLAN OUT=$OUT/sglang timeout "${SG_TIMEOUT_S:-6600}" bash "$SCRIPTS/run_variants_v5.sh" ;;
    spike)  PLAN=$SPIKE_PLAN OUT=$OUT/spike timeout "${SPIKE_TIMEOUT_S:-4800}" bash "$SCRIPTS/mk_spike_driver.sh" ;;
    mk)     PLAN=$MK_PLAN OUT=$OUT/mk timeout "${MK_TIMEOUT_S:-6000}" bash "$SCRIPTS/mk_driver.sh" ;;
    *) echo "[combo] unknown phase $P" ;;
  esac
  echo "[combo] phase $P rc=$? $(date -u +%T)"
  cleanup
  barrier "post_$P" 1800
done
echo "[combo] all phases done $(date -u)"
