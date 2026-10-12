#!/usr/bin/env bash
# Per-host driver: serve each variant of $PLAN in turn on every host of the slice,
# run the eval client from host 0, then tear down and barrier before the next one.
# Env: PLAN, OUT, NHOSTS, JOBSET_JOB_INDEX, JS_NAME, POD_NAMESPACE.
set -u
IDX=${JOBSET_JOB_INDEX}
SCRIPTS=/mnt/workspace/dagent/e2e/scripts
# shellcheck disable=SC1090
source "$PLAN"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/driver.host${IDX}.log") 2>&1
echo "[driver] host ${IDX} plan=${PLAN} variants='${VARIANTS}' $(date -u)"
cp "$PLAN" "$OUT/plan.sh" 2>/dev/null || true

VARIANT_TIMEOUT_S=${VARIANT_TIMEOUT_S:-9900}
STALL_DUMP_S=${STALL_DUMP_S:-5400}

kill_sglang() {
  # The image has no pkill; scan /proc for server processes left behind.
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    [ "$pid" = "$$" ] && continue
    c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)
    case "$c" in
      *sglang.launch_server*|sglang::*|*benchmark/tpu/launch_server.sh*) kill -9 "$pid" 2>/dev/null && echo "[driver] killed leftover $pid: ${c:0:120}";;
    esac
  done
}

alive() {
  # kill -0 also succeeds on an unreaped zombie; read the state field instead.
  local st
  st=$(awk '{print $3}' "/proc/$1/stat" 2>/dev/null)
  [ -n "$st" ] && [ "$st" != "Z" ]
}

stall_dump() {
  for p in /proc/[0-9]*; do
    c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)
    case "$c" in *sglang.launch_server*|sglang::*) echo "=== [StallDump] ${p#/proc/} ${c:0:160}"; py-spy dump --pid "${p#/proc/}" 2>&1 | head -80;; esac
  done
}

barrier() {
  # barrier <name> [timeout_s]: every host touches a marker on NFS, then waits for all.
  touch "$OUT/barrier.$1.host${IDX}"
  local t0 n
  t0=$(date +%s)
  while :; do
    n=$(ls "$OUT" 2>/dev/null | grep -c "^barrier\.$1\.host")
    [ "$n" -ge "$NHOSTS" ] && break
    if [ $(( $(date +%s) - t0 )) -gt "${2:-1800}" ]; then echo "[driver] barrier $1 timed out at $n/$NHOSTS"; break; fi
    sleep 5
  done
  echo "[driver] barrier $1 passed ($(date -u +%T))"
}

install_main_stack() {
  # main's pinned TPU stack (python/pyproject_tpu.toml)
  pip install --no-deps -q /mnt/workspace/dagent_wheels/torch_tpu-0.1.1+release.2026.9.22.20260927223741-cp312-cp312-manylinux_2_31_x86_64.whl "tpu-sync-torch==0.0.1.dev20261002213151" "libtpu==0.0.47" "jax==0.10.2" "jaxlib==0.10.2" \
    || pip install --no-deps -q /mnt/workspace/dagent_wheels/torch_tpu-0.1.1+release.2026.9.22.20260927223741-cp312-cp312-manylinux_2_31_x86_64.whl "tpu-sync-torch==0.0.1.dev20260930190039" "libtpu==0.0.47" "jax==0.10.2" "jaxlib==0.10.2"
}

stack_versions() {
  python3 - <<'PYEOF' 2>&1
import importlib.metadata as m
for d in ("torch", "torch_tpu", "jax", "jaxlib", "libtpu", "tpu-sync-torch", "tpu-raiden-torch"):
    try:
        print(d, m.version(d))
    except Exception as e:
        print(d, "-")
PYEOF
}

CUR_STACK=image
k=0
for V in $VARIANTS; do
  k=$((k + 1))
  VDIR="$OUT/$V"
  mkdir -p "$VDIR"
  PORT_V=$((47903 + 10 * k))
  DIST_V=$((10011 + 10 * k))
  if [ "${STACK[$V]}" = "main" ] && [ "$CUR_STACK" != "main" ]; then
    echo "[driver] installing main's pinned stack"
    install_main_stack
    CUR_STACK=main
  fi
  stack_versions > "$VDIR/stack.host${IDX}.txt"
  echo "[driver] === variant $V (code ${CODE[$V]}, stack $CUR_STACK, port $PORT_V) $(date -u +%T)"
  cat "$VDIR/stack.host${IDX}.txt"
  kill_sglang
  # Fresh compile caches for every server: torch_tpu tier-2 (/dev/shm) and
  # tier-3, sglang/inductor caches. Per-variant names below as a second guard.
  rm -rf /dev/shm/torch_tpu_cache "$HOME/.cache/torch_tpu_compile_cache" "$HOME/.cache/sglang" /tmp/torchinductor_* /tmp/sglcache_* /tmp/tt3_* 2>/dev/null
  barrier "pre_${V}" 3600
  (
    cd "${CODE[$V]}" || exit 97
    export PYTHONPATH="${CODE[$V]}/python:${CODE[$V]}"
    for kv in ${ENVS[$V]}; do export "$kv"; echo "VARIANT_ENV $kv"; done
    export PORT=$PORT_V DIST_PORT=$DIST_V
    export SGLANG_CACHE_DIR=/tmp/sglcache_$V TORCHINDUCTOR_CACHE_DIR=/tmp/sglcache_$V/inductor
    export TORCH_TPU_TIER2_NAME=torchtpu_k3_$V TORCH_TPU_TIER3_ROOT=/tmp/tt3_$V
    export EXTRA_ARGS="${BASE_EXTRA_ARGS} ${ARGS[$V]:-}"
    echo "EXTRA_ARGS=$EXTRA_ARGS"
    exec "$(command -v setsid)" bash benchmark/tpu/launch_server.sh
  ) > "$VDIR/host${IDX}.log" 2>&1 &
  SRV=$!
  T0=$(date +%s)
  CLI=""
  if [ "$IDX" = "0" ]; then
    "$(command -v setsid)" bash "$SCRIPTS/client_v5.sh" "$V" "$VDIR" "$PORT_V" > "$VDIR/client.log" 2>&1 &
    CLI=$!
  fi
  DUMPED=0
  while alive "$SRV" && [ ! -f "$VDIR/shutdown" ]; do
    sleep 15
    el=$(( $(date +%s) - T0 ))
    if [ "$el" -gt "$VARIANT_TIMEOUT_S" ]; then echo "timeout host${IDX} after ${el}s" >> "$VDIR/shutdown"; fi
    if [ "$DUMPED" = "0" ] && [ "$el" -gt "$STALL_DUMP_S" ] && [ ! -f "$VDIR/up" ]; then DUMPED=1; stall_dump > "$VDIR/stall.host${IDX}.txt" 2>&1; fi
  done
  if ! alive "$SRV"; then
    wait "$SRV"; rc=$?
    echo "[driver] $V: server on host ${IDX} exited on its own rc=$rc after $(( $(date +%s) - T0 ))s"
    echo "host${IDX} server exited rc=$rc" >> "$VDIR/shutdown"
  fi
  echo "[driver] $V: stopping server on host ${IDX} ($(cat "$VDIR/shutdown" 2>/dev/null | head -3 | tr '\n' ';'))"
  kill -TERM -- -"$SRV" 2>/dev/null
  for _ in $(seq 1 12); do alive "$SRV" || break; sleep 5; done
  kill -KILL -- -"$SRV" 2>/dev/null
  kill_sglang
  wait "$SRV" 2>/dev/null
  if [ -n "$CLI" ]; then
    # Normally finished (it writes the shutdown marker); stop it if the server died first.
    for _ in $(seq 1 6); do alive "$CLI" || break; sleep 5; done
    kill -KILL -- -"$CLI" 2>/dev/null
    wait "$CLI" 2>/dev/null
  fi
  touch "$VDIR/down.host${IDX}"
  sleep 30
  barrier "post_${V}" 1200
done
echo "[driver] host ${IDX} all variants done $(date -u)"
