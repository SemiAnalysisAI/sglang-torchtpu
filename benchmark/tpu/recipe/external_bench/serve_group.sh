#!/usr/bin/env bash
# Launch-group entry point for the external TPU benchmark matrix. The external
# JobSet runs this on EVERY host of the slice (via a thin per-group wrapper
# that sets BENCH_MODEL/BENCH_RECIPE); what a host does depends on the recipe
# and its TPU_WORKER_ID:
#
#   1host_*                rank 0 serves a single-host tp=8 engine on :30000
#                          (FORCE_SINGLE_HOST); every other rank exits 0 and
#                          leaves its host idle. A 2x2x2 slice is the smallest
#                          the external cluster offers, so a 1-host recipe
#                          deliberately strands the second host.
#   pd_2p2d_pc_spec_kvoff  4-host slice split into two 2-host engines:
#                          ranks 0-1 form the PREFILL engine (tp=16), ranks
#                          2-3 the DECODE engine (tp=16), KV moving over
#                          raiden. Rank 0 additionally fronts both with the
#                          mini-LB on :30000 -- the port the JobSet
#                          health-gates and rank-0's test files bench.
#
# The sub-slice split works by overriding the GKE-injected slice env per
# engine: TPU_WORKER_HOSTNAMES is cut to the engine's 2 hosts (worker id
# remapped to 0/1) and TPU_TOPOLOGY is overridden to the 2x2x2 sub-cube, which
# is what _prepare_tpu_env derives the slicebuilder + TORCH_TPU_TOPOLOGY from.
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/config.sh"
TPU_BENCH="$TPU_BENCH_DIR"

WORKER_ID=${TPU_WORKER_ID:-0}
IFS=',' read -r -a SLICE_HOSTS <<< "${TPU_WORKER_HOSTNAMES:-$(hostname)}"

# Children must die with this script: the JobSet template kills the recipe
# process on teardown, and a surviving engine would hold the slice's chips
# against the next group.
CHILD_PIDS=()
cleanup() {
    for pid in "${CHILD_PIDS[@]:-}"; do
        [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 5
    for pid in "${CHILD_PIDS[@]:-}"; do
        [[ -n "$pid" ]] && kill -KILL "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT TERM INT

wait_url() { # url label timeout [watch_pid]
    local url="$1" label="$2" timeout="$3" watch_pid="${4:-}" start=$SECONDS code
    echo "[serve_group] waiting up to ${timeout}s for $label at $url"
    while :; do
        code=$(curl -sS -o /dev/null -m "$CURL_MAX_TIME" -w "%{http_code}" "$url/health" 2>/dev/null || echo 000)
        [[ "$code" == "200" ]] && { echo "[serve_group] $label healthy after $((SECONDS - start))s"; return 0; }
        if [[ -n "$watch_pid" ]] && ! kill -0 "$watch_pid" 2>/dev/null; then
            echo "[serve_group] ERROR: $label process died before healthy" >&2; return 1
        fi
        (( SECONDS - start >= timeout )) && { echo "[serve_group] ERROR: $label not healthy in ${timeout}s (last=$code)" >&2; return 1; }
        sleep "$HEALTH_POLL_INTERVAL"
    done
}

# Keep the torch_tpu tier-3 compile cache off the pod's ephemeral overlay:
# its $HOME default has filled the 40-80Gi overlay and had pods EVICTED,
# gang-restarting the whole JobSet.
export TORCH_TPU_TIER3_ROOT=${TORCH_TPU_TIER3_ROOT:-/dev/shm/torch_tpu_tier3}

# Env every engine shares, translated to launch_server.sh's interface.
export MODEL QUANTIZATION KV_CACHE_DTYPE ATTENTION_BACKEND PAGE_SIZE
export CONTEXT_LEN MEM_FRACTION_STATIC MAX_TOTAL_TOKENS
export CHUNKED_PREFILL_SIZE
export MAX_PREFILL_TOKENS=$CHUNKED_PREFILL_SIZE
export MAX_RUNNING TORCH_COMPILE_MAX_BS
export ENABLE_METRICS ENABLE_CACHE_REPORT
export ENABLE_TORCH_COMPILE=${ENABLE_TORCH_COMPILE:-1}
export DISABLE_RADIX_CACHE=$(( PREFIX_CACHE == 1 ? 0 : 1 ))
export PYTHON

BASE_EXTRA_ARGS="$MODEL_EXTRA_ARGS ${EXTRA_ARGS:-}"
[[ "$SPEC_DECODE" == "1" ]] && BASE_EXTRA_ARGS="$BASE_EXTRA_ARGS $SPEC_ARGS --sampling-backend pytorch"

log_config

# Engine supervisor: relaunch on early death. TPU multi-host bring-up can fail
# nondeterministically at graph capture ("error: replica id #N seen more than
# once" from the collective's replica map -- observed 2026-09-01 iteration 6,
# identical config compiled fine before and after). An engine's ranks die
# together on such a failure, so each rank retrying independently re-forms the
# rendezvous. Bounded: a deterministic failure still surfaces after 3 tries.
engine_supervisor() {
    local tries=0 rc child=""
    # Teardown must never look like a crash: on TERM/INT forward the signal to
    # the live engine's WHOLE process group and stop -- only an engine that
    # died on its own retries.
    trap 'kill -TERM -- "-$child" 2>/dev/null; wait "$child" 2>/dev/null; exit 143' TERM INT
    while :; do
        # setsid: the attempt is its own process group, so a failed attempt can
        # be reaped WHOLE before relaunching. sglang's schedulers are
        # grandchildren -- killing only the wrapper orphaned them holding their
        # ports, and every relaunch died on EADDRINUSE/TCPStore ping failures
        # (iteration 16).
        setsid bash "$TPU_BENCH/launch_server.sh" &
        child=$!
        wait "$child"
        rc=$?
        kill -KILL -- "-$child" 2>/dev/null || true
        tries=$(( tries + 1 ))
        if (( tries >= ENGINE_LAUNCH_TRIES )); then
            echo "[serve_group] engine exited rc=$rc (attempt $tries/$ENGINE_LAUNCH_TRIES); giving up" >&2
            return "$rc"
        fi
        echo "[serve_group] engine exited rc=$rc (attempt $tries/$ENGINE_LAUNCH_TRIES); relaunching in ${ENGINE_RELAUNCH_DELAY}s" >&2
        sleep "$ENGINE_RELAUNCH_DELAY"
    done
}

# Strip the slice-wide multihost env before launching any SINGLE-host engine:
# libtpu on GKE is env-driven (TPU_SKIP_MDS_QUERY=true), and the injected
# TPU_WORKER_HOSTNAMES / TPU_PROCESS_ADDRESSES / TPU_TOPOLOGY describe the
# WHOLE slice -- _prepare_tpu_env would build a slice-wide slicebuilder that
# never forms (idle peers), and FORCE_SINGLE_HOST only stops launch_server's
# own nnodes logic, not that env detection. Local hardware discovery then
# resolves the single-host topology.
strip_slice_env() {
    unset TPU_WORKER_ID TPU_WORKER_HOSTNAMES TPU_PROCESS_ADDRESSES \
          TPU_TOPOLOGY TPU_TOPOLOGY_WRAP TPU_TOPOLOGY_ALT TPU_HOST_BOUNDS \
          TPU_ACCELERATOR_TYPE 2>/dev/null || true
}

if [[ "$PD_DISAGG" != "1" ]]; then
    # ------------------------- single-host recipes -------------------------
    if [[ "$WORKER_ID" != "0" ]]; then
        echo "[serve_group] rank $WORKER_ID idle for 1-host recipe; exiting 0"
        exit 0
    fi
    strip_slice_env
    export FORCE_SINGLE_HOST=1
    export TP_SIZE EP_SIZE
    export PORT HOST=${HOST:-0.0.0.0}
    export KV_OFFLOAD HICACHE_RATIO HICACHE_IO_BACKEND HICACHE_MEM_LAYOUT
    export EXTRA_ARGS="$BASE_EXTRA_ARGS"
    exec "$TPU_BENCH/launch_server.sh"
fi

# ----------------------------- P/D recipe ------------------------------
if (( ${#SLICE_HOSTS[@]} < 4 )); then
    echo "[serve_group] ERROR: pd recipe needs a 4-host slice, got" \
         "TPU_WORKER_HOSTNAMES='${TPU_WORKER_HOSTNAMES:-}'" >&2
    exit 1
fi
# GKE injects MEGASCALE_* when it treats TPU jobs as one multislice workload;
# these engines are independent sub-slices (KV moves at application level).
for v in $(compgen -e | grep '^MEGASCALE' || true); do unset "$v"; done

# What the GKE TPU webhook injected, before any override -- libtpu on GKE is
# env-driven (TPU_SKIP_MDS_QUERY=true), so this is the topology ground truth
# and the debugging record for any grouping issue.
echo "[serve_group] injected TPU env:"
env | grep -E '^TPU_' | sed 's/^/[serve_group]   /' || true

# FOUR single-host tp=8 engines: pods 0-1 prefill, pods 2-3 decode, paired at
# request time by the mini-LB. This is the raiden P/D topology every existing
# gate validates (single-host engines; task-10 multihost PD likewise). The
# 2-host-tp=16-per-engine variant on a subdivided slice brings up cleanly with
# a consistently re-scoped libtpu env; its 2026-09-01 "KV pulls never
# complete" failures were the same IPv6 ZMQ rendezvous bug this topology hit
# (PR #606), so it is untested since the fix rather than known-broken --
# revisit if tp=16-per-engine P/D is wanted.
if (( WORKER_ID < 2 )); then
    ROLE=prefill
    export PORT=$PD_PREFILL_PORT
else
    ROLE=decode
    export PORT=$PD_DECODE_PORT
fi
echo "[serve_group] pd role=$ROLE (single-host engine on ${SLICE_HOSTS[$WORKER_ID]})"
# The engine host list for the test/sweep side on this pod.
printf '%s\n' "${SLICE_HOSTS[@]}" > /tmp/ext_bench_phys_hosts

strip_slice_env
export FORCE_SINGLE_HOST=1
export TP_SIZE EP_SIZE
export HOST=0.0.0.0         # engines must be reachable across hosts
export SGLANG_HOST_IP=${SGLANG_HOST_IP:-$(hostname -i | awk '{print $1}')}
export SGLANG_RAIDEN_KV_PORT=${SGLANG_RAIDEN_KV_PORT:-$RAIDEN_KV_BASE_PORT}
export SGLANG_RAIDEN_RESHARD=${SGLANG_RAIDEN_RESHARD:-0}   # symmetric P/D TP

if [[ "$ROLE" == "prefill" ]]; then
    # Prefix caching + KV offload live on the prefill engines, where shared
    # prefixes are matched; spec runs on BOTH roles (the P/D handoff carries
    # the spec bootstrap state).
    export MAX_TOTAL_TOKENS=$MAX_TOTAL_TOKENS_PREFILL
    export KV_OFFLOAD HICACHE_RATIO HICACHE_IO_BACKEND HICACHE_MEM_LAYOUT
    export EXTRA_ARGS="$BASE_EXTRA_ARGS \
--disaggregation-mode prefill \
--disaggregation-transfer-backend raiden \
--disaggregation-bootstrap-port $PD_BOOTSTRAP_PORT"
else
    # Decode: no hicache (spec draft KV is not offloaded; matches the
    # validated P/D+spec gates); its disaggregation radix cache stays off.
    export KV_OFFLOAD=0
    export EXTRA_ARGS="$BASE_EXTRA_ARGS \
--disaggregation-mode decode \
--disaggregation-transfer-backend raiden"
fi

engine_supervisor &
ENGINE_PID=$!
CHILD_PIDS+=("$ENGINE_PID")

if [[ "$WORKER_ID" == "0" ]]; then
    # POD 0 fronts all four engines with the mini-LB on :30000 -- the port the
    # JobSet health-gates and rank-0's test files bench, both on this pod. The
    # --prefill hostnames are injected into requests as the KV bootstrap
    # addresses the decode engines dial, so they must be the cluster-reachable
    # names, never 127.0.0.1.
    LB_PORT=${LB_PORT:-$PORT}   # the health-gated/benched port
    wait_url "http://${SLICE_HOSTS[0]}:$PD_PREFILL_PORT" "prefill engine 0" "$HEALTH_TIMEOUT" "$ENGINE_PID" || exit 1
    wait_url "http://${SLICE_HOSTS[1]}:$PD_PREFILL_PORT" "prefill engine 1" "$HEALTH_TIMEOUT" || exit 1
    wait_url "http://${SLICE_HOSTS[2]}:$PD_DECODE_PORT" "decode engine 0" "$HEALTH_TIMEOUT" || exit 1
    wait_url "http://${SLICE_HOSTS[3]}:$PD_DECODE_PORT" "decode engine 1" "$HEALTH_TIMEOUT" || exit 1
    "$PYTHON" -m sglang_router.launch_router \
        --pd-disaggregation --mini-lb \
        --prefill "http://${SLICE_HOSTS[0]}:$PD_PREFILL_PORT" "$PD_BOOTSTRAP_PORT" \
        --prefill "http://${SLICE_HOSTS[1]}:$PD_PREFILL_PORT" "$PD_BOOTSTRAP_PORT" \
        --decode "http://${SLICE_HOSTS[2]}:$PD_DECODE_PORT" \
        --decode "http://${SLICE_HOSTS[3]}:$PD_DECODE_PORT" \
        --host 0.0.0.0 --port "$LB_PORT" > >(sed 's/^/[mini-lb] /') 2>&1 &
    CHILD_PIDS+=("$!")
fi

wait "$ENGINE_PID"
