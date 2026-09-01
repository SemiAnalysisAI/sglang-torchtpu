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
        code=$(curl -sS -o /dev/null -m 10 -w "%{http_code}" "$url/health" 2>/dev/null || echo 000)
        [[ "$code" == "200" ]] && { echo "[serve_group] $label healthy after $((SECONDS - start))s"; return 0; }
        if [[ -n "$watch_pid" ]] && ! kill -0 "$watch_pid" 2>/dev/null; then
            echo "[serve_group] ERROR: $label process died before healthy" >&2; return 1
        fi
        (( SECONDS - start >= timeout )) && { echo "[serve_group] ERROR: $label not healthy in ${timeout}s (last=$code)" >&2; return 1; }
        sleep 10
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
    # the live engine and stop -- only an engine that died on its own retries.
    trap 'kill -TERM "$child" 2>/dev/null; wait "$child" 2>/dev/null; exit 143' TERM INT
    while :; do
        bash "$TPU_BENCH/launch_server.sh" &
        child=$!
        wait "$child"
        rc=$?
        tries=$(( tries + 1 ))
        if (( tries >= 3 )); then
            echo "[serve_group] engine exited rc=$rc (attempt $tries/3); giving up" >&2
            return "$rc"
        fi
        echo "[serve_group] engine exited rc=$rc (attempt $tries/3); relaunching in 20s" >&2
        sleep 20
    done
}

if [[ "$PD_DISAGG" != "1" ]]; then
    # ------------------------- single-host recipes -------------------------
    if [[ "$WORKER_ID" != "0" ]]; then
        echo "[serve_group] rank $WORKER_ID idle for 1-host recipe; exiting 0"
        exit 0
    fi
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

PREFILL_LEADER=${SLICE_HOSTS[0]}
DECODE_LEADER=${SLICE_HOSTS[2]}
if (( WORKER_ID < 2 )); then
    ROLE=prefill
    SUB_HOSTS="${SLICE_HOSTS[0]},${SLICE_HOSTS[1]}"
    LOCAL_RANK=$WORKER_ID
    export PORT=$PD_PREFILL_PORT
else
    ROLE=decode
    SUB_HOSTS="${SLICE_HOSTS[2]},${SLICE_HOSTS[3]}"
    LOCAL_RANK=$(( WORKER_ID - 2 ))
    export PORT=$PD_DECODE_PORT
fi
echo "[serve_group] pd role=$ROLE local_rank=$LOCAL_RANK sub_hosts=$SUB_HOSTS"
# The engine-leader host list for the test/sweep side on this pod.
printf '%s\n' "${SLICE_HOSTS[@]}" > /tmp/ext_bench_phys_hosts

# Re-scope the slice env to this engine's 2-host sub-slice -- the WHOLE
# libtpu env set, consistently. GKE injects TPU_SKIP_MDS_QUERY=true plus a
# full-slice TPU_PROCESS_ADDRESSES/TPU_ACCELERATOR_TYPE; leaving those at the
# 4-host values while remapping TPU_WORKER_ID to 0/1 makes two pods claim the
# same worker id WITHIN THE SAME libtpu process set -- collectives then
# compile with "replica id #N seen more than once" (iterations 6/8; whether a
# run survived depended on which env libtpu happened to consult first).
export TPU_WORKER_HOSTNAMES="$SUB_HOSTS"
export TPU_WORKER_ID="$LOCAL_RANK"
export NODE_RANK="$LOCAL_RANK"
export NNODES=2
export TPU_TOPOLOGY=2x2x2   # the engine's sub-cube, not the slice's 2x2x4
if [[ -n "${TPU_PROCESS_ADDRESSES:-}" ]]; then
    _tpu_port=${TPU_PROCESS_PORT:-${TPU_PROCESS_ADDRESSES##*:}}
    export TPU_PROCESS_ADDRESSES="${SUB_HOSTS%%,*}:${_tpu_port},${SUB_HOSTS##*,}:${_tpu_port}"
fi
[[ -n "${TPU_ACCELERATOR_TYPE:-}" ]] && export TPU_ACCELERATOR_TYPE=tpu7x-16
[[ -n "${TPU_TOPOLOGY_WRAP:-}" ]] && export TPU_TOPOLOGY_WRAP=false,false,false
export TP_SIZE EP_SIZE
export HOST=0.0.0.0         # engines must be reachable across hosts
export SGLANG_HOST_IP=${SGLANG_HOST_IP:-$(hostname -i | awk '{print $1}')}
export SGLANG_RAIDEN_KV_PORT=${SGLANG_RAIDEN_KV_PORT:-31000}
export SGLANG_RAIDEN_RESHARD=${SGLANG_RAIDEN_RESHARD:-0}   # symmetric P/D TP

if [[ "$ROLE" == "prefill" ]]; then
    # Prefix caching + KV offload live on the prefill engine, where shared
    # prefixes are matched; spec runs on BOTH engines (the P/D handoff
    # carries the spec bootstrap state).
    # Unscaled pool: prefill holds KV only until handoff, and the 8k
    # EXTEND-graph capture needs the HBM the scaled pool would eat.
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
    # POD 0 (whatever its physical role) fronts both engines with the mini-LB
    # on :30000 -- the port the JobSet health-gates and rank-0's test files
    # bench, both of which run on this pod. The engine leaders are addressed
    # by their cluster-reachable names (the --prefill hostname is injected
    # into requests as the KV bootstrap address the decode engine dials, so
    # it must never be 127.0.0.1).
    LB_PORT=${LB_PORT:-30000}
    _watch_pid=""
    [[ "${SLICE_HOSTS[$WORKER_ID]}" == "$PREFILL_LEADER" ]] && _watch_pid=$ENGINE_PID
    wait_url "http://$PREFILL_LEADER:$PD_PREFILL_PORT" "prefill engine" "$HEALTH_TIMEOUT" "$_watch_pid" || exit 1
    wait_url "http://$DECODE_LEADER:$PD_DECODE_PORT" "decode engine" "$HEALTH_TIMEOUT" || exit 1
    "$PYTHON" -m sglang_router.launch_router \
        --pd-disaggregation --mini-lb \
        --prefill "http://$PREFILL_LEADER:$PD_PREFILL_PORT" \
        --decode "http://$DECODE_LEADER:$PD_DECODE_PORT" \
        --host 0.0.0.0 --port "$LB_PORT" > >(sed 's/^/[mini-lb] /') 2>&1 &
    CHILD_PIDS+=("$!")
fi

wait "$ENGINE_PID"
