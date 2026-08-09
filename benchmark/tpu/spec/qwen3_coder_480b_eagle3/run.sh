#!/usr/bin/env bash
# Launch Qwen3-Coder EAGLE3, wait for TPU compilation, run the complete
# c1/c16/c32 sweep, and stop the server. Intended to run inside a TPU pod.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VARIANT=${VARIANT:-eagle3-real}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-3600}
HEALTH_POLL_SECONDS=${HEALTH_POLL_SECONDS:-10}
BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}

if [[ -n "${CDK_OUTPUT_DIR:-}" ]]; then
    RUN_ROOT=${RUN_ROOT:-$CDK_OUTPUT_DIR/$VARIANT/$RUN_ID}
else
    RUN_ROOT=${RUN_ROOT:-$SCRIPT_DIR/results/$VARIANT/$RUN_ID}
fi
SERVER_LOG=${SERVER_LOG:-$RUN_ROOT/server.log}
BENCHMARK_LOG=${BENCHMARK_LOG:-$RUN_ROOT/benchmark.log}
export OUTPUT_DIR=${OUTPUT_DIR:-$RUN_ROOT/benchmark}
export RUN_DIR=${RUN_DIR:-$RUN_ROOT/server-artifacts}

mkdir -p "$RUN_ROOT" "$OUTPUT_DIR" "$RUN_DIR"

server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

printf 'Starting %s server; log=%s\n' "$VARIANT" "$SERVER_LOG"
VARIANT=$VARIANT "$SCRIPT_DIR/server.sh" >"$SERVER_LOG" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$RUN_ROOT/server.pid"

deadline=$((SECONDS + HEALTH_TIMEOUT))
until curl -fsS --max-time 2 "$BASE_URL/health" >/dev/null 2>&1; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        printf 'Server exited before becoming ready. Last log lines:\n' >&2
        tail -n 80 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        printf 'Server did not become ready within %ss. Last log lines:\n' "$HEALTH_TIMEOUT" >&2
        tail -n 80 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    sleep "$HEALTH_POLL_SECONDS"
done

printf 'Server ready; starting concurrency sweep.\n'
VARIANT=$VARIANT "$SCRIPT_DIR/bench.sh" 2>&1 | tee "$BENCHMARK_LOG"
printf 'Sweep complete; artifacts=%s\n' "$RUN_ROOT"
