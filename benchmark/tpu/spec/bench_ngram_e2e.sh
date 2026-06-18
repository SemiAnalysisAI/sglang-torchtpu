#!/bin/bash
# End-to-end NGram speculative decoding benchmark on TPU.
#
# Launches a baseline server (no spec decode), benchmarks it, then launches an
# NGram server and benchmarks it. Reports throughput and accept length for both.
#
# NGram on TorchTPU runs in a single fixed mode (the linear-greedy MVP):
#   - linear draft chain: --speculative-ngram-max-bfs-breadth 1
#   - greedy verification only (the benchmark uses temperature=0); a non-greedy
#     request is rejected request-scoped with HTTP 400
#   - pallas_tpu_rpa_v3 attention backend, page_size forced to 1
#   - eager mode (do NOT pass --enable-torch-compile)
# The n-gram proposer runs on host CPU; only the target verify forward runs on TPU.
# Unlike EAGLE there is no draft model.
#
# Usage:
#   ./bench_ngram_e2e.sh
#
# Environment variables:
#   MODEL             Target model (default: Qwen/Qwen3-0.6B)
#   TP_SIZE           Tensor parallel size (default: 1)
#   PORT              Server port (default: 30000)
#   NUM_QUESTIONS     MT-Bench questions (default: 5)
#   PARALLEL          Concurrent requests (default: 1)
#   DRAFT_TOKENS      Speculative num draft tokens (default: 4)
#   NGRAM_BREADTH     N-gram max BFS breadth, must be 1 on TPU (default: 1)
#   SKIP_BASELINE     Set to 1 to skip the baseline run
#   SKIP_NGRAM        Set to 1 to skip the NGram run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
TP_SIZE="${TP_SIZE:-1}"
PORT="${PORT:-30000}"
NUM_QUESTIONS="${NUM_QUESTIONS:-5}"
PARALLEL="${PARALLEL:-1}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-1}"
DRAFT_TOKENS="${DRAFT_TOKENS:-4}"
NGRAM_BREADTH="${NGRAM_BREADTH:-1}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
SKIP_NGRAM="${SKIP_NGRAM:-0}"

export SGLANG_PLATFORM=tpu
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
unset TPU_VISIBLE_CHIPS TPU_CHIPS_PER_HOST_BOUNDS TPU_HOST_BOUNDS

export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_scoped_vmem_limit_kib=32768 \
  --xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_tpu_enable_latency_hiding_scheduler=true}"

RESULT_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cleanup() {
    echo "Cleaning up..."
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 2
}
trap cleanup EXIT

wait_for_server() {
    local max_wait="${1:-600}"
    echo "Waiting for server (max ${max_wait}s)..."
    for i in $(seq 1 $((max_wait / 10))); do
        code=$(curl -sS -o /dev/null -m 5 -w "%{http_code}" "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            echo "Server ready after $((i * 10))s"
            return 0
        fi
        sleep 10
    done
    echo "ERROR: Server not ready after ${max_wait}s"
    return 1
}

run_benchmark() {
    local label="$1"
    local result_file="$RESULT_DIR/${TIMESTAMP}_${label}.jsonl"

    # Warmup rounds: run the benchmark to trigger XLA compilation for all shapes,
    # then discard the results. Only the final run is kept.
    for round in $(seq 1 "$WARMUP_ROUNDS"); do
        echo ""
        echo "Warmup round $round/$WARMUP_ROUNDS ($label, ${NUM_QUESTIONS} questions)..."
        python3 "$REPO_ROOT/benchmark/mtbench/bench_sglang_eagle.py" \
            --num-questions "$NUM_QUESTIONS" \
            --parallel "$PARALLEL" \
            --host 127.0.0.1 \
            --port "$PORT" \
            --backend srt \
            --result-file "/dev/null"
    done

    echo ""
    echo "Benchmark run ($label, ${NUM_QUESTIONS} questions, parallel=${PARALLEL})..."
    python3 "$REPO_ROOT/benchmark/mtbench/bench_sglang_eagle.py" \
        --num-questions "$NUM_QUESTIONS" \
        --parallel "$PARALLEL" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --backend srt \
        --result-file "$result_file"

    echo "Results: $result_file"
}

# Common server args (eager: no --enable-torch-compile).
BASE_ARGS=(
    --model-path "$MODEL"
    --context-length 4096
    --device tpu
    --tp-size "$TP_SIZE"
    --dtype bfloat16
    --attention-backend pallas_tpu_rpa_v3
    --disable-overlap-schedule
    --decode-log-interval 1
    --host 127.0.0.1
    --port "$PORT"
    --watchdog-timeout 3600
    --skip-server-warmup
)

echo "============================================"
echo "  NGram E2E Benchmark (TPU)"
echo "============================================"
echo "Model:        $MODEL"
echo "TP:           $TP_SIZE"
echo "Questions:    $NUM_QUESTIONS"
echo "Parallel:     $PARALLEL"
echo "Draft tokens: $DRAFT_TOKENS"
echo "NGram breadth:$NGRAM_BREADTH"
echo "============================================"

# ---- Baseline (no spec decode) ----
if [ "$SKIP_BASELINE" != "1" ]; then
    echo ""
    echo "==== Phase 1: Baseline (no speculative decoding) ===="
    cleanup

    python -u -m sglang.launch_server "${BASE_ARGS[@]}" &
    SERVER_PID=$!

    if wait_for_server 600; then
        run_benchmark "baseline"
    else
        echo "Baseline server failed to start"
        tail -20 /proc/$SERVER_PID/fd/2 2>/dev/null || true
    fi

    cleanup
fi

# ---- NGram ----
if [ "$SKIP_NGRAM" != "1" ]; then
    echo ""
    echo "==== Phase 2: NGram speculative decoding ===="
    cleanup

    NGRAM_ARGS=(
        "${BASE_ARGS[@]}"
        --page-size 1
        --speculative-algorithm NGRAM
        --speculative-num-draft-tokens "$DRAFT_TOKENS"
        --speculative-ngram-max-bfs-breadth "$NGRAM_BREADTH"
    )

    python -u -m sglang.launch_server "${NGRAM_ARGS[@]}" &
    SERVER_PID=$!

    if wait_for_server 600; then
        run_benchmark "ngram"
    else
        echo "NGram server failed to start"
        tail -20 /proc/$SERVER_PID/fd/2 2>/dev/null || true
    fi

    cleanup
fi

echo ""
echo "============================================"
echo "  Results Summary"
echo "============================================"
echo "Results saved to: $RESULT_DIR/"
for f in "$RESULT_DIR/${TIMESTAMP}_"*.jsonl; do
    if [ -f "$f" ]; then
        label=$(basename "$f" .jsonl | sed "s/${TIMESTAMP}_//")
        echo ""
        echo "--- $label ---"
        python3 -c "
import json, sys
with open('$f') as fh:
    for line in fh:
        d = json.loads(line)
        print(f'  Throughput:     {d[\"throughput\"]:.1f} tok/s')
        print(f'  Accept length:  {d[\"accept_length\"]:.2f}')
        print(f'  Latency:        {d[\"latency\"]:.1f}s')
        print(f'  Requests:       {d[\"num_requests\"]}')
"
    fi
done
echo ""
echo "============================================"
