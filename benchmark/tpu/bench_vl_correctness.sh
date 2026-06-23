#!/usr/bin/env bash
# Model-agnostic VL (vision-language) CORRECTNESS driver for an sglang
# torch_tpu server. Companion to ./bench_vl_serving.sh: that script measures
# throughput/TTFT/latency on an image workload but does NOT grade answers; this
# one sends MMMU image+question prompts to the *already-running* server, parses
# the model's multiple-choice answer, and reports accuracy (correct / total).
#
# It is a thin wrapper over ./bench_vl_correctness.py, which in turn reuses the
# existing MMMU harness in ../mmmu read-only for sample loading, image/prompt
# construction, robust answer parsing (handles long "Thinking" reasoning before
# the final choice), and grading. The only things added on top are a clean
# NUM_QUESTIONS cap and a fixed concurrency=1 sequential send loop.
#
# Concurrency is intentionally hard-pinned to 1: the current TPU server crashes
# on concurrent multimodal batches, and concurrency=1 also keeps grading
# deterministic. Image prefills can be slow (tens of seconds TTFT each), so keep
# NUM_QUESTIONS small and REQUEST_TIMEOUT generous.
#
# Set MODEL to match whatever ./launch_server.sh is serving (a VL model, e.g.
# Qwen/Qwen3-VL-30B-A3B-Thinking). Launch the server first; see ./README.md for
# the shared env contract (PYTHON, JAX_BAZEL, HF cache).
#
# Example — grade 10 MMMU questions against a running Thinking server:
#   MODEL=Qwen/Qwen3-VL-30B-A3B-Thinking NUM_QUESTIONS=10 ./bench_vl_correctness.sh
#
# Example — grade the full MMMU validation split (slow!):
#   MODEL=Qwen/Qwen3-VL-2B-Instruct NUM_QUESTIONS=0 ./bench_vl_correctness.sh
#
# Overridable env vars (defaults shown):
#   MODEL=                          # REQUIRED — must match the served VL model
#   BASE_URL=http://127.0.0.1:30000 # server URL; PORT is derived from it
#   NUM_QUESTIONS=10                # MMMU questions to grade (0 = full split)
#   MAX_NEW_TOKENS=                 # cap generated tokens (unset = server default;
#                                   # Thinking models need a high cap, e.g. 4096)
#   TEMPERATURE=                    # sampling temperature (unset = server default)
#   REASONING_EFFORT=               # none | high  (Thinking-style models only)
#   RESPONSE_ANSWER_REGEX=          # optional regex w/ 1 capture group to pull the
#                                   # answer out before MMMU's own parser runs
#   DATASET_PATH=MMMU/MMMU          # HF dataset id
#   SPLIT=validation                # dataset split
#   SEED=1
#   REQUEST_TIMEOUT=1800            # per-request client timeout (seconds)
#   OUTPUT_DIR=./results/<model-slug>-vl-correctness-tpu/<ts>
#   HEALTH_TIMEOUT=600              # seconds to wait for /health -> 200
#   PYTHON=python3                  # interpreter with torch_tpu + sglang + openai
#   JAX_BAZEL=                      # opt-in; only for a Bazel-built torch_tpu

set -euo pipefail

MODEL=${MODEL:-}
if [[ -z "$MODEL" ]]; then
    echo "ERROR: set MODEL, e.g. MODEL=Qwen/Qwen3-VL-30B-A3B-Thinking ./bench_vl_correctness.sh" >&2
    exit 2
fi
# Slug for the default results dir: lowercase, non-alnum -> '_'.
MODEL_SLUG=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/__*/_/g; s/^_//; s/_$//')

BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
# Derive the port from BASE_URL (bench_vl_correctness.py talks to 127.0.0.1:PORT).
PORT=$(printf '%s' "$BASE_URL" | sed -E 's#.*:([0-9]+).*#\1#')
[[ "$PORT" =~ ^[0-9]+$ ]] || PORT=30000

NUM_QUESTIONS=${NUM_QUESTIONS:-10}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-}
TEMPERATURE=${TEMPERATURE:-}
REASONING_EFFORT=${REASONING_EFFORT:-}
RESPONSE_ANSWER_REGEX=${RESPONSE_ANSWER_REGEX:-}
DATASET_PATH=${DATASET_PATH:-MMMU/MMMU}
SPLIT=${SPLIT:-validation}
SEED=${SEED:-1}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-1800}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-600}

TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR:-./results/${MODEL_SLUG}-vl-correctness-tpu/$TS}
mkdir -p "$OUTPUT_DIR"

# Same JAX_BAZEL handling as bench_vl_serving.sh: a Bazel-built torch_tpu keeps
# jax/jaxlib/ml_dtypes/opt_einsum/scipy in its Bazel external repo. Point
# JAX_BAZEL at <bazel-output-user-root>/<hash>/external to import them.
# Pip-installed torch_tpu does not need this — leave JAX_BAZEL unset.
JAX_BAZEL=${JAX_BAZEL:-}
if [[ -n "$JAX_BAZEL" ]]; then
    _jax_pp=""
    for _pkg in jax jaxlib ml_dtypes opt_einsum scipy; do
        _jax_pp="${_jax_pp:+$_jax_pp:}$JAX_BAZEL/rules_python++pip+torch_tpu_pypi_312_${_pkg}/site-packages"
    done
    export PYTHONPATH="${_jax_pp}${PYTHONPATH:+:$PYTHONPATH}"
fi

# Poll /health (it returns 503 while compile-buckets warm up, then 200).
echo "waiting up to ${HEALTH_TIMEOUT}s for $BASE_URL/health to return 200..."
__health_start=$SECONDS
while :; do
    __code=$(curl -sS -o /dev/null -m 10 -w "%{http_code}" "$BASE_URL/health" 2>/dev/null || echo "000")
    if [[ "$__code" == "200" ]]; then
        echo "  /health -> 200 after $((SECONDS - __health_start))s"
        break
    fi
    if (( SECONDS - __health_start >= HEALTH_TIMEOUT )); then
        echo "ERROR: $BASE_URL/health never reached 200 (last code=$__code, waited ${HEALTH_TIMEOUT}s)" >&2
        echo "Start it first with ./launch_server.sh, then rerun." >&2
        exit 1
    fi
    sleep 5
done

PYTHON=${PYTHON:-python3}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

args=(
    "$SCRIPT_DIR/bench_vl_correctness.py"
    --model "$MODEL"
    --base-url "$BASE_URL"
    --port "$PORT"
    --num-questions "$NUM_QUESTIONS"
    --output-dir "$OUTPUT_DIR"
    --request-timeout "$REQUEST_TIMEOUT"
    --dataset-path "$DATASET_PATH"
    --split "$SPLIT"
    --seed "$SEED"
)
[[ -n "$MAX_NEW_TOKENS"        ]] && args+=(--max-new-tokens "$MAX_NEW_TOKENS")
[[ -n "$TEMPERATURE"           ]] && args+=(--temperature "$TEMPERATURE")
[[ -n "$REASONING_EFFORT"      ]] && args+=(--reasoning-effort "$REASONING_EFFORT")
[[ -n "$RESPONSE_ANSWER_REGEX" ]] && args+=(--response-answer-regex "$RESPONSE_ANSWER_REGEX")

echo "==== vl correctness config ===="
echo "  base_url        = $BASE_URL  (port=$PORT)"
echo "  model           = $MODEL"
echo "  dataset         = $DATASET_PATH  split=$SPLIT"
echo "  num_questions   = $NUM_QUESTIONS  (0 = full split)"
echo "  concurrency     = 1  (hard-pinned)"
echo "  max_new_tokens  = ${MAX_NEW_TOKENS:-<server default>}"
echo "  temperature     = ${TEMPERATURE:-<server default>}"
echo "  reasoning_effort= ${REASONING_EFFORT:-<none>}"
echo "  request_timeout = ${REQUEST_TIMEOUT}s"
echo "  output_dir      = $OUTPUT_DIR"
echo "==============================="

"$PYTHON" "${args[@]}"

echo
echo "-> accuracy summary in $OUTPUT_DIR/summary.json"
