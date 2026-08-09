#!/usr/bin/env bash
# Sweep real-text serving performance for the Qwen3-Coder baseline or EAGLE3 server.
# Defaults to 10 requests per concurrency at 1k input / 1k output.
set -euo pipefail

VARIANT=${VARIANT:-eagle3-real}
export ISL=${ISL:-1024}
export OSL=${OSL:-1024}
CONCUR_SWEEP=${CONCUR_SWEEP:-"1 16 32"}
export PROMPTS_PER_CONCURRENCY=${PROMPTS_PER_CONCURRENCY:-10}
export WARMUP_REQUESTS=${WARMUP_REQUESTS:-auto}

# CDK recipes set MODEL to the GCS weight URI used by the server.  The client
# must instead use a tokenizer/served-model identifier that Transformers can
# resolve, so keep the benchmark override in a separate variable.
export MODEL=${BENCH_MODEL:-Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8}
export TOKENIZER=${BENCH_TOKENIZER:-$MODEL}
export DATASET_NAME=random
export TOKENIZE_PROMPT=0

# CDK_OUTPUT_DIR is already unique per pod/job. Keep local runs timestamped by
# the shared benchmark driver, while placing CDK results in that isolated root.
if [[ -n "${CDK_OUTPUT_DIR:-}" ]]; then
    export OUTPUT_DIR=${OUTPUT_DIR:-$CDK_OUTPUT_DIR/$VARIANT}
fi

SWEEP=""
for concurrency in $CONCUR_SWEEP; do
    SWEEP="${SWEEP:+$SWEEP }${concurrency}:${ISL}:${OSL}"
done
export SWEEP

cd "$(dirname "$0")/../.."
exec ./bench_serving.sh "$@"
