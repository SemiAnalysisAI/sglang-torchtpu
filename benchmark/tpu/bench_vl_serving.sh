#!/usr/bin/env bash
# Model-agnostic VL (vision-language) bench_serving driver for an sglang
# torch_tpu server. Companion to ./bench_serving.sh, but drives an *image*
# workload: each request carries one or more images plus a text prompt, so it
# exercises the multimodal path (ViT encode + embedding scatter + LLM
# prefill/decode) rather than text-only prefill.
#
# Two datasets are supported via DATASET_NAME:
#   mmmu  (default) — real multimodal VQA prompts/images pulled from the MMMU
#                     benchmark. Input length comes from the dataset (ISL is
#                     ignored); OSL caps generation. More realistic, but image
#                     resolutions vary, so on TPU the first request at each new
#                     ViT patch-count bucket pays a one-time compile cost unless
#                     the server already precompiled that bucket.
#   image           — synthetic images at a fixed resolution. Reproducible and
#                     download-free; lets you pin a resolution that matches the
#                     server's precompiled TPU ViT patch-count buckets. Uses
#                     ISL/OSL and the IMAGE_* knobs below.
#
# Set MODEL to match whatever ./launch_server.sh is serving (a VL model, e.g.
# Qwen/Qwen3-VL-2B-Instruct). Launch the server first; see ./README.md for the
# shared env contract (PYTHON, JAX_BAZEL, HF cache).
#
# Example — MMMU, default single run (concurrency=16, osl=512):
#   MODEL=Qwen/Qwen3-VL-2B-Instruct ./bench_vl_serving.sh
#
# Example — synthetic 448x448 images (784 ViT patches at merge_size=2),
# 1 image/request, concurrency sweep:
#   MODEL=Qwen/Qwen3-VL-2B-Instruct DATASET_NAME=image \
#   IMAGE_RESOLUTION=448x448 IMAGE_COUNT=1 \
#   SWEEP="1:256:128 8:256:128 16:256:128" ./bench_vl_serving.sh
#
# Each SWEEP token is "concurrency:isl:osl". The server's max-running-requests
# must be >= the largest concurrency. For DATASET_NAME=mmmu the isl field is
# ignored (input comes from the dataset); it still must be present so the tuple
# parses. Per-tuple num_prompts defaults to concurrency x PROMPTS_PER_CONCURRENCY.
#
# Overridable env vars (defaults shown):
#   MODEL=                          # REQUIRED — must match the served VL model
#   BASE_URL=http://127.0.0.1:30000
#   BACKEND=sglang
#   DATASET_NAME=mmmu               # mmmu | image
#   ISL=256                         # input length — synthetic 'image' only
#   OSL=512                         # output (generation) length cap
#   BATCH_SIZE=16                   # single-run max-concurrency
#   NUM_PROMPTS=                    # single-run total prompts (defaults to
#                                   # BATCH_SIZE * PROMPTS_PER_CONCURRENCY)
#   PROMPTS_PER_CONCURRENCY=4       # used to derive NUM_PROMPTS when unset
#   IMAGE_COUNT=1                   # images per request (synthetic 'image')
#   IMAGE_RESOLUTION=448x448        # 'image' dataset: 4k/1080p/720p/360p preset
#                                   # or custom HxW. 448x448 -> 784 ViT patches
#                                   # at patch_size*merge_size=32, a common
#                                   # precompiled TPU bucket.
#   IMAGE_FORMAT=jpeg               # jpeg | png  (synthetic 'image')
#   IMAGE_CONTENT=random            # random | blank  (synthetic 'image')
#   RANDOM_IMAGE_COUNT=0            # 1 -> randomize per-request image count
#   WARMUP_REQUESTS=0               # >0 or "auto" (->leg concurrency) re-enables
#                                   # bench_serving's own warmup
#   SEED=1
#   PROFILE=0                       # 1 -> torch profiler /start_profile capture
#   PROFILE_NUM_STEPS=5
#   PROFILE_OUTPUT_DIR=             # auto-generated when PROFILE=1
#   OUTPUT_DIR=./results/<model-slug>-vl-tpu/<ts>
#   SWEEP=                          # space-separated "c:isl:osl" tuples
#   HEALTH_TIMEOUT=600              # seconds to wait for /health -> 200
#   PYTHON=python3                  # interpreter with torch_tpu + sglang
#   JAX_BAZEL=                      # opt-in; only for a Bazel-built torch_tpu

set -euo pipefail

MODEL=${MODEL:-}
if [[ -z "$MODEL" ]]; then
    echo "ERROR: set MODEL, e.g. MODEL=Qwen/Qwen3-VL-2B-Instruct ./bench_vl_serving.sh" >&2
    exit 2
fi
# Slug for the default results dir / profile prefix: lowercase, non-alnum -> '_'.
MODEL_SLUG=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/__*/_/g; s/^_//; s/_$//')

BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
BACKEND=${BACKEND:-sglang}
DATASET_NAME=${DATASET_NAME:-mmmu}
case "$DATASET_NAME" in
    mmmu|image) ;;
    *) echo "ERROR: DATASET_NAME must be 'mmmu' or 'image' (got '$DATASET_NAME')" >&2; exit 2 ;;
esac

ISL=${ISL:-256}
OSL=${OSL:-512}
BATCH_SIZE=${BATCH_SIZE:-16}
PROMPTS_PER_CONCURRENCY=${PROMPTS_PER_CONCURRENCY:-4}
NUM_PROMPTS=${NUM_PROMPTS:-}

IMAGE_COUNT=${IMAGE_COUNT:-1}
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-448x448}
IMAGE_FORMAT=${IMAGE_FORMAT:-jpeg}
IMAGE_CONTENT=${IMAGE_CONTENT:-random}
RANDOM_IMAGE_COUNT=${RANDOM_IMAGE_COUNT:-0}

WARMUP_REQUESTS=${WARMUP_REQUESTS:-0}
SEED=${SEED:-1}
PROFILE=${PROFILE:-0}
PROFILE_NUM_STEPS=${PROFILE_NUM_STEPS:-5}
PROFILE_OUTPUT_DIR=${PROFILE_OUTPUT_DIR:-}
SWEEP=${SWEEP:-}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-600}

TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR:-./results/${MODEL_SLUG}-vl-tpu/$TS}
mkdir -p "$OUTPUT_DIR"

# A torch_tpu built from source with Bazel keeps jax/jaxlib/ml_dtypes/
# opt_einsum/scipy in its Bazel external repo. If you built that way, point
# JAX_BAZEL at <bazel-output-user-root>/<hash>/external so `python -m
# sglang.bench_serving` can import them. Pip-installed torch_tpu does not
# need this — leave JAX_BAZEL unset. (See ./launch_server.sh / ./README.md.)
JAX_BAZEL=${JAX_BAZEL:-}
if [[ -n "$JAX_BAZEL" ]]; then
    _jax_pp=""
    for _pkg in jax jaxlib ml_dtypes opt_einsum scipy; do
        _jax_pp="${_jax_pp:+$_jax_pp:}$JAX_BAZEL/rules_python++pip+torch_tpu_pypi_312_${_pkg}/site-packages"
    done
    export PYTHONPATH="${_jax_pp}${PYTHONPATH:+:$PYTHONPATH}"
fi

# /health returns 503 while sglang is still warming up compile-buckets,
# then flips to 200 once the scheduler accepts requests. `curl -fsS` would
# fail-fast on 503, so we poll: re-issue every 5 s until we either see
# 200 or exhaust HEALTH_TIMEOUT seconds.
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

# Single-config runner; the sweep loop calls this once per tuple. Echoes
# the path of the JSONL summary it produced.
run_one() {
    local bs="$1"
    local isl="$2"
    local osl="$3"
    local num_prompts="$4"

    local tag="c${bs}-isl${isl}-osl${osl}"
    local output_file="$OUTPUT_DIR/${tag}.jsonl"
    local profile_prefix="${MODEL_SLUG}-vl-${tag}"

    # Resolve "auto" → concurrency so the warmup actually exercises the
    # measurement's batch bucket.
    local warmup_n="$WARMUP_REQUESTS"
    [[ "$warmup_n" == "auto" ]] && warmup_n="$bs"

    local args=(
        -m sglang.bench_serving
        --backend "$BACKEND"
        --base-url "$BASE_URL"
        --model "$MODEL"
        --dataset-name "$DATASET_NAME"
        --random-output-len "$osl"
        --num-prompts "$num_prompts"
        --max-concurrency "$bs"
        --warmup-requests "$warmup_n"
        --seed "$SEED"
        --output-file "$output_file"
    )

    # Synthetic 'image' uses input length + the image-shape knobs; mmmu pulls
    # both prompt text and images from the dataset, so they don't apply there.
    if [[ "$DATASET_NAME" == "image" ]]; then
        args+=(
            --random-input-len "$isl"
            --image-count "$IMAGE_COUNT"
            --image-resolution "$IMAGE_RESOLUTION"
            --image-format "$IMAGE_FORMAT"
            --image-content "$IMAGE_CONTENT"
        )
        [[ "$RANDOM_IMAGE_COUNT" == "1" ]] && args+=(--random-image-count)
    fi

    if [[ "$PROFILE" == "1" ]]; then
        local pdir="${PROFILE_OUTPUT_DIR:-$OUTPUT_DIR/profile-${tag}}"
        mkdir -p "$pdir"
        args+=(
            --profile
            --profile-activities CPU GPU
            --profile-num-steps "$PROFILE_NUM_STEPS"
            --profile-prefix "$profile_prefix"
            --profile-output-dir "$pdir"
        )
    fi

    echo "==== vl bench_serving config ($tag) ===="
    echo "  base_url        = $BASE_URL"
    echo "  model           = $MODEL"
    echo "  backend         = $BACKEND"
    echo "  dataset         = $DATASET_NAME"
    if [[ "$DATASET_NAME" == "image" ]]; then
        echo "  input_len       = $isl"
        echo "  image_count     = $IMAGE_COUNT  (random_count=$RANDOM_IMAGE_COUNT)"
        echo "  image_resolution= $IMAGE_RESOLUTION"
        echo "  image_format    = $IMAGE_FORMAT  content=$IMAGE_CONTENT"
    else
        echo "  input_len       = <from mmmu dataset>"
    fi
    echo "  output_len      = $osl"
    echo "  concurrency     = $bs"
    echo "  num_prompts     = $num_prompts  (= $bs x $PROMPTS_PER_CONCURRENCY)"
    echo "  warmup_requests = $warmup_n  (input: $WARMUP_REQUESTS)"
    echo "  profile         = $PROFILE"
    echo "  output_file     = $output_file"
    echo "========================================"

    "$PYTHON" "${args[@]}"
    echo
    echo "→ summary appended to $output_file"
}

if [[ -n "$SWEEP" ]]; then
    echo "==== sweep over ${SWEEP} ===="
    echo "  output_dir = $OUTPUT_DIR"
    echo "============================="
    for tuple in $SWEEP; do
        IFS=':' read -r sw_bs sw_isl sw_osl <<<"$tuple"
        if [[ -z "$sw_bs" || -z "$sw_isl" || -z "$sw_osl" ]]; then
            echo "ERROR: malformed SWEEP token '$tuple' (expected c:isl:osl)" >&2
            exit 2
        fi
        sw_num_prompts=$(( sw_bs * PROMPTS_PER_CONCURRENCY ))
        run_one "$sw_bs" "$sw_isl" "$sw_osl" "$sw_num_prompts"
    done
    echo "==== sweep done — JSONL summaries in $OUTPUT_DIR ===="
    ls -la "$OUTPUT_DIR"/*.jsonl 2>/dev/null || true
else
    [[ -z "$NUM_PROMPTS" ]] && NUM_PROMPTS=$(( BATCH_SIZE * PROMPTS_PER_CONCURRENCY ))
    run_one "$BATCH_SIZE" "$ISL" "$OSL" "$NUM_PROMPTS"
fi
