#!/usr/bin/env bash
# Qwen3-Coder-480B-A35B-FP8 baseline or SpecForge EAGLE3 on one TPU v7x host.
# Pair with ./bench.sh for the 1/16/32 serving sweep. Set VARIANT=baseline
# for the non-speculative control; the default is eagle3-real.
set -euo pipefail

VARIANT=${VARIANT:-eagle3-real}
ISL=${ISL:-1024}
OSL=${OSL:-1024}
CONCUR=${CONCUR:-32}
export CONTEXT_LEN=${CONTEXT_LEN:-$((ISL + OSL + 20))}
export MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-$((CONCUR * CONTEXT_LEN))}
CHUNK=${CHUNK:-$ISL}

export MODEL=${MODEL:-Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8}
DRAFT_MODEL=${DRAFT_MODEL:-lmsys/SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8}
if [[ "$MODEL" == gs://* ]]; then
    DEFAULT_LOAD_FORMAT=runai_streamer
else
    DEFAULT_LOAD_FORMAT=auto
fi
LOAD_FORMAT=${LOAD_FORMAT:-$DEFAULT_LOAD_FORMAT}
if [[ "$DRAFT_MODEL" == gs://* ]]; then
    DEFAULT_DRAFT_LOAD_FORMAT=runai_streamer
else
    DEFAULT_DRAFT_LOAD_FORMAT=auto
fi
DRAFT_LOAD_FORMAT=${DRAFT_LOAD_FORMAT:-$DEFAULT_DRAFT_LOAD_FORMAT}
SPEC_STEPS=${SPEC_STEPS:-3}
SPEC_TOPK=${SPEC_TOPK:-1}
SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-4}

export TP_SIZE=8
export EP_SIZE=8
export QUANTIZATION=fp8
export KV_CACHE_DTYPE=fp8_e4m3
export ATTENTION_BACKEND=pallas_tpu_rpa_v3
export PAGE_SIZE=64
export MEM_FRACTION_STATIC=0.8
export MAX_RUNNING=$CONCUR
export CHUNKED_PREFILL_SIZE=$CHUNK
export MAX_PREFILL_TOKENS=$CHUNK
export ENABLE_TORCH_COMPILE=1
export TORCH_COMPILE_MAX_BS=$CONCUR
export DISABLE_RADIX_CACHE=1

# Keep shared-NFS caches collision-free. A caller can override either path.
COMMIT_KEY=$(git rev-parse --short=12 HEAD 2>/dev/null || printf unknown)
RUN_LABEL=${RUN_LABEL:-qwen3-perf-testing}
RUN_INSTANCE=${RUN_INSTANCE:-${HOSTNAME:-local}}
RUN_INSTANCE_KEY=$(printf '%s' "$RUN_INSTANCE" | tr -c '[:alnum:]_' '_')
VARIANT_KEY=${VARIANT//-/_}
export TORCH_TPU_TIER2_NAME=${TORCH_TPU_TIER2_NAME:-${RUN_LABEL}_${COMMIT_KEY}_${RUN_INSTANCE_KEY}_${VARIANT_KEY}}
if [[ -d /workspace/cache ]]; then
    # The CDK image supplies a shared generic default. Replace it unless this
    # harness receives a dedicated override, so unrelated jobs never share a
    # writable compilation-cache directory.
    export TORCH_TPU_TIER3_ROOT=${QWEN3_PERF_TIER3_ROOT:-/workspace/cache/${RUN_LABEL}/${COMMIT_KEY}/${RUN_INSTANCE_KEY}/${VARIANT_KEY}/torch_tpu_compile_cache}
fi

export SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD=2048
export SGLANG_TPU_DECODE_PROCESS_INTERVAL=${SGLANG_TPU_DECODE_PROCESS_INTERVAL:-8}
export SGLANG_SIMULATE_ACC_LEN=${SGLANG_SIMULATE_ACC_LEN:--1}

COMMON_EXTRA_ARGS="--tpu-fp8-runtime channelwise --sampling-defaults openai --served-model-name $SERVED_MODEL_NAME"
if [[ "$LOAD_FORMAT" != auto ]]; then
    COMMON_EXTRA_ARGS="$COMMON_EXTRA_ARGS --load-format $LOAD_FORMAT"
fi
case "$VARIANT" in
    baseline)
        export EXTRA_ARGS="$COMMON_EXTRA_ARGS${EXTRA_ARGS:+ $EXTRA_ARGS}"
        ;;
    eagle3-real)
        # Main freezes torch.compile after warmup by default. The speculative
        # path has an explicit opt-in because a new serving shape can still
        # require a lazy compile after the static warmup sweep.
        export SGLANG_TPU_ALLOW_SPEC_RECOMPILE=${SGLANG_TPU_ALLOW_SPEC_RECOMPILE:-1}
        export EXTRA_ARGS="$COMMON_EXTRA_ARGS \
--speculative-algorithm EAGLE3 \
--speculative-draft-model-path $DRAFT_MODEL \
--speculative-draft-load-format $DRAFT_LOAD_FORMAT \
--speculative-num-steps $SPEC_STEPS \
--speculative-eagle-topk $SPEC_TOPK \
--speculative-num-draft-tokens $SPEC_DRAFT_TOKENS${EXTRA_ARGS:+ $EXTRA_ARGS}"
        ;;
    *)
        printf 'Unsupported VARIANT=%s (expected baseline or eagle3-real)\n' "$VARIANT" >&2
        exit 2
        ;;
esac

cd "$(dirname "$0")/../.."
exec ./launch_server.sh "$@"
