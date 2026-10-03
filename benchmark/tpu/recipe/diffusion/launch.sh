#!/usr/bin/env bash
# Shared launcher for the diffusion recipes in this directory. Model scripts set
# MODEL, PARALLEL_ARGS and MODEL_ARGS; everything else can be overridden from the
# environment.
set -euo pipefail

export SGLANG_PLATFORM="${SGLANG_PLATFORM:-tpu}"
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
# Highest TPU clock, and a 128 MiB scoped-VMEM limit for XLA fusions (larger
# tiles: FLUX.2-klein 0.96 -> 0.86 s, Wan2.2-T2V-A14B 2.77 -> 2.73 s/step). The
# published numbers use both.
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_dvfs_p_state=7 --xla_tpu_scoped_vmem_limit_kib=131072}"

: "${MODEL:?set MODEL}"
NUM_GPUS="${NUM_GPUS:-8}"
PARALLEL_ARGS="${PARALLEL_ARGS:-}"
MODEL_ARGS="${MODEL_ARGS:-}"
PORT="${PORT:-30000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# shellcheck disable=SC2086
exec python3 -m sglang.multimodal_gen.runtime.launch_server \
  --model-path "${MODEL}" \
  --num-gpus "${NUM_GPUS}" \
  ${PARALLEL_ARGS} \
  ${MODEL_ARGS} \
  --attention-backend tpu_splash_attn \
  --enable-torch-compile \
  --encoder-parallel fold \
  --text-encoder-cpu-offload false \
  --host 0.0.0.0 \
  --port "${PORT}" \
  ${EXTRA_ARGS}
