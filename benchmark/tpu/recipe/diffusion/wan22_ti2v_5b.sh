#!/usr/bin/env bash
# Wan2.2-TI2V-5B text/image-to-video on one tpu7x 2x2x1 host (8 cores): CFG 2 x Ulysses 2 x ring 2.
set -euo pipefail
exec env MODEL="${MODEL:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}" NUM_GPUS="${NUM_GPUS:-8}" \
  PARALLEL_ARGS="${PARALLEL_ARGS:---cfg-parallel-size 2 --ulysses-degree 2 --ring-degree 2}" \
  MODEL_ARGS="${MODEL_ARGS:---regional-compile}" \
  bash "$(dirname "$0")/launch.sh"
