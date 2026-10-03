#!/usr/bin/env bash
# LTX-2 text-to-video + audio on one tpu7x 2x2x1 host (8 cores): CFG 2 x Ulysses 4.
set -euo pipefail
exec env MODEL="${MODEL:-Lightricks/LTX-2}" NUM_GPUS="${NUM_GPUS:-8}" \
  PARALLEL_ARGS="${PARALLEL_ARGS:---cfg-parallel-size 2 --ulysses-degree 4}" \
  bash "$(dirname "$0")/launch.sh"
