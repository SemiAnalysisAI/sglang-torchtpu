#!/usr/bin/env bash
# FLUX.2-klein-4B text-to-image. 24 heads: Ulysses degree must divide 24.
#   NUM_GPUS=1 ./flux2_klein_4b.sh
#   NUM_GPUS=4 ./flux2_klein_4b.sh        # Ulysses 4
set -euo pipefail
NUM_GPUS="${NUM_GPUS:-1}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  DEFAULT_PARALLEL="--ulysses-degree ${NUM_GPUS}"
else
  DEFAULT_PARALLEL=""
fi
exec env MODEL="${MODEL:-black-forest-labs/FLUX.2-klein-4B}" NUM_GPUS="${NUM_GPUS}" \
  PARALLEL_ARGS="${PARALLEL_ARGS:-${DEFAULT_PARALLEL}}" \
  bash "$(dirname "$0")/launch.sh"
