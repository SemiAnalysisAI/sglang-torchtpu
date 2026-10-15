#!/bin/bash
# Usage: deploy-port.sh <pod> [git ref, default HEAD] [dest dir under work/, default glm-port]
# Copies that commit of the glm-dsa-tpu-rebase tree (ported onto main) to the pod.
set -euo pipefail
P=$1; REF=${2:-HEAD}; DEST=${3:-glm-port}; H=/dev/shm/home/qinghanc
K=(kubectl --context gke_sglang-torchtpu_us-central1-c_tpu-cluster -n tpuq exec -i "$P" -c main --)
"${K[@]}" mkdir -p $H/work/$DEST
cd /Users/qinghan.chen/Desktop/sglang-jax-workdir/glm-dsa-tpu-rebase
git archive --format=tar "$REF" python sglang_tpu benchmark/tpu test/registered tools pyproject.toml \
  | "${K[@]}" tar xf - -C $H/work/$DEST
echo "deployed $(git rev-parse --short "$REF") to $P:$H/work/$DEST"
