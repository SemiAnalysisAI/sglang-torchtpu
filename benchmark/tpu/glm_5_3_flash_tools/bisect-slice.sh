#!/bin/bash
# Usage: bisect-slice.sh <pod> <variant file> <tag> [path under glm-port, default the kpool step]
# Bootstrap a fresh slice at HEAD, drop in the variant file, boot the 8k1k MTP+DSA config once.
set -euo pipefail
P=$1; V=$2; TAG=$3; DEST=${4:-sglang_tpu/attention/pallas/dsa_kpool_topk.py}; S=$(cd "$(dirname "$0")" && pwd); W=/dev/shm/home/qinghanc/work
K=(kubectl --context gke_sglang-torchtpu_us-central1-c_tpu-cluster -n tpuq exec -i "$P" -c main --)
bash $S/bootstrap-port-pod.sh "$P"
"${K[@]}" tee $W/glm-port/$DEST < "$V" > /dev/null
"${K[@]}" tee $W/tpu-boot-only.sh < $S/tpu-boot-only.sh > /dev/null
"${K[@]}" env HOME=/dev/shm/home/qinghanc setsid -f bash $W/tpu-boot-only.sh prof_dsa_mtp516_acc35 "$TAG" < /dev/null > /dev/null 2>&1
echo "bisect $TAG launched on $P"
