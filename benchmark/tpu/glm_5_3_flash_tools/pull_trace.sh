#!/bin/bash
# Usage: pull_trace.sh <remote xplane.pb> <local dir> -- chunked pull with md5 check.
set -euo pipefail
R=$1; D=$2; P=${POD:-tpu-dev-k3-workers-0-1-x9c8c}
K=(kubectl --context gke_sglang-torchtpu_us-central1-c_tpu-cluster -n "${NS:-tpu-dev-wenhao}" exec "$P" -c "${CONTAINER:-main}" --)
mkdir -p "$D/plugins/profile/run"; OUT="$D/plugins/profile/run/$(basename "$R")"
size=$("${K[@]}" stat -c %s "$R"); n=$(( (size + 20971519) / 20971520 ))
: > "$OUT"
for ((i=0; i<n; i++)); do
  for t in 1 2 3 4; do
    "${K[@]}" dd if="$R" bs=20M skip=$i count=1 iflag=fullblock status=none > "$OUT.part" && break
  done
  cat "$OUT.part" >> "$OUT"
done
rm -f "$OUT.part"
want=$("${K[@]}" md5sum "$R" | cut -c1-32); got=$(md5 -q "$OUT")
echo "size=$size local=$(stat -f %z "$OUT") md5 remote=$want local=$got"; [[ $want == "$got" ]]
