#!/bin/bash
# Usage: deploy-memfix.sh -- push the memory fixes to the three slices:
#   run-cfg-port.sh       compile cache on shared storage only (tier 2 in /dev/shm off), next boot on
#   memguard.sh           above 760 GiB drop checkpoint files from the page cache, stop sglang at 820
#   drop_weight_cache.py  posix_fadvise(DONTNEED) over the weight dirs
# Each file is written to a temp name and renamed, so a running copy keeps its old inode. A new
# memguard is started next to the old one (the old one only ever stops sglang at 820 GiB).
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd); W=/dev/shm/home/qinghanc/work
C=gke_sglang-torchtpu_us-central1-c_tpu-cluster
for P in tpuq-qinghan-4d14-w-0-0-m27gm tpuq-qinghan-3131-w-0-0-7bnjs tpuq-qinghan-4836-w-0-0-d29h5; do
  K=(kubectl --context $C -n tpuq exec -i "$P" -c main --)
  for f in run-cfg-port.sh memguard.sh drop_weight_cache.py; do
    "${K[@]}" tee "$W/.$f.tmp" < "$S/$f" > /dev/null
    "${K[@]}" mv "$W/.$f.tmp" "$W/$f"
  done
  "${K[@]}" setsid -f bash "$W/memguard.sh" 820 760 < /dev/null > /dev/null 2>&1
  echo "== $P"
  "${K[@]}" python3 "$W/drop_weight_cache.py" /weights /work/qinghan/weights
done
