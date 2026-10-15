#!/bin/bash
# Usage: nohup bash run-dsa-fix-round.sh > dsa-fix-round.log 2>&1 &
# One DSA fix round on the three slices (DFlash + DSA, 8k1k c1, simulated accept 3):
#   slice 1 (4d14): option 2   -- MoE prefill gathers on XLA, indexer top-k on SparseCore
#   3131, 4836    : option 1+2 -- MoE gathers and indexer top-k both on XLA
# Tier-2 compile cache is off in the config; checkpoint files are dropped from the page cache
# after weight load and every 2 min while compiling. A slice that boots is benched right away.
# Results: this script's log, plus $W/glm-run/boot-<tag>.out and bench-<tag>-c1.log on each slice.
S=$(cd "$(dirname "$0")" && pwd); R=$S/../../glm-dsa-tpu-rebase
W=/dev/shm/home/qinghanc/work; C=gke_sglang-torchtpu_us-central1-c_tpu-cluster
RUNS="tpuq-qinghan-4d14-w-0-0-m27gm:glm-port-b:venv-port-b:r-opt2
tpuq-qinghan-3131-w-0-0-7bnjs:glm-port:venv-port:r-opt12
tpuq-qinghan-4836-w-0-0-d29h5:glm-port:venv-port:r-opt12b"

# 4836 gets the XLA top-k indexer back (it was reset to HEAD for an option-2 run).
kubectl --context $C -n tpuq exec -i tpuq-qinghan-4836-w-0-0-d29h5 -c main -- \
  tee $W/glm-port/sglang_tpu/attention/pallas/dsa_kpool_topk.py \
  < $R/sglang_tpu/attention/pallas/dsa_kpool_topk.py > /dev/null

for spec in $RUNS; do
  IFS=: read P T V TAG <<< "$spec"
  K=(kubectl --context $C -n tpuq exec $P -c main --)
  echo "$(date +%T) $P: xla-topk=$("${K[@]}" grep -c 'topk_impl="xla"' $W/$T/sglang_tpu/attention/pallas/dsa_kpool_topk.py), launching $TAG"
  "${K[@]}" env HOME=/dev/shm/home/qinghanc setsid -f bash $W/tpu-boot-only.sh \
    c1_dsa_dflash_acc3_moe_nosc $TAG $T $V < /dev/null > /dev/null 2>&1
done

left=$(echo $RUNS | tr ' ' '\n' | awk -F: '{print $1":"$4}' | tr '\n' ' '); n=0
while [ -n "${left// }" ] && [ $n -lt 120 ]; do
  next=""
  for r in $left; do
    P=${r%%:*}; TAG=${r#*:}; K=(kubectl --context $C -n tpuq exec $P -c main --)
    if "${K[@]}" grep -q BOOT_DONE $W/glm-run/boot-$TAG.out 2>/dev/null; then
      res=$("${K[@]}" grep -av PrivateUse1 $W/glm-run/boot-$TAG.out | tr '\n' ' ' | cut -c1-300)
      echo "$(date +%T) $TAG: $res"
      if echo "$res" | grep -q BOOT_OK; then
        "${K[@]}" env HOME=/dev/shm/home/qinghanc setsid -f bash $W/bench-cb.sh $TAG 1 < /dev/null > /dev/null 2>&1
        echo "$(date +%T) $TAG: bench started (glm-run/bench-$TAG-c1.log)"
      fi
    else
      next="$next $r"
      [ $((n % 4)) -eq 0 ] && "${K[@]}" python3 $W/drop_weight_cache.py /weights /work/qinghan/weights > /dev/null 2>&1
    fi
  done
  left=$next; n=$((n + 1)); [ -n "${left// }" ] && sleep 30
done
echo "$(date +%T) round done${left:+, still running: $left}"
