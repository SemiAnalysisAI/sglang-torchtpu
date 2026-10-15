#!/bin/bash
# Usage: std-chain.sh <config:tag> ... -- for each, restart the server on that recipe and run the standard 8k1k c1 bench.
W=/dev/shm/home/qinghanc/work
for pair in "$@"; do
  cfg=${pair%%:*}; tag=${pair##*:}
  bash $W/stop_sglang.sh; sleep 15
  bash $W/run-cfg.sh glm_5_3_flash/$cfg $tag
  until grep -qE "fired up|Traceback" $W/glm-run/$tag.log 2>/dev/null; do sleep 20; done
  grep -q Traceback $W/glm-run/$tag.log && { echo "$tag BOOT FAILED"; continue; }
  bash $W/bench-std.sh $tag 1
  until grep -qE "summary appended|Traceback" $W/glm-run/bench-$tag.log 2>/dev/null; do sleep 30; done
  echo "== $tag"; grep -E "Output token thr|Accept|Mean TTFT|Median TTFT|Mean TPOT|Median TPOT" $W/glm-run/bench-$tag.log
done
echo CHAIN_DONE
