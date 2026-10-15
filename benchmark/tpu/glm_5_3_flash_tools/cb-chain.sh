#!/bin/bash
# Usage: cb-chain.sh <tag> ... -- start glm_5_3_flash_eval/cb_<tag>.py, run cookbook-parity c1 and c16.
W=/dev/shm/home/qinghanc/work
for tag in "$@"; do
  bash $W/stop_wait.sh
  bash $W/run-cfg.sh glm_5_3_flash_eval/cb_$tag.py cb-$tag
  until grep -qE "fired up|Traceback" $W/glm-run/cb-$tag.log 2>/dev/null; do sleep 20; done
  grep -q Traceback $W/glm-run/cb-$tag.log && { echo "$tag BOOT FAILED"; continue; }
  bash $W/bench-cb.sh $tag 1; bash $W/bench-cb.sh $tag 16
done
echo CHAIN_DONE
