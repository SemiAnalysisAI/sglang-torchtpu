#!/bin/bash
# Usage: mla-chain.sh <tag> ... -- start glm_5_3_flash_eval/<tag>.py; sample outputs, chat c1 bench, standard 8k1k c1.
W=/dev/shm/home/qinghanc/work
. $W/tpu_env.sh
for tag in "$@"; do
  bash $W/stop_wait.sh
  bash $W/run-cfg.sh glm_5_3_flash_eval/$tag.py $tag
  until grep -qE "fired up|Traceback" $W/glm-run/$tag.log 2>/dev/null; do sleep 20; done
  grep -q Traceback $W/glm-run/$tag.log && { echo "$tag BOOT FAILED"; continue; }
  echo "== $tag samples"; $HOME/venv/bin/python $W/chat_sample.py 300 2>&1 | tail -40
  echo "== $tag chat"; $HOME/venv/bin/python $W/bench_real.py --runs 3 --osl 1024 --docs $W/glm-dsa-tpu 2>&1 | tail -4
  bash $W/bench-std.sh $tag 1 > /dev/null
  until grep -qE "summary appended|Traceback" $W/glm-run/bench-$tag.log 2>/dev/null; do sleep 20; done
  echo "== $tag std8k1k"; grep -E "Output token thr|Accept|Mean TTFT|Median TPOT" $W/glm-run/bench-$tag.log
done
echo CHAIN_DONE
