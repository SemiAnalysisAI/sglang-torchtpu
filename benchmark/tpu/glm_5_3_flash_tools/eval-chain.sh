#!/bin/bash
# Usage: eval-chain.sh <tag> ... -- for each, start recipe glm_5_3_flash_eval/eval_<tag>.py and run AIME26.
W=/dev/shm/home/qinghanc/work
. $W/tpu_env.sh
export OPENAI_API_KEY=EMPTY
for tag in "$@"; do
  bash $W/stop_wait.sh
  bash $W/run-cfg.sh glm_5_3_flash_eval/eval_$tag.py eval-$tag
  until grep -qE "fired up|Traceback" $W/glm-run/eval-$tag.log 2>/dev/null; do sleep 20; done
  grep -q Traceback $W/glm-run/eval-$tag.log && { echo "$tag BOOT FAILED"; continue; }
  cd $W/glm-dsa-tpu/python && $HOME/venv/bin/python $W/aime26.py --tag $tag --max-tokens ${AIME_MAX_TOKENS:-290000} 2>&1 | grep -vE "Warning|warn" | tail -5
  cp aime26-$tag.json $W/glm-run/ 2>/dev/null
done
echo CHAIN_DONE
