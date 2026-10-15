#!/bin/bash
# Usage: gsm-ab.sh <cfg> ... -- GSM8K (200, greedy) against each long-context no-spec server.
W=/dev/shm/home/qinghanc/work
. $W/tpu_env.sh
export OPENAI_API_KEY=EMPTY
for cfg in "$@"; do
  tag=gsm-$(basename $cfg .py)
  bash $W/stop_wait.sh
  bash $W/run-cfg.sh glm_5_3_flash_eval/$cfg $tag
  until grep -qE "fired up|Traceback" $W/glm-run/$tag.log 2>/dev/null; do sleep 20; done
  grep -q Traceback $W/glm-run/$tag.log && { echo "$tag BOOT FAILED"; continue; }
  cd $W/glm-dsa-tpu/python && $HOME/venv/bin/python -m sglang.test.run_eval --eval-name gsm8k --port 30000 \
     --num-examples 200 --num-threads 8 --max-tokens 4096 --temperature 0 2>&1 | grep -iE "score|accuracy|latency" | tail -3 | sed "s/^/$tag /"
done
echo GSM_DONE
