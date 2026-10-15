#!/bin/bash
# Rerun the AIME26 problems truncated at 290k (P9 P15 P28 P29 P30) with MTP at a 320k budget.
W=/dev/shm/home/qinghanc/work
exec > $W/glm-run/aime-rerun-failed.out 2>&1
. $W/tpu_env.sh
export OPENAI_API_KEY=EMPTY
tag=mla_mtp516_330k
bash $W/stop_wait.sh
bash $W/run-cfg.sh glm_5_3_flash_eval/eval_$tag.py eval-$tag
until grep -qE "fired up|Traceback" $W/glm-run/eval-$tag.log 2>/dev/null; do sleep 20; done
grep -q Traceback $W/glm-run/eval-$tag.log && { echo "BOOT FAILED"; exit 1; }
cd $W/glm-dsa-tpu/python && $HOME/venv/bin/python $W/aime26.py --tag rerun320k --problems 9,15,28,29,30 \
  --threads 5 --max-tokens 320000 2>&1 | grep -vE "Warning|warn" | tail -5
cp aime26-rerun320k.json $W/glm-run/
echo CHAIN_DONE
