#!/bin/bash
# Usage: a-round.sh <config> <tag> -- boot, sample outputs, chat c1 bench, accept-3-free std 8k1k, then profile a round.
W=/dev/shm/home/qinghanc/work
. $W/tpu_env.sh
bash $W/stop_wait.sh
bash $W/run-cfg.sh glm_5_3_flash_eval/$1.py $2
until grep -qE "fired up|Traceback" $W/glm-run/$2.log 2>/dev/null; do sleep 20; done
grep -q Traceback $W/glm-run/$2.log && { echo "BOOT FAILED"; exit 1; }
echo "== samples"; $HOME/venv/bin/python $W/chat_sample.py 300 2>&1 | grep -E "PROMPT|rep4|391" | head -8
echo "== chat"; $HOME/venv/bin/python $W/bench_real.py --runs 3 --osl 1024 --docs $W/glm-dsa-tpu 2>&1 | tail -1
cd $W && $HOME/venv/bin/python prof_round.py $W/prof-$2 TPU,CPU && find $W/prof-$2 -name "*.xplane.pb" | grep TP-0
echo ROUND_DONE
