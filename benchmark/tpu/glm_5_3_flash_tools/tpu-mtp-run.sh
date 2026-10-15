#!/bin/bash
# Usage: tpu-mtp-run.sh <config under glm_5_3_flash_eval/> <tree> <venv> <tag> -- boot, cookbook 8k1k c1 bench, round profile.
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc; CFG=$1; T=$2; V=$3; tag=$4
export HOME=$H
exec > $W/glm-run/run-$tag.out 2>&1
. $W/tpu_env.sh
bash $W/stop_wait.sh
bash $W/run-cfg-port.sh glm_5_3_flash_eval/$CFG.py $tag $T $V
until grep -qE "fired up|Traceback" $W/glm-run/$tag.log 2>/dev/null; do sleep 20; done
grep -q Traceback $W/glm-run/$tag.log && { echo "BOOT FAILED"; grep -m1 -B5 -A40 Traceback $W/glm-run/$tag.log | tail -45; echo RUN_DONE; exit 1; }
bash $W/bench-cb.sh $tag 1
$H/$V/bin/python $W/prof_round.py $W/prof-$tag TPU,CPU
echo RUN_DONE
