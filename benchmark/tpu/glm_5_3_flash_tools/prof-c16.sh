#!/bin/bash
# Usage: prof-c16.sh <config under glm_5_3_flash_eval, no .py> -- boot it and capture c16 decode rounds.
W=/dev/shm/home/qinghanc/work
. $W/tpu_env.sh
bash $W/stop_wait.sh
bash $W/run-cfg.sh glm_5_3_flash_eval/$1.py profc16-$1
until grep -qE "fired up|Traceback" $W/glm-run/profc16-$1.log 2>/dev/null; do sleep 20; done
grep -q Traceback $W/glm-run/profc16-$1.log && { echo "BOOT FAILED"; exit 1; }
cd $W && $HOME/venv/bin/python prof_c16.py $W/profc16-$1
find $W/profc16-$1 -name "*.xplane.pb" | grep TP-0
echo PROF_DONE
