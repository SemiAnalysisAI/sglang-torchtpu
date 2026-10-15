#!/bin/bash
# Usage: boot-prof.sh <config under glm_5_3_flash_eval, no .py> -- boot it, warm up, capture decode rounds.
W=/dev/shm/home/qinghanc/work
bash $W/stop_wait.sh
bash $W/run-cfg.sh glm_5_3_flash_eval/$1.py prof-$1
until grep -qE "fired up|Traceback" $W/glm-run/prof-$1.log 2>/dev/null; do sleep 20; done
grep -q Traceback $W/glm-run/prof-$1.log && { echo "BOOT FAILED"; exit 1; }
bash $W/prof_mtp.sh $W/prof-$1
echo PROF_DONE
