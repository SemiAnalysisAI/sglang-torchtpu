#!/bin/bash
# Usage: tpu-real-run.sh <config under glm_5_3_flash_eval/> <tree> <venv> <tag> -- boot, real-text 8k1k c1 (3 runs), cookbook random 8k1k c1 (no simulated acceptance).
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc; CFG=$1; T=$2; V=$3; tag=$4
export HOME=$H
exec > $W/glm-run/real-$tag.out 2>&1
. $W/tpu_env.sh
bash $W/stop_wait.sh
bash $W/run-cfg-port.sh glm_5_3_flash_eval/$CFG.py $tag $T $V
until grep -qE "fired up|Traceback" $W/glm-run/$tag.log 2>/dev/null; do sleep 20; done
grep -q Traceback $W/glm-run/$tag.log && { echo "BOOT FAILED"; grep -m1 -B5 -A40 Traceback $W/glm-run/$tag.log | tail -45; echo REAL_DONE; exit 1; }
echo "== real-text 8k1k c1"; $H/venv/bin/python $W/bench_real.py --runs 3 --osl 1024 --docs $W/glm-dsa-tpu 2>&1 | tail -4
grep -a "accept len" $W/glm-run/$tag.log | tail -3 | sed 's/.*accept len/accept len/'
bash $W/bench-cb.sh $tag-random 1
echo REAL_DONE
