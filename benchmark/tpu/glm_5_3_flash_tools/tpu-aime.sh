#!/bin/bash
# Usage: tpu-aime.sh <config> <tree> <venv> <tag> [concurrency, default 32] [reps, e.g. 2,3] -- boot, AIME26 avg@8 at a 320k budget (resumable), copy results to /work.
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc; CFG=$1; T=$2; V=$3; tag=$4; CONC=${5:-32}; REPS=${6:-}
export HOME=$H
exec > $W/glm-run/aime-$tag.out 2>&1
. $W/tpu_env.sh
bash $W/stop_wait.sh
bash $W/run-cfg-port.sh glm_5_3_flash_eval/$CFG.py $tag $T $V
until grep -qE "fired up|Traceback" $W/glm-run/$tag.log 2>/dev/null; do sleep 20; done
grep -q Traceback $W/glm-run/$tag.log && { echo "BOOT FAILED"; grep -m1 -B5 -A40 Traceback $W/glm-run/$tag.log | tail -45; echo AIME_DONE; exit 1; }
grep -a -m3 -E "max_total_num_tokens|max_running_requests" $W/glm-run/$tag.log | cut -c1-220
# Results go straight to /work (survives pod eviction and reservations); resumable.
mkdir -p /work/qinghan/aime26
cd /work/qinghan/aime26 && $H/$V/bin/python $W/aime26.py --tag $tag --repeats 8 --threads $CONC ${REPS:+--reps $REPS} --max-tokens 320000
echo AIME_DONE
