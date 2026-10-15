#!/bin/bash
# Slice 1: DFlash with DSA at 330k -- boot, smoke (needle past index_topk), then AIME26 avg@4.
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc; tag=aime-dflash-dsa
export HOME=$H
exec > $W/glm-run/dsa-s1.out 2>&1
. $W/tpu_env.sh
bash $W/stop_wait.sh
bash $W/run-cfg-port.sh glm_5_3_flash_eval/aime_dflash_dsa_330k.py $tag glm-port-b venv-port-b
until grep -qE "fired up|Traceback" $W/glm-run/$tag.log 2>/dev/null; do sleep 20; done
grep -q Traceback $W/glm-run/$tag.log && { echo "DSA BOOT FAILED"; grep -m1 -B5 -A40 Traceback $W/glm-run/$tag.log | tail -45; echo DSA_DONE; exit 1; }
grep -a -m3 -E "attends densely|max_total_num_tokens|max_running_requests" $W/glm-run/$tag.log | cut -c1-220
echo "== smoke"
$H/venv-port-b/bin/python $W/dsa_sample.py 2>&1 | tail -8
echo SMOKE_DONE
mkdir -p /work/qinghan/aime26
cd /work/qinghan/aime26 && $H/venv-port-b/bin/python $W/aime26.py --tag $tag --repeats 4 --threads 8 --max-tokens 320000
echo DSA_DONE
