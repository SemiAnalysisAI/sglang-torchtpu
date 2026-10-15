#!/bin/bash
# Usage: tpu-mtp-hosttimer.sh <tree> <venv> -- MTP accept-3.5 8k1k c1 with the in-process host timer; prints HOSTTIMER reports.
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc; T=$1; V=$2
exec > $W/glm-run/tpu-mtp-hosttimer.out 2>&1
. $W/tpu_env.sh
export HOME=$H
export QH_HOST_TIMER=1 QH_HOST_TIMER_EVERY=100 QH_CPROFILE=${QH_CPROFILE:-}
cp $W/host_timer_eagle.py $W/$T/python/sglang/srt/managers/qh_host_timer.py
$H/$V/bin/python $W/patch_host_timer_tree.py $W/$T
tag=mtp-hosttimer
bash $W/stop_wait.sh
bash $W/run-cfg-port.sh glm_5_3_flash_eval/prof_mla_fp8_mtp516_acc35.py $tag $T $V
until grep -qE "fired up|Traceback" $W/glm-run/$tag.log 2>/dev/null; do sleep 20; done
grep -q Traceback $W/glm-run/$tag.log && { echo "BOOT FAILED"; grep -m1 -B5 -A40 Traceback $W/glm-run/$tag.log | tail -45; echo HT_DONE; exit 1; }
bash $W/bench-cb.sh mtp_hosttimer 1
grep -c "HOSTTIMER" $W/glm-run/$tag.log
$H/$V/bin/python $W/patch_host_timer_tree.py $W/$T --undo
echo HT_DONE
