#!/bin/bash
# MTP 5/1/6 e2e at simulated acceptance 3.5 (tree D, MLA fp8): 8k1k cookbook bench c1, then a decode profile.
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
exec > $W/glm-run/tpu-mtp-acc35-b.out 2>&1
. $W/tpu_env.sh
tag=mtp-acc35-b
bash $W/stop_wait.sh
bash $W/run-cfg-port.sh glm_5_3_flash_eval/prof_mla_fp8_mtp516_acc35.py $tag glm-port-b venv-port-b
until grep -qE "fired up|Traceback" $W/glm-run/$tag.log 2>/dev/null; do sleep 20; done
grep -q Traceback $W/glm-run/$tag.log && { echo "BOOT FAILED"; grep -m1 -B5 -A40 Traceback $W/glm-run/$tag.log | tail -45; echo ACC35_DONE; exit 1; }
bash $W/bench-cb.sh mtp_acc35_verifyop 1
$H/venv-port-b/bin/python $W/prof_round.py $W/prof-mtp-acc35-b TPU,CPU
echo ACC35_DONE
