#!/bin/bash
# Boot the sync-site diagnostic MTP server, drive decode, then show the SYNCSITES report.
W=/dev/shm/home/qinghanc/work
. $W/tpu_env.sh
bash $W/stop_wait.sh
rm -f /dev/shm/home/qinghanc/work/syncsites/ARM
bash $W/run-cfg.sh glm_5_3_flash_eval/mla_fp8_mtp516_sync.py syncdiag
until grep -qE "fired up|Traceback" $W/glm-run/syncdiag.log 2>/dev/null; do sleep 20; done
grep -q Traceback $W/glm-run/syncdiag.log && { echo "BOOT FAILED"; exit 1; }
touch $W/syncsites/ARM; sleep 5
$HOME/venv/bin/python $W/bench_real.py --runs 2 --osl 1024 --docs $W/glm-dsa-tpu 2>&1 | tail -2
sleep 25
rm -f /dev/shm/home/qinghanc/work/syncsites/ARM
echo DIAG_DONE
