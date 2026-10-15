#!/bin/bash
# Boot the op-site diagnostic MTP server, arm after warmup, drive c1 decode, report.
W=/dev/shm/home/qinghanc/work
exec > $W/glm-run/ops-diag.out 2>&1
. $W/tpu_env.sh
bash $W/stop_wait.sh
bash $W/run-cfg.sh glm_5_3_flash_eval/mla_fp8_mtp516_ops.py opsdiag
until grep -qE "fired up|Traceback" $W/glm-run/opsdiag.log 2>/dev/null; do sleep 20; done
grep -q Traceback $W/glm-run/opsdiag.log && { echo "BOOT FAILED"; exit 1; }
touch $W/opsites/ARM; sleep 5
$HOME/venv/bin/python $W/bench_real.py --runs 1 --osl 1024 --docs $W/glm-dsa-tpu 2>&1 | tail -1
sleep 25
echo OPS_DONE
