#!/bin/bash
# Boot the scopes diagnostic MTP server, arm scopes after warmup, capture a TPU+CPU round profile.
W=/dev/shm/home/qinghanc/work
exec > $W/glm-run/scope-prof.out 2>&1
. $W/tpu_env.sh
bash $W/stop_wait.sh
bash $W/run-cfg.sh glm_5_3_flash_eval/mla_fp8_mtp516_scopes.py scopes
until grep -qE "fired up|Traceback" $W/glm-run/scopes.log 2>/dev/null; do sleep 20; done
grep -q Traceback $W/glm-run/scopes.log && { echo "BOOT FAILED"; exit 1; }
touch $W/scopes/ARM; sleep 8
grep "SCOPES armed" $W/glm-run/scopes.log | head -1
cd $W && $HOME/venv/bin/python prof_round.py $W/prof-scopes TPU,CPU && find $W/prof-scopes -name "*.xplane.pb" | grep TP-0
echo SCOPE_DONE
