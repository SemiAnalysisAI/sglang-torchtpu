#!/bin/bash
# Usage: tpu-boot-only.sh <config under glm_5_3_flash_eval/> <tag> [tree=glm-port] [venv=venv-port]
# Boot once and report exactly one of: BOOT_OK, BOOT_FAILED (first error), BOOT_DIED (server gone
# without a traceback, e.g. SIGTERM/OOM-kill), BOOT_STALLED (log silent for 20 min). Then BOOT_DONE.
W=/dev/shm/home/qinghanc/work; export HOME=/dev/shm/home/qinghanc; L=$W/glm-run/$2.log
exec > $W/glm-run/boot-$2.out 2>&1
. $W/tpu_env.sh
bash $W/stop_wait.sh
bash $W/run-cfg-port.sh glm_5_3_flash_eval/$1.py $2 ${3:-glm-port} ${4:-venv-port}
alive() { grep -lqsE "sglang.launch_server" /proc/[0-9]*/cmdline 2>/dev/null; }
sleep 30
while true; do
  if grep -aqE "fired up" $L; then echo BOOT_OK; break; fi
  if grep -aqE "Traceback|RuntimeUnexpectedCoreHalt" $L; then
    echo BOOT_FAILED; grep -a -m1 -E "RuntimeError|Error:" $L | cut -c1-300; grep -a -m2 "halted" $L | cut -c1-200; break
  fi
  if ! alive; then echo BOOT_DIED; grep -a -m2 -E "SIGTERM|SIGKILL|Killed" $L | cut -c1-200; tail -n 3 $W/glm-run/memguard.log 2>/dev/null; break; fi
  if [ $(( $(date +%s) - $(stat -c %Y $L) )) -gt 1200 ]; then echo BOOT_STALLED; tail -c 300 $L; break; fi
  sleep 15
done
echo BOOT_DONE
