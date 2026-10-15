#!/bin/bash
# Usage: tpu-chain-cprofile.sh <tree> <venv> -- after the running host-timer run, rerun it with cProfile on TP0.
W=/dev/shm/home/qinghanc/work
until grep -q HT_DONE $W/glm-run/tpu-mtp-hosttimer.out 2>/dev/null; do sleep 15; done
cp $W/glm-run/mtp-hosttimer.log $W/glm-run/mtp-hosttimer-skipguard.log
cp $W/host_timer_eagle.py.next $W/host_timer_eagle.py; cp $W/tpu-mtp-hosttimer.sh.next $W/tpu-mtp-hosttimer.sh
QH_CPROFILE=1 bash $W/tpu-mtp-hosttimer.sh $1 $2
