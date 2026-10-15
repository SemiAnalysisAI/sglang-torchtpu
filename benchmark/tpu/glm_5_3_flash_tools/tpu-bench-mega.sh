#!/bin/bash
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
export HOME=$H
exec > $W/glm-run/bench-mega.out 2>&1
. $W/tpu_env.sh
bash $W/stop_wait.sh
$H/venv-port/bin/python $W/bench_mega_kda.py $W/mega_kda_ref 8192 4096 1024
echo MEGA_DONE
