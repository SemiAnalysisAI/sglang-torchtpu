#!/bin/bash
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
export HOME=$H
exec > $W/glm-run/check-mega.out 2>&1
. $W/tpu_env.sh
bash $W/stop_wait.sh
$H/venv-port/bin/python $W/check_mega_kda.py $W/$1
echo CHECK_DONE
