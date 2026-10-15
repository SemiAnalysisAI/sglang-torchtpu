#!/bin/bash
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
export HOME=$H
exec > $W/glm-run/bench-ar.out 2>&1
. $W/tpu_env.sh
bash $W/stop_wait.sh
$H/venv-port/bin/python $W/bench_allreduce_big.py 1024 2048 4096 8192
echo AR_DONE
