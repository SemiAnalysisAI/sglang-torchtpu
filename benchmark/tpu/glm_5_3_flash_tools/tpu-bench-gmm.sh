#!/bin/bash
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
export HOME=$H
exec > $W/glm-run/bench-gmm.out 2>&1
. $W/tpu_env.sh
bash $W/stop_wait.sh
$H/venv-port/bin/python $W/bench_gmm_tiles.py $W/$1
echo GMM_DONE
