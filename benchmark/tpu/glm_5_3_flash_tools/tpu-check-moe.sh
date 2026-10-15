#!/bin/bash
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
export HOME=$H
exec > $W/glm-run/check-moe.out 2>&1
. $W/tpu_env.sh
bash $W/stop_wait.sh
$H/venv-port/bin/python $W/check_moe_oldnew.py $W/$1 $W/fused_moe_op_old.py
echo CHECK_DONE
