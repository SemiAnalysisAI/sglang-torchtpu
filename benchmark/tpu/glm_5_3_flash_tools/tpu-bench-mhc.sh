#!/bin/bash
# Usage: tpu-bench-mhc.sh <tree> <venv> -- stop the server, then sweep the mHC block size.
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
export HOME=$H
exec > $W/glm-run/bench-mhc.out 2>&1
. $W/tpu_env.sh
bash $W/stop_wait.sh
$H/$2/bin/python $W/bench_mhc_oldnew.py $W/glm_mhc_old.py $W/glm_mhc_new.py
echo MHC_DONE
