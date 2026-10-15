#!/bin/bash
# MLA fp8 prefill (MIXED slot) block sweep at GLM-5.3-Flash per-rank shape, one chip, server down.
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
export HOME=$H
exec > $W/glm-run/tune-mla.out 2>&1
. $W/tpu_env.sh
bash $W/stop_wait.sh
ENV="env -u TPU_WORKER_HOSTNAMES -u TPU_PROCESS_ADDRESSES TPU_WORKER_ID=0 TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0 TPU_TOPOLOGY=1x1x1 TPU_PROCESS_BOUNDS=1,1,1"
$ENV $H/venv-port-b/bin/python $W/tune_mla_fp8_flash.py --heads 8 --page-size 128 --decode-bs 1 --kv-len 2048 --dbs 1 \
  --mixed-mnt 8192 --mixed-kv-len 8192 --bkv 8,16,32,64 --bq-mixed 128,256,512,1024 --iters 10
echo TUNE_DONE
