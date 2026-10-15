#!/bin/bash
# Slice 1: 8k1k c1 on DSA (MTP accept-3.5, DFlash accept-3, DFlash real acceptance), then AIME26 DFlash+DSA.
W=/dev/shm/home/qinghanc/work
export HOME=/dev/shm/home/qinghanc
bash $W/tpu-mtp-run.sh prof_dsa_mtp516_acc35 glm-port-b venv-port-b dsa-mtp-acc35
bash $W/tpu-mtp-run.sh cb_dsa_dflash glm-port-b venv-port-b dsa-dflash-acc3
bash $W/tpu-real-run.sh real_dsa_dflash glm-port-b venv-port-b dsa-dflash-real
echo DSA_8K1K_DONE > $W/glm-run/dsa-8k1k.done
bash $W/tpu-dsa-s1.sh
