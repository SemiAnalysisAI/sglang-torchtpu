#!/bin/bash
# Slice 1: finish MTP AIME26 avg@4 (10 left), then DFlash avg@4 (repeats 0-3, resumes), both at concurrency 8.
W=/dev/shm/home/qinghanc/work
bash $W/tpu-aime.sh aime_mtp516_330k glm-port-b venv-port-b aime-mtp 8 0,1,2,3
bash $W/tpu-aime.sh aime_dflash_330k glm-port-b venv-port-b aime-dflash 8 0,1,2,3
