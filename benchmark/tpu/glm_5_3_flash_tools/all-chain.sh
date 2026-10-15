#!/bin/bash
# Queue: cookbook-parity (accept pinned at 3) DFlash and MTP 5/1/6, then AIME26 DFlash and no-spec.
W=/dev/shm/home/qinghanc/work
bash $W/cb-chain.sh dflash mtp516
bash $W/eval-chain.sh dflash nospec
echo ALL_DONE
