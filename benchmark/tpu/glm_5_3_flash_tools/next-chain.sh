#!/bin/bash
# After all-chain: MTP c16 with the full batch list, then the AIME26 DFlash retry.
W=/dev/shm/home/qinghanc/work
until grep -q ALL_DONE $W/glm-run/all-chain.out; do sleep 30; done
bash $W/cb-chain.sh mtp516full
bash $W/eval-chain.sh dflash
echo NEXT_DONE
