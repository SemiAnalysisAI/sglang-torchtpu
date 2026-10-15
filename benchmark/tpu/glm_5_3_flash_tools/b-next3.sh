#!/bin/bash
W=/dev/shm/home/qinghanc/work
until grep -q CHAIN_DONE $W/glm-run/b-next2.out 2>/dev/null; do sleep 30; done
bash $W/cb-chain.sh mla_dflash_ep8
