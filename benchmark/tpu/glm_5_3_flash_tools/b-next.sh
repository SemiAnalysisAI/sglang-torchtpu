#!/bin/bash
W=/dev/shm/home/qinghanc/work
until grep -q CHAIN_DONE $W/glm-run/cb-mla.out 2>/dev/null; do sleep 30; done
bash $W/prof-c16.sh cb_mla_dflash
