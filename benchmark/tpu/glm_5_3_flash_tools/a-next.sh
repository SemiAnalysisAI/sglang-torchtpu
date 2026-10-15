#!/bin/bash
W=/dev/shm/home/qinghanc/work
until grep -qE "ROUND_DONE|BOOT FAILED" $W/glm-run/a-round-r2.out 2>/dev/null; do sleep 30; done
bash $W/mla-chain.sh mla_fp8_dflash_t
