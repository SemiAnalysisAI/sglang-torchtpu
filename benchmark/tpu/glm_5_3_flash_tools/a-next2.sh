#!/bin/bash
W=/dev/shm/home/qinghanc/work
until grep -q CHAIN_DONE $W/glm-run/a-next.out 2>/dev/null; do sleep 30; done
bash $W/a-round.sh mla_fp8_mtp516 mtp-r3
