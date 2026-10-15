#!/bin/bash
# After the MLA DFlash retry: retry MLA + fp8 + MTP 5/1/6.
W=/dev/shm/home/qinghanc/work
until grep -q CHAIN_DONE $W/glm-run/mla-next2.out 2>/dev/null; do sleep 30; done
bash $W/mla-chain.sh mla_fp8_mtp516
