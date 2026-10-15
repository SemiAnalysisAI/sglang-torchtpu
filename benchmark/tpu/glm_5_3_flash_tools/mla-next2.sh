#!/bin/bash
# After the MLA MTP run: retry MLA + fp8 + DFlash.
W=/dev/shm/home/qinghanc/work
until grep -q CHAIN_DONE $W/glm-run/mla-next.out 2>/dev/null; do sleep 30; done
bash $W/mla-chain.sh mla_fp8_dflash
