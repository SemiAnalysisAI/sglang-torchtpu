#!/bin/bash
# After the first MLA chain: MLA + fp8 + MTP 5/1/6.
W=/dev/shm/home/qinghanc/work
until grep -q CHAIN_DONE $W/glm-run/mla-chain.out; do sleep 30; done
bash $W/mla-chain.sh mla_fp8_mtp516
