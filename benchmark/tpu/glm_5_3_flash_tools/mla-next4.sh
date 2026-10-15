#!/bin/bash
# After the MLA MTP retry: MLA no-spec and DFlash again with the seeded block sizes.
W=/dev/shm/home/qinghanc/work
until grep -q CHAIN_DONE $W/glm-run/mla-next3.out 2>/dev/null; do sleep 30; done
bash $W/mla-chain.sh mla_fp8_nospec_t mla_fp8_dflash_t
