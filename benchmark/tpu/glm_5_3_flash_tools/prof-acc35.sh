#!/bin/bash
# Stop whatever runs, boot MLA fp8 MTP 5/1/6 at simulated accept 3.5, capture decode rounds, bench 8k1k c1.
W=/dev/shm/home/qinghanc/work
exec > $W/glm-run/prof-acc35.out 2>&1
bash $W/boot-prof.sh prof_mla_fp8_mtp516_acc35
bash $W/bench-cb.sh prof_acc35 1
echo ALL_DONE
