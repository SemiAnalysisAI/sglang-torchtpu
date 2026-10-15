#!/bin/bash
# Usage: tpu-chain-aime.sh <bench tag> <config> <tree> <venv> <aime tag> -- wait for the real-acceptance bench, then AIME26.
W=/dev/shm/home/qinghanc/work
until grep -q REAL_DONE $W/glm-run/real-$1.out 2>/dev/null; do sleep 30; done
bash $W/tpu-aime.sh $2 $3 $4 $5
