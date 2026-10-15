#!/bin/bash
# AIME26 at a 300k context / 290k-token budget on the deployed (pinned) tree.
W=/dev/shm/home/qinghanc/work
exec > $W/glm-run/aime-$1.out 2>&1
echo "pinned: glm-flash-mla-pin-20261013 690e5f6831; max_tokens 290000"
export AIME_MAX_TOKENS=290000
bash $W/eval-chain.sh $1
