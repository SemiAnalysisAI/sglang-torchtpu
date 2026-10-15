#!/bin/bash
# Usage: aime-run.sh <eval tag> -- detached AIME26 on the deployed (pinned) tree; log at glm-run/aime-<tag>.out.
W=/dev/shm/home/qinghanc/work
exec > $W/glm-run/aime-$1.out 2>&1
echo "pinned: glm-flash-mla-pin-20261013 690e5f6831"
bash $W/eval-chain.sh $1
