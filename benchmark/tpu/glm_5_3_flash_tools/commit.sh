#!/bin/bash
# usage: commit.sh <msgfile> <paths...> -- stage, commit; if a formatter hook rewrote files, re-stage and commit again (never --amend).
set -uo pipefail
msg=$1; shift
cd /Users/qinghan.chen/Desktop/sglang-jax-workdir/glm-dsa-tpu
/usr/bin/git add "$@"
SKIP=check-registered-tests /usr/bin/git commit -q -F "$msg" >/dev/null 2>&1 || { /usr/bin/git add "$@"; SKIP=check-registered-tests /usr/bin/git commit -q -F "$msg" 2>&1 | grep -E "Failed|ERROR"; }
/usr/bin/git log --oneline -1; /usr/bin/git status --short | head -5
