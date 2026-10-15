#!/bin/bash
# Usage: pyrun.sh <python args...> -- run the pod venv python from the worktree's python/ dir with the TPU env.
cd /dev/shm/home/qinghanc/work
. ./tpu_env.sh
cd glm-dsa-tpu/python
exec $HOME/venv/bin/python "$@"
