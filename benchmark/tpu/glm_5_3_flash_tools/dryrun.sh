#!/bin/bash
# Usage: dryrun.sh <config under recipe/configs> -- print the resolved launch command.
cd /dev/shm/home/qinghanc/work
. ./tpu_env.sh
cd glm-dsa-tpu
exec $HOME/venv/bin/python -m benchmark.tpu.launch_server --config "recipe/configs/$1" --dry-run
