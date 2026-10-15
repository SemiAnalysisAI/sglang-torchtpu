#!/bin/bash
# Usage: prof_mtp.sh <out dir> -- capture ~0.6 s of steady decode rounds (TPU+CPU) on the running server.
cd /dev/shm/home/qinghanc/work
. ./tpu_env.sh
$HOME/venv/bin/python prof_round.py "$1" TPU,CPU
find "$1" -name "*.xplane.pb" | head -3
