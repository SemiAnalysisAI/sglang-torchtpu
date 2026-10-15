#!/bin/bash
# Usage: devrun.sh <script.py> [args...] -- sync python tree + script to worker 0-2 and run on its TPU.
set -euo pipefail
C=gke_sglang-torchtpu_us-central1-c_tpu-cluster; P=${POD:-tpu-dev-k3-workers-0-2-tpgpz}; W=/Users/qinghan.chen/Desktop/sglang-jax-workdir/glm-dsa-tpu
S=$(cd "$(dirname "$0")" && pwd)
K=(kubectl --context $C -n tpu-dev-wenhao exec -i "$P" --)
( cd $W && /usr/bin/git ls-files python | COPYFILE_DISABLE=1 tar cf - --no-xattrs -T - ; ) | "${K[@]}" su qinghanc -c 'mkdir -p /dev/shm/home/qinghanc/dev && tar xf - -C /dev/shm/home/qinghanc/dev'
( cd $W && /usr/bin/git ls-files -o --exclude-standard python | COPYFILE_DISABLE=1 tar cf - --no-xattrs -T - ; ) | "${K[@]}" su qinghanc -c 'tar xf - -C /dev/shm/home/qinghanc/dev'
( cd $S && COPYFILE_DISABLE=1 tar cf - --no-xattrs "$1" tpu_env.sh ) | "${K[@]}" su qinghanc -c 'tar xf - -C /dev/shm/home/qinghanc/dev'
"${K[@]}" su qinghanc -c "cd /dev/shm/home/qinghanc/dev && . ./tpu_env.sh 2>/dev/null; export PYTHONPATH=/dev/shm/home/qinghanc/dev/python; timeout 1200 python3 $* 2>&1 | grep -vE 'Could not open|log file|TPU backend initialization|warnings.warn'"
