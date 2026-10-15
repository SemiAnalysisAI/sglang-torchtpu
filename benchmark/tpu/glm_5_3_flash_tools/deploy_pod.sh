#!/bin/bash
# Usage: deploy_pod.sh <pod> [namespace] -- venv + the full glm-dsa-tpu worktree source + scripts.
set -euo pipefail
C=gke_sglang-torchtpu_us-central1-c_tpu-cluster; P=$1; NS=${2:-tpu-dev-wenhao}
# RUN_AS=root for pods whose container user is root (e.g. tpuq); default su to qinghanc.
RUN_AS=${RUN_AS:-qinghanc}
S=$(cd "$(dirname "$0")" && pwd); W=/Users/qinghan.chen/Desktop/sglang-jax-workdir/glm-dsa-tpu
K="kubectl --context $C -n $NS exec -i $P -c ${CONTAINER:-main} --"
as_user() { if [[ $RUN_AS == root ]]; then echo "sh -c"; else echo "su $RUN_AS -c"; fi; }
$K $(as_user) 'mkdir -p /dev/shm/home/qinghanc && tar xzf - -C /dev/shm/home/qinghanc' < $S/../glm-dsa-tpu/ci-tp16/base.tgz
( cd $W && { /usr/bin/git ls-files python benchmark/tpu test/registered; /usr/bin/git ls-files -o --exclude-standard; } | sort -u \
  | grep -E '^(python|benchmark/tpu|test)/' | while read f; do [[ -e $f ]] && echo "$f"; done \
  | COPYFILE_DISABLE=1 tar cf - --no-xattrs -T - ) | $K $(as_user) 'tar xf - -C /dev/shm/home/qinghanc/work/glm-dsa-tpu'
( cd $S && COPYFILE_DISABLE=1 tar cf - --no-xattrs run-glm53-flash.sh bench_8k1k.py ) \
  | $K $(as_user) 'cd /dev/shm/home/qinghanc/work && tar xf - && mkdir -p glm-run ../tmp ../tpu_logs ../.cache && echo "deployed $(hostname)"'
