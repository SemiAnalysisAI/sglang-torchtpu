#!/bin/bash
# Usage: real-chain.sh <config:tag|-:tag> ... -- (re)start each recipe ("-" reuses the running server), run bench_real.py.
W=/dev/shm/home/qinghanc/work
. $W/tpu_env.sh
for pair in "$@"; do
  cfg=${pair%%:*}; tag=${pair##*:}
  if [ "$cfg" != "-" ]; then
    bash $W/stop_sglang.sh; sleep 15
    bash $W/run-cfg.sh glm_5_3_flash/$cfg $tag
    until grep -qE "fired up|Traceback" $W/glm-run/$tag.log 2>/dev/null; do sleep 20; done
    grep -q Traceback $W/glm-run/$tag.log && { echo "$tag BOOT FAILED"; continue; }
  fi
  echo "== $tag"; $HOME/venv/bin/python $W/bench_real.py --runs 3 --osl 1024 --docs $W/glm-dsa-tpu 2>&1 | tail -6
done
echo CHAIN_DONE
