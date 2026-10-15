#!/bin/bash
# Usage: dump-ab.sh <cfgA> <cfgB> -- greedy dumps from two no-spec servers on the same prompts.
W=/dev/shm/home/qinghanc/work
. $W/tpu_env.sh
for cfg in "$@"; do
  bash $W/stop_wait.sh
  bash $W/run-cfg.sh $cfg dump-$(basename $cfg .py)
  until grep -qE "fired up|Traceback" $W/glm-run/dump-$(basename $cfg .py).log 2>/dev/null; do sleep 20; done
  cd $W && $HOME/venv/bin/python gen_dump.py $W/glm-run/dump-$(basename $cfg .py).json --osl 512 2>&1 | tail -1
done
cd $W && $HOME/venv/bin/python cmp_dump.py $W/glm-run/dump-$(basename $1 .py).json $W/glm-run/dump-$(basename $2 .py).json
echo DUMP_DONE
