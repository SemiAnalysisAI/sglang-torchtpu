#!/bin/bash
# Ported tree on main: boot MLA bf16 DFlash, real-text c1 bench, GSM8K 200, then a real-text profile.
W=/dev/shm/home/qinghanc/work
exec > $W/glm-run/port-validate.out 2>&1
. $W/tpu_env.sh
export OPENAI_API_KEY=EMPTY
cfg=mla_bf16_dflash; tag=port-$cfg
bash $W/stop_wait.sh
bash $W/run-cfg-port.sh glm_5_3_flash_eval/$cfg.py $tag
until grep -qE "fired up|Traceback" $W/glm-run/$tag.log 2>/dev/null; do sleep 20; done
grep -q Traceback $W/glm-run/$tag.log && { echo "BOOT FAILED"; grep -m3 -A20 Traceback $W/glm-run/$tag.log | tail -25; exit 1; }
cd $W
/dev/shm/home/qinghanc/venv-port/bin/python bench_real.py --runs 2 2>&1 | grep -E "^run|SUMMARY"
cd $W/glm-port/python && /dev/shm/home/qinghanc/venv-port/bin/python -m sglang.test.run_eval --eval-name gsm8k --port 30000 \
   --num-examples 200 --num-threads 8 --max-tokens 4096 --temperature 0 2>&1 | grep -iE "score|accuracy" | tail -2
cd $W && /dev/shm/home/qinghanc/venv-port/bin/python prof_real.py $W/prof-port-dflash TPU,CPU
echo PORT_DONE
