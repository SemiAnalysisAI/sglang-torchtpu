#!/bin/bash
# DFlash C (fix+MoE+mHC) vs D (+ fewer per-round launches): bench, profile, GSM8K on D.
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
exec > $W/glm-run/tpu-round3.out 2>&1
. $W/tpu_env.sh
export OPENAI_API_KEY=EMPTY
bash $W/stop_wait.sh
for side in D; do
  if [ $side = C ]; then tree=glm-port; venv=venv-port; else tree=glm-port-b; venv=venv-port-b; fi
  tag=r3-$side
  cd $W
  bash $W/stop_wait.sh
  bash $W/run-cfg-port.sh glm_5_3_flash_eval/mla_bf16_dflash.py $tag $tree $venv
  until grep -qE "fired up|Traceback" $W/glm-run/$tag.log 2>/dev/null; do sleep 20; done
  grep -q Traceback $W/glm-run/$tag.log && { echo "$side BOOT FAILED"; grep -m1 -B2 -A25 Traceback $W/glm-run/$tag.log | tail -30; continue; }
  echo "== DFlash $side ($tree)"
  $H/$venv/bin/python bench_real.py --runs 3 2>&1 | grep -E "^run|SUMMARY"
  if [ $side = D ]; then
    cd $W/$tree/python && $H/$venv/bin/python -m sglang.test.run_eval --eval-name gsm8k --port 30000 \
      --num-examples 200 --num-threads 8 --max-tokens 4096 --temperature 0 2>&1 | grep -iE "gsm8k_score" | tail -1
    cd $W
  fi
  $H/$venv/bin/python prof_real.py $W/prof-r3-$side TPU,CPU
done
echo ROUND3_DONE
