#!/bin/bash
# mHC lane-layout Sinkhorn: unit test + seam microbenchmark on tree C, then DFlash B (fix+MoE) vs C (+mHC).
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
exec > $W/glm-run/tpu-round2.out 2>&1
. $W/tpu_env.sh
export OPENAI_API_KEY=EMPTY
bash $W/stop_wait.sh
echo "== mHC kernel unit test (tree C)"
cd $W/glm-port && $H/venv-port/bin/python -m unittest test/registered/tpu/unit/test_glm_mhc_kernel.py 2>&1 | grep -E "Ran|OK|FAIL|Error|error|relative" | tail -12
echo "== mHC seam microbenchmark (tree C)"
cd $W && $H/venv-port/bin/python $W/bench_mhc_seam.py 2>&1 | grep -vE "^I0|^W0|absl|Initializ|renamed|Registered|Device type|UserWarning|warnings.warn" | tail -12
for side in B C; do
  if [ $side = C ]; then tree=glm-port; venv=venv-port; else tree=glm-port-b; venv=venv-port-b; fi
  tag=r2-$side
  cd $W
  bash $W/stop_wait.sh
  bash $W/run-cfg-port.sh glm_5_3_flash_eval/mla_bf16_dflash.py $tag $tree $venv
  until grep -qE "fired up|Traceback" $W/glm-run/$tag.log 2>/dev/null; do sleep 20; done
  grep -q Traceback $W/glm-run/$tag.log && { echo "$side BOOT FAILED"; grep -m1 -B2 -A25 Traceback $W/glm-run/$tag.log | tail -30; continue; }
  echo "== DFlash $side ($tree)"
  $H/$venv/bin/python bench_real.py --runs 3 2>&1 | grep -E "^run|SUMMARY"
  if [ $side = C ]; then
    cd $W/$tree/python && $H/$venv/bin/python -m sglang.test.run_eval --eval-name gsm8k --port 30000 \
      --num-examples 200 --num-threads 8 --max-tokens 4096 --temperature 0 2>&1 | grep -iE "gsm8k_score" | tail -1
    cd $W
  fi
  $H/$venv/bin/python prof_real.py $W/prof-r2-$side TPU,CPU
done
echo ROUND2_DONE
