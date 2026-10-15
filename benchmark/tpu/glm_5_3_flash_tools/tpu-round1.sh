#!/bin/bash
# Microbenchmarks, the few-token MoE test, then DFlash A (fix) vs B (fix + few-token MoE).
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
exec > $W/glm-run/tpu-round1.out 2>&1
. $W/tpu_env.sh
export OPENAI_API_KEY=EMPTY
bash $W/stop_wait.sh
echo "== all-reduce microbenchmark"
$H/venv-port/bin/python $W/bench_allreduce.py 2>&1 | grep -E "devices|T=|Error|error" | tail -8
echo "== mHC seam microbenchmark"
$H/venv-port/bin/python $W/bench_mhc_seam.py 2>&1 | grep -E "T=|Error|error" | tail -6
echo "== few-token MoE unit test (tree B)"
cd $W/glm-port-b && $H/venv-port-b/bin/python -m unittest test/registered/tpu/unit/test_moe_stream_bs1.py 2>&1 | grep -vE "^I0|^W0|absl|Initializ|renamed|Registered|Device type" | tail -15
for side in A B; do
  if [ $side = A ]; then tree=glm-port; venv=venv-port; else tree=glm-port-b; venv=venv-port-b; fi
  tag=r1-$side
  cd $W
  bash $W/stop_wait.sh
  bash $W/run-cfg-port.sh glm_5_3_flash_eval/mla_bf16_dflash.py $tag $tree $venv
  until grep -qE "fired up|Traceback" $W/glm-run/$tag.log 2>/dev/null; do sleep 20; done
  grep -q Traceback $W/glm-run/$tag.log && { echo "$side BOOT FAILED"; grep -m1 -B2 -A25 Traceback $W/glm-run/$tag.log | tail -30; continue; }
  echo "== DFlash $side ($tree)"
  $H/$venv/bin/python bench_real.py --runs 3 2>&1 | grep -E "^run|SUMMARY"
  if [ $side = B ]; then
    cd $W/$tree/python && $H/$venv/bin/python -m sglang.test.run_eval --eval-name gsm8k --port 30000 \
      --num-examples 200 --num-threads 8 --max-tokens 4096 --temperature 0 2>&1 | grep -iE "gsm8k_score" | tail -1
    cd $W
  fi
  $H/$venv/bin/python prof_real.py $W/prof-r1-$side TPU,CPU
done
echo ROUND1_DONE
