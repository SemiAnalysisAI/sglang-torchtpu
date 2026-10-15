#!/bin/bash
# New pod: MTP 5/1/6 on tree D (dense MLA) for comparison with DFlash, then GLM-5.3-Flash DSA on the DSA tip.
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
exec > $W/glm-run/tpu-round5.out 2>&1
. $W/tpu_env.sh
export OPENAI_API_KEY=EMPTY
setsid nohup bash $W/memguard.sh 820 > /dev/null 2>&1 &
boot() {  # boot <config> <tag> <tree> <venv>
  bash $W/stop_wait.sh
  bash $W/run-cfg-port.sh glm_5_3_flash_eval/$1.py $2 $3 $4
  until grep -qE "fired up|Traceback" $W/glm-run/$2.log 2>/dev/null; do sleep 20; done
  if grep -q Traceback $W/glm-run/$2.log; then
    echo "$2 BOOT FAILED"; grep -m1 -B5 -A40 Traceback $W/glm-run/$2.log | tail -45; return 1
  fi
}
gsm() {  # gsm <tree> <venv>
  (cd $W/$1/python && $H/$2/bin/python -m sglang.test.run_eval --eval-name gsm8k --port 30000 \
    --num-examples 200 --num-threads 8 --max-tokens 4096 --temperature 0 2>&1 | grep -iE "gsm8k_score" | tail -1)
}
echo "== MTP 5/1/6, MLA bf16, dense (tree D)"
if boot mla_bf16_mtp516 r5-mtp glm-port venv-port; then
  cd $W && $H/venv-port/bin/python bench_real.py --runs 3 2>&1 | grep -E "^run|SUMMARY"
  gsm glm-port venv-port
  cd $W && $H/venv-port/bin/python prof_real.py $W/prof-r5-mtp TPU,CPU
fi
echo "== kpool indexer step unit test (DSA tip)"
cd $W/glm-port-b && $H/venv-port-b/bin/python -m unittest test/registered/unit/hardware_backend/tpu/test_dsa_kpool_topk.py 2>&1 | grep -E "Ran|OK|FAIL|Error|error" | tail -6
echo "== GLM-5.3-Flash DSA, DFlash (DSA tip)"
if boot dsa_dflash r5-dsa glm-port-b venv-port-b; then
  cd $W && $H/venv-port-b/bin/python dsa_sample.py 2>&1 | tail -8
  gsm glm-port-b venv-port-b
  cd $W && $H/venv-port-b/bin/python bench_real.py --runs 3 2>&1 | grep -E "^run|SUMMARY"
  $H/venv-port-b/bin/python prof_real.py $W/prof-r5-dsa TPU,CPU
fi
echo ROUND5_DONE
