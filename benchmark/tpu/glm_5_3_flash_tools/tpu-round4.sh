#!/bin/bash
# GLM-5.3-Flash DSA (kpool) on TPU: unit test, boot, samples, GSM8K, real-text bench, profile.
W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
exec > $W/glm-run/tpu-round4.out 2>&1
. $W/tpu_env.sh
export OPENAI_API_KEY=EMPTY
bash $W/stop_wait.sh
echo "== kpool indexer step unit test (TPU)"
cd $W/glm-port && $H/venv-port/bin/python -m unittest test/registered/unit/hardware_backend/tpu/test_dsa_kpool_topk.py 2>&1 | grep -E "Ran|OK|FAIL|Error|error" | tail -6
cd $W
tag=r4-dsa
bash $W/run-cfg-port.sh glm_5_3_flash_eval/dsa_dflash.py $tag glm-port venv-port
until grep -qE "fired up|Traceback" $W/glm-run/$tag.log 2>/dev/null; do sleep 20; done
grep -q Traceback $W/glm-run/$tag.log && { echo "DSA BOOT FAILED"; grep -m1 -B5 -A40 Traceback $W/glm-run/$tag.log | tail -45; echo ROUND4_DONE; exit 1; }
echo "== samples"
$H/venv-port/bin/python dsa_sample.py 2>&1 | tail -8
echo "== GSM8K"
cd $W/glm-port/python && $H/venv-port/bin/python -m sglang.test.run_eval --eval-name gsm8k --port 30000 \
  --num-examples 200 --num-threads 8 --max-tokens 4096 --temperature 0 2>&1 | grep -iE "gsm8k_score" | tail -1
cd $W
echo "== DFlash DSA real-text c1"
$H/venv-port/bin/python bench_real.py --runs 3 2>&1 | grep -E "^run|SUMMARY"
$H/venv-port/bin/python prof_real.py $W/prof-r4-dsa TPU,CPU
echo ROUND4_DONE
