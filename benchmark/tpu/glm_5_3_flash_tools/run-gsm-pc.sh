#!/bin/bash
# x5kk: GSM8K (200, greedy, 8 threads) on the prefix_cache recipe, radix on then radix off.
S=$(cd "$(dirname "$0")" && pwd); R=$S/../../glm-dsa-tpu-rebase; W=/dev/shm/home/qinghanc/work; H=/dev/shm/home/qinghanc
C=gke_sglang-torchtpu_us-central1-c_tpu-cluster; P=tpuq-qinghan-4d14-w-0-0-m27gm; T=glm-port-b; V=venv-port-b
k() { kubectl --context $C -n tpuq exec -i "$P" -c main -- "$@"; }
for f in $(cd $R && git diff --name-only 369d483ac1 HEAD -- python sglang_tpu benchmark/tpu/recipe/configs/glm_5_3_flash); do k tee $W/$T/$f < $R/$f > /dev/null; done
for f in pc_recipe.py pc_recipe_off.py; do k tee $W/$T/benchmark/tpu/recipe/configs/glm_5_3_flash_eval/$f < $S/eval_configs/$f > /dev/null; done
k tee $W/tpu-boot-only.sh < $S/tpu-boot-only.sh > /dev/null
for cfg in pc_recipe pc_recipe_off; do
  k bash $W/stop_wait.sh < /dev/null
  k env HOME=$H setsid -f bash $W/tpu-boot-only.sh $cfg gsm-$cfg $T $V < /dev/null > /dev/null 2>&1
  until k grep -q BOOT_DONE $W/glm-run/boot-gsm-$cfg.out < /dev/null 2>/dev/null; do sleep 30; done
  echo "$(date +%T) $cfg: $(k grep -av PrivateUse1 $W/glm-run/boot-gsm-$cfg.out < /dev/null | tr '\n' ' ' | cut -c1-120)"
  k bash -c "cd $W/$T/python && OPENAI_API_KEY=EMPTY $H/$V/bin/python -m sglang.test.run_eval --eval-name gsm8k --port 30000 --num-examples 200 --num-threads 8 --max-tokens 4096 --temperature 0 2>&1 | grep -iE 'score|Error' | tail -2" < /dev/null | sed "s/^/$cfg: /"
  k bash -c "grep -a 'cache hit rate\|#cached-token' $W/glm-run/gsm-$cfg.log | tail -1 | cut -c1-200" < /dev/null | sed "s/^/$cfg: /"
done
echo GSM_PC_DONE
